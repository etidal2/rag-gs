from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from rag_gs.core.config import Config
from rag_gs.core.io import read_json, read_jsonl, write_jsonl
from rag_gs.core.logging import setup_logger
from rag_gs.core.manifest import write_stage_manifest
from rag_gs.workspace import RunPaths, default_run_id, q_segment


LISTWISE_BATCH = 5
EPSILON_INFO = 1e-6

from rag_gs.plugins.judges.prompts import RANK_SYSTEM_PROMPT as SYSTEM_PROMPT


def item_sigma(info_value: float) -> float:
    return 1.0 / math.sqrt(max(info_value, EPSILON_INFO))


def margin_clear(winner: "ItemState", loser: "ItemState", z_value: float) -> bool:
    w_sigma = item_sigma(winner.info)
    l_sigma = item_sigma(loser.info)
    w_lcb = winner.score - z_value * w_sigma
    l_ucb = loser.score + z_value * l_sigma
    return w_lcb > l_ucb


def iter_pairs(order: Sequence[str]) -> Iterable[Tuple[str, str]]:
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            yield order[i], order[j]


def dag_has_path(adj: Dict[str, List[str]], start: str, target: str) -> bool:
    if start == target:
        return True
    stack = [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj.get(node, []))
    return False


def has_two_disjoint_paths(adj: Dict[str, List[str]], start: str, target: str, max_depth: int = 10) -> bool:
    if start == target:
        return False
    paths: List[set] = []
    stack: List[Tuple[str, List[str]]] = [(start, [start])]
    while stack and len(paths) < 8:
        node, path = stack.pop()
        if len(path) > max_depth:
            continue
        for nxt in adj.get(node, []):
            if nxt in path:
                continue
            new_path = path + [nxt]
            if nxt == target:
                if len(new_path) >= 3:
                    internal_nodes = set(new_path[1:-1])
                    paths.append(internal_nodes)
            else:
                stack.append((nxt, new_path))
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if paths[i].isdisjoint(paths[j]):
                return True
    return False


def topological_order(item_ids: Sequence[str], scores: Dict[str, float], edges: Sequence[Tuple[str, str]]) -> List[str]:
    indeg: Dict[str, int] = {x: 0 for x in item_ids}
    adj: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        if v not in indeg:
            continue
        adj[u].append(v)
        indeg[v] += 1
    frontier: List[Tuple[float, str]] = []
    for x, d in indeg.items():
        if d == 0:
            frontier.append((-scores.get(x, 0.0), x))
    frontier.sort()
    order: List[str] = []
    while frontier:
        s, x = frontier.pop(0)
        order.append(x)
        for y in adj.get(x, []):
            indeg[y] -= 1
            if indeg[y] == 0:
                frontier.append((-scores.get(y, 0.0), y))
        frontier.sort()
    if len(order) != len(item_ids):
        raise RuntimeError("Cycle detected in DAG")
    return order


@dataclass
class ItemState:
    item_id: str
    doc_id: str
    attributes: Dict[str, Any]
    score: float
    exposures: int = 0
    info: float = EPSILON_INFO


@dataclass
class RankingState:
    iteration: int
    stable_streak: int
    random_seed: int
    items: Dict[str, ItemState]
    locked_edges: List[Tuple[str, str]] = field(default_factory=list)
    pending_counts: Dict[str, int] = field(default_factory=dict)
    top20_ids: List[str] = field(default_factory=list)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False


class Top20Ranker:
    def __init__(self, qid: str, question_text: str, records: List[Dict[str, Any]], cfg: Config, client: OpenAI, temp_dir: Path, run_paths: RunPaths) -> None:
        self.qid = qid
        self.question_text = question_text
        self.cfg = cfg
        self.client = client
        self.temp_dir = temp_dir
        self.segment = q_segment(qid)
        self.run_paths = run_paths
        self.stability_turns = cfg.s6.stability_turns
        self.confirmations_required = cfg.s6.confirmations_before_lock
        self.iteration_limit = cfg.s6.max_turns or None
        self.margin_z = cfg.s6.margin_z
        self.temperature = cfg.s6.temperature
        self.top_p = cfg.s6.top_p
        self.model_name = cfg.s6.model_name
        self.pl_step_size = cfg.s6.pl_step_size
        self.pl_step_min = cfg.s6.pl_step_min
        self.pl_clip = cfg.s6.pl_clip
        self.pl_recenter_every = cfg.s6.pl_recenter_every
        self.pl_decay = cfg.s6.pl_decay
        self.seed = cfg.s6.seed
        self.checkpoint_path = temp_dir / f"{self.segment}.checkpoint.json"
        self.snapshot_path = temp_dir / f"{self.segment}.top20_snapshot.jsonl"
        self.text_lookup = self._load_text_lookup()
        self.records = [self._augment_record(r) for r in records]
        self.state = self._init_state()

    def _load_text_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        merge_path = self.run_paths.s3_merge_path(self.qid)
        if merge_path.exists():
            with merge_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    doc_id = rec.get("doc_id")
                    text = rec.get("text")
                    if doc_id and text:
                        lookup[doc_id] = text
        else:
            logging.warning("Merge file missing for %s: %s", self.qid, merge_path)
        return lookup

    def _augment_record(self, r: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(r)
        if not out.get("text"):
            txt = self.text_lookup.get(out.get("doc_id")) if out.get("doc_id") else None
            if txt:
                out["text"] = txt
        if not out.get("question"):
            out["question"] = self.question_text
        return out

    def _init_state(self) -> RankingState:
        if self.checkpoint_path.exists():
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            items = {
                iid: ItemState(
                    item_id=iid,
                    doc_id=data["doc_id"],
                    attributes=data.get("attributes", {}),
                    score=float(data["score"]),
                    exposures=int(data.get("exposures", 0)),
                    info=float(data.get("info", EPSILON_INFO)),
                )
                for iid, data in payload["items"].items()
            }
            state = RankingState(
                iteration=int(payload["iteration"]),
                stable_streak=int(payload["stable_streak"]),
                random_seed=int(payload["random_seed"]),
                items=items,
                locked_edges=[tuple(e) for e in payload.get("locked_edges", [])],
                pending_counts=dict(payload.get("pending_counts", {})),
                top20_ids=list(payload.get("top20_ids", [])),
                hyperparams=dict(payload.get("hyperparams", {})),
                finished=bool(payload.get("finished", False)),
            )
            # Sync local knobs from checkpoint hyperparams if present
            hp = state.hyperparams
            if "margin_z" in hp:
                try:
                    self.margin_z = float(hp["margin_z"])
                except Exception:
                    pass
            if "temperature" in hp:
                self.temperature = hp["temperature"]
            if "top_p" in hp:
                self.top_p = hp["top_p"]
            if "pl_step_size" in hp:
                try:
                    self.pl_step_size = float(hp["pl_step_size"])
                except Exception:
                    pass
            if "pl_step_min" in hp:
                try:
                    self.pl_step_min = float(hp["pl_step_min"])
                except Exception:
                    pass
            if "pl_clip" in hp:
                try:
                    self.pl_clip = float(hp["pl_clip"])
                except Exception:
                    pass
            if "pl_recenter_every" in hp:
                try:
                    self.pl_recenter_every = int(hp["pl_recenter_every"])
                except Exception:
                    pass
            if "pl_decay" in hp:
                try:
                    self.pl_decay = str(hp["pl_decay"]).strip().lower()
                except Exception:
                    pass
            # Enrich state items to include question/text
            for it in state.items.values():
                if not it.attributes.get("text"):
                    txt = self.text_lookup.get(it.doc_id)
                    if txt:
                        it.attributes["text"] = txt
                if not it.attributes.get("question"):
                    it.attributes["question"] = self.question_text
            return state

        rng = random.Random(self.seed)
        items: Dict[str, ItemState] = {}
        for rec in self.records:
            doc_id = rec.get("doc_id")
            if not doc_id:
                continue
            if doc_id in items:
                continue
            score = float(rec.get("grade_1_5", 0))
            items[doc_id] = ItemState(item_id=doc_id, doc_id=doc_id, attributes=rec, score=score)
        state = RankingState(
            iteration=0,
            stable_streak=0,
            random_seed=self.seed,
            items=items,
            hyperparams={
                "stability_turns": self.stability_turns,
                "confirmations_required": self.confirmations_required,
                "margin_z": self.margin_z,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "model_name": self.model_name,
            },
        )
        self._update_top20(state)
        return state

    def _adj(self) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = defaultdict(list)
        for u, v in self.state.locked_edges:
            g[u].append(v)
        return g

    def _rng(self) -> random.Random:
        return random.Random(self.state.random_seed + self.state.iteration)

    def _update_top20(self, state: Optional[RankingState] = None) -> List[str]:
        if state is None:
            state = self.state
        scores = {iid: it.score for iid, it in state.items.items()}
        order = topological_order(list(state.items.keys()), scores, state.locked_edges)
        state.top20_ids = order[:20]
        return order

    def _sample_batch(self) -> List[ItemState]:
        rng = self._rng()
        items = list(self.state.items.values())
        items.sort(key=lambda it: (it.exposures + 0.3 * math.log1p(it.info) + rng.random()))
        batch = items[:LISTWISE_BATCH]
        rng.shuffle(batch)
        return batch

    def _build_prompt(self, batch: List[ItemState]) -> str:
        labels = ["A", "B", "C", "D", "E"]
        lines = [f"Question:\n{self.question_text}", "Passages:"]
        for lab, it in zip(labels, batch):
            text = it.attributes.get("text") or "[Texte absent]"
            lines.append(f"{lab}. doc_id={it.doc_id}\n{text}\n")
        lines.append('Retourne uniquement: {"order": ["A","B","C","D","E"]}')
        return "\n".join(lines)

    def _call_judge(self, batch: List[ItemState]) -> List[str]:
        labels = ["A", "B", "C", "D", "E"]
        prompt = self._build_prompt(batch)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = completion.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
            else:
                raise ValueError(f"Judge did not return JSON: {content!r}")
        order: List[str] = data.get("order")
        if not isinstance(order, list) or sorted(order) != labels:
            raise ValueError(f"Invalid order from judge: {order!r}")
        return order

    def _update_locks(self, batch: List[ItemState], labels: List[str]) -> int:
        lab_to_item = {lab: it for lab, it in zip(["A", "B", "C", "D", "E"], batch)}
        ranking = [lab_to_item[l].item_id for l in labels]
        adj = self._adj()
        added = 0
        for w, l in iter_pairs(ranking):
            if dag_has_path(adj, w, l):
                continue
            if dag_has_path(adj, l, w):
                continue
            key = f"{w}||{l}"
            cnt = self.state.pending_counts.get(key, 0) + 1
            self.state.pending_counts[key] = cnt
            winner_item = self.state.items[w]
            loser_item = self.state.items[l]
            margin_ok = margin_clear(winner_item, loser_item, self.margin_z)
            strong_trans = has_two_disjoint_paths(adj, w, l)
            should_lock = False
            if cnt >= self.confirmations_required:
                should_lock = True
            elif cnt >= 1 and margin_ok:
                should_lock = True
            elif strong_trans and margin_ok:
                should_lock = True
            if should_lock:
                self.state.locked_edges.append((w, l))
                self.state.pending_counts.pop(key, None)
                adj = self._adj()
                added += 1
        # drop implied pendings
        for key in list(self.state.pending_counts.keys()):
            w, l = key.split("||", 1)
            if dag_has_path(adj, w, l):
                self.state.pending_counts.pop(key, None)
        return added

    def _pl_update(self, batch: List[ItemState], labels: List[str]) -> None:
        # Implement the legacy PL update: stage-wise softmax over suffixes
        lab_to_item = {lab: it for lab, it in zip(["A", "B", "C", "D", "E"], batch)}
        items_ordered = [lab_to_item[l] for l in labels]
        current_iter = self.state.iteration
        if self.pl_decay == "inv_sqrt":
            eta = self.pl_step_size / math.sqrt(1.0 + current_iter / 50.0)
            eta = max(self.pl_step_min, eta)
        else:
            eta = self.pl_step_size

        deltas: Dict[str, float] = {item.item_id: 0.0 for item in items_ordered}

        for position in range(len(items_ordered)):
            suffix = items_ordered[position:]
            if not suffix:
                continue
            max_score = max(item.score for item in suffix)
            exp_values = [math.exp(item.score - max_score) for item in suffix]
            denom = sum(exp_values)
            probs = [1.0 / len(suffix)] * len(suffix) if denom == 0.0 else [v / denom for v in exp_values]
            winner_item = suffix[0]
            winner_prob = probs[0]
            deltas[winner_item.item_id] += eta * (1.0 - winner_prob)
            for prob, item in zip(probs, suffix):
                info_increment = prob * (1.0 - prob)
                item.info += info_increment
                if item is winner_item:
                    continue
                deltas[item.item_id] -= eta * prob

        exposures_updated: set[str] = set()
        for item in items_ordered:
            delta_value = deltas[item.item_id]
            if abs(delta_value) > self.pl_clip:
                delta_value = math.copysign(self.pl_clip, delta_value)
            item.score += delta_value
            if item.item_id not in exposures_updated:
                item.exposures += 1
                exposures_updated.add(item.item_id)
        if self.pl_recenter_every > 0 and (current_iter + 1) % self.pl_recenter_every == 0:
            all_scores = [it.score for it in self.state.items.values()]
            if all_scores:
                mean_score = sum(all_scores) / len(all_scores)
                for it in self.state.items.values():
                    it.score -= mean_score

    def _write_checkpoint(self) -> None:
        # Update hyperparams snapshot
        self.state.hyperparams.update(
            {
                "stability_turns": self.stability_turns,
                "confirmations_required": self.confirmations_required,
                "margin_z": self.margin_z,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "pl_step_size": self.pl_step_size,
                "pl_step_min": self.pl_step_min,
                "pl_clip": self.pl_clip,
                "pl_recenter_every": self.pl_recenter_every,
                "pl_decay": self.pl_decay,
            }
        )
        payload = {
            "iteration": self.state.iteration,
            "stable_streak": self.state.stable_streak,
            "random_seed": self.state.random_seed,
            "items": {
                iid: {
                    "doc_id": it.doc_id,
                    "attributes": it.attributes,
                    "score": it.score,
                    "exposures": it.exposures,
                    "info": it.info,
                }
                for iid, it in self.state.items.items()
            },
            "locked_edges": list(self.state.locked_edges),
            "pending_counts": dict(self.state.pending_counts),
            "top20_ids": list(self.state.top20_ids),
            "hyperparams": dict(self.state.hyperparams),
            "finished": self.state.finished,
        }
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.checkpoint_path)
        # Snapshot: human readable preview
        order = self._update_top20()
        top20 = order[:20]
        preview: List[Dict[str, Any]] = []
        for rank, iid in enumerate(top20, start=1):
            it = self.state.items[iid]
            rec = dict(it.attributes)
            rec["final_rank_preview"] = rank
            rec["current_score"] = it.score
            preview.append(rec)
        with self.snapshot_path.open("w", encoding="utf-8") as h:
            for rec in preview:
                json.dump(rec, h, ensure_ascii=False)
                h.write("\n")

    def _finalize(self, output_path: Path) -> Dict[str, Any]:
        order = self._update_top20()
        top20 = order[:20]
        recs: List[Dict[str, Any]] = []
        for rank, iid in enumerate(top20, start=1):
            it = self.state.items[iid]
            record = dict(it.attributes)
            record["final_rank"] = rank
            record["final_score"] = it.score
            record["exposures"] = it.exposures
            record["info"] = it.info
            record["qid"] = self.qid
            recs.append(record)
        write_jsonl(output_path, recs)
        self.state.finished = True
        self._write_checkpoint()
        return {
            "qid": self.qid,
            "output": str(output_path),
            "iterations": self.state.iteration,
            "stable_streak": self.state.stable_streak,
            "locked_edges": self.state.locked_edges,
            "pending_counts": self.state.pending_counts,
            "top20_ids": top20,
            "hyperparams": dict(self.state.hyperparams),
        }

    def run(self, output_path: Path) -> Dict[str, Any]:
        if self.state.finished and output_path.exists():
            return {
                "qid": self.qid,
                "output": str(output_path),
                "iterations": self.state.iteration,
                "stable_streak": self.state.stable_streak,
                "locked_edges": self.state.locked_edges,
                "pending_counts": self.state.pending_counts,
                "top20_ids": self.state.top20_ids[:20],
            }
        if self.state.stable_streak >= self.stability_turns:
            return self._finalize(output_path)
        while True:
            before = self._update_top20()
            top20_before = before[:20]
            if self.iteration_limit and self.state.iteration >= self.iteration_limit:
                return self._finalize(output_path)
            batch = self._sample_batch()
            labels = self._call_judge(batch)
            locks_added = self._update_locks(batch, labels)
            self._pl_update(batch, labels)
            after = self._update_top20()
            top20_after = after[:20]
            self.state.stable_streak = self.state.stable_streak + 1 if top20_after == top20_before else 0
            self.state.iteration += 1
            self.state.random_seed += 1
            self._write_checkpoint()
            if self.state.stable_streak >= self.stability_turns:
                return self._finalize(output_path)


def run_rank_stage(*, qids: Sequence[str], run_id: Optional[str], override_stability: Optional[int], cfg: Config) -> None:
    run_id = run_id or default_run_id()
    paths = RunPaths(run_id)
    setup_logger(paths.logs_dir / f"s6_rank_{run_id}.log")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for ranking")
    client = OpenAI(api_key=api_key)

    # override stability
    if override_stability is not None:
        cfg.s6.stability_turns = int(override_stability)

    for qid in (list(qids) if qids else [f"Q{i}" for i in range(1, 6)]):
        # question text
        qtext = read_json(paths.s1_rewrite_path(qid)).get("text", "")
        # pruned records
        # Support legacy fallback filename if present
        pruned_path = paths.s5_pruned_path(qid)
        fallback = paths.q_dir(qid) / "s5_pruned" / f"{q_segment(qid)}.top40plus.jsonl"
        if not pruned_path.exists() and fallback.exists():
            pruned_path = fallback
        records = read_jsonl(pruned_path)

        # Guard: if fewer than LISTWISE_BATCH items, finalize directly without ranking
        if len(records) < LISTWISE_BATCH:
            # Order by grade desc, then rank_rrf asc, then doc_id for determinism
            def _rrf(rec: Dict[str, Any]) -> int:
                v = rec.get("rank_rrf")
                try:
                    return int(v)
                except Exception:
                    return 10**9

            sorted_recs = sorted(
                records,
                key=lambda r: (
                    -(r.get("grade_1_5") or 0),
                    _rrf(r),
                    str(r.get("doc_id")),
                ),
            )
            out: List[Dict[str, Any]] = []
            for rank, rec in enumerate(sorted_recs[:20], start=1):
                row = dict(rec)
                row["final_rank"] = rank
                row["final_score"] = float(rec.get("grade_1_5") or 0)
                row["exposures"] = 0
                row["info"] = EPSILON_INFO
                row["qid"] = qid
                out.append(row)
            write_jsonl(paths.s6_ranked_path(qid), out)
            write_stage_manifest(
                paths.q_dir(qid) / "s6_ranked",
                {
                    "run_id": run_id,
                    "qid": qid,
                    "stage": "s6_rank",
                    "iterations": 0,
                    "stable_streak": 0,
                    "locked_edges": [],
                    "pending_counts": {},
                    "top20_ids": [r.get("doc_id") for r in out],
                    "hyperparams": {
                        "note": "finalized without ranking; fewer than batch size",
                        "batch_size": LISTWISE_BATCH,
                    },
                },
            )
            continue
        # Respect S6_TEMP_DIR override if provided
        temp_override = os.getenv("S6_TEMP_DIR")
        if temp_override:
            temp_dir = Path(temp_override).expanduser()
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = (paths.root.parent / "cache" / "ranking")
        temp_dir.mkdir(parents=True, exist_ok=True)
        ranker = Top20Ranker(qid, qtext, records, cfg, client, temp_dir, paths)
        result = ranker.run(paths.s6_ranked_path(qid))
        write_stage_manifest(
            paths.q_dir(qid) / "s6_ranked",
            {"run_id": run_id, "qid": qid, "stage": "s6_rank", **result},
        )
