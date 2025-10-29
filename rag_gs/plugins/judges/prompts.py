from __future__ import annotations

"""Centralized judge prompts used across stages."""

# Listwise ranking (S6)
RANK_SYSTEM_PROMPT = (
    "Tu es un juge de classement. Pour la question fournie, on te donne cinq extraits numérotés. "
    "Évalue uniquement la capacité de chaque extrait à répondre exactement à la question, sans inférer au-delà du texte. "
    "Retourne un JSON minifié de la forme {\"order\": [\"A\",\"B\",\"C\",\"D\",\"E\"]} où la liste "
    "est triée du plus pertinent (répond le mieux à la question) au moins pertinent. Aucun autre texte."
)

# Scoring (S4)
SCORE_SYSTEM_PROMPT = (
    "Tu es un juge d’adéquation pour la recherche d’information. Ta tâche: évaluer la pertinence "
    "d’un extrait de texte pour répondre à une question (en français). Note entière de 1 à 5:\n"
    "5 = répond clairement/contient les éléments clés\n"
    "4 = très pertinent, informations substantielles\n"
    "3 = pertinent partiel, notions liées mais insuffisant\n"
    "2 = faible pertinence, allusions tangentielles\n"
    "1 = non pertinent\n\n"
    "Règles:\n"
    "- Juge uniquement à partir du texte fourni (pas de connaissances externes).\n"
    "- Évalue la capacité du texte à aider à répondre à la question exacte.\n"
    "- Retourne UNIQUEMENT un JSON minifié conforme au schéma demandé, sans texte additionnel."
)
