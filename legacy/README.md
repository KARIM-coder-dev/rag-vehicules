# Ancien code — non maintenu

Premières versions du projet, conservées pour référence. Elles ne sont ni testées
ni déployées, et certains chemins pointent vers d'anciens emplacements.

| Fichier | Rôle | Remplacé par |
|---|---|---|
| `main.py`, `main_tools.py` | Premières versions du pipeline RAG et de l'agent, en un seul script | `src/rag_vehicules/` |
| `eval-rag.py` | Génération d'un jeu de test et exécution du RAG (branché sur `main_tools.py`) | à refaire (étape évaluation) |
| `evale-rag.py` | Calcul des métriques RAGAS sur `eval_data.json` | à refaire (étape évaluation) |
| `eval_data.json` | Résultats de l'ancienne évaluation | — |
