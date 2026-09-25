# RAG Véhicules

Agent conversationnel qui répond à des questions sur des véhicules et concessionnaires
(corpus scrapé d'automobile.tn), la météo et le prix des carburants. Recherche hybride
(BM25 + vecteurs), reranking, agent LangGraph, exposé en API FastAPI.

## Arborescence

```
rag-vehicules/
├── src/rag_vehicules/      Code de production (package Python)
│   ├── config.py           Configuration lue depuis l'environnement, validée au démarrage
│   ├── ingest.py           Ingestion : documents → chunks → embeddings → index
│   ├── core.py             Requête : retrieval hybride, reranking, agent et outils
│   └── api.py              API HTTP (POST /ask, GET /health)
├── frontend/app.py         Interface Streamlit, simple client de l'API
├── tests/                  Tests automatisés (sans appel à OpenAI)
├── data/
│   ├── docs/               Documents sources (.md)
│   └── index/              Index généré par l'ingestion (non versionné)
├── docker/                 Dockerfile.api (API + ingestion), Dockerfile.front
├── requirements/           Dépendances : *.in édités à la main, *.txt verrouillés (pip-tools)
├── legacy/                 Ancien code, conservé pour référence, non maintenu
├── compose.yaml            Toute la chaîne en local : ingest → api → front
├── pyproject.toml          Métadonnées du package et configuration pytest
└── .env.example            Liste des variables de configuration
```

## Démarrage en local

```bash
python -m venv env && source env/bin/activate
pip install -r requirements/dev.txt && pip install -e . --no-deps
cp .env.example .env                      # puis renseigner OPENAI_API_KEY

python -m rag_vehicules.ingest            # construit data/index/ (une seule fois)
uvicorn rag_vehicules.api:app --reload    # API  : http://localhost:8000/docs
streamlit run frontend/app.py             # UI   : http://localhost:8501
```

## Avec Docker

```bash
docker compose up --build                 # ingest, puis api, puis front
```

## Tests

```bash
pytest
```

## Modifier une dépendance

Éditer `requirements/api.in` (ou `front.in`), puis depuis `requirements/` :

```bash
pip-compile api.in --strip-extras --no-emit-index-url
pip-compile front.in --strip-extras
```
