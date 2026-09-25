"""
config.py — Paramètres partagés entre l'ingestion (ingest.py) et la requête (rag_core.py).

Les deux côtés DOIVENT lire les mêmes valeurs : si la requête utilise un
autre modèle d'embedding ou une autre collection que l'ingestion, la
recherche vectorielle renvoie des résultats faux sans lever d'erreur.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Source des documents
DOSSIER_DOCS = os.path.join(BASE_DIR, "DATA_TEST")

# Index produit par ingest.py et lu par rag_core.py
INDEX_DIR = os.path.join(BASE_DIR, "index")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.jsonl")
MANIFEST_FILE = os.path.join(INDEX_DIR, "manifest.json")
COLLECTION_NAME = "vehicules"

# Paramètres d'indexation
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60
EMBEDDING_MODEL = "text-embedding-3-large"
