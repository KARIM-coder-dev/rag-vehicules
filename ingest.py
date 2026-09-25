"""
ingest.py — Pipeline d'ingestion : documents → chunks → embeddings → index.

S'exécute à part de l'application, uniquement quand les documents changent :
    python ingest.py           # réindexe seulement si les sources ont changé
    python ingest.py --force   # réindexe dans tous les cas

Produit le dossier index/ lu par rag_core.py :
    index/chroma/        base vectorielle
    index/chunks.jsonl   chunks bruts (pour reconstruire BM25 sans relire les sources)
    index/manifest.json  hash des sources + paramètres utilisés pour indexer

L'index est construit dans un dossier temporaire puis échangé d'un coup :
une ingestion qui échoue en cours de route ne laisse jamais un index à moitié écrit.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOSSIER_DOCS,
    EMBEDDING_MODEL,
    INDEX_DIR,
)

BATCH_SIZE = 1000


def compute_source_hash():
    """Hash des fichiers sources ET des paramètres d'indexation :
    changer le chunking ou le modèle d'embedding force aussi une réindexation."""
    hash_md5 = hashlib.md5()
    hash_md5.update(f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{EMBEDDING_MODEL}".encode("utf-8"))

    for root, _, files in sorted(os.walk(DOSSIER_DOCS)):
        for filename in sorted(files):
            if filename.endswith(".md"):
                with open(os.path.join(root, filename), "rb") as f:
                    hash_md5.update(f.read())

    return hash_md5.hexdigest()


def read_manifest(index_dir=INDEX_DIR):
    path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_chunk_documents():
    loader = DirectoryLoader(
        DOSSIER_DOCS,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return documents, splitter.split_documents(documents)


def write_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_vectorstore(chunks, chroma_dir):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        vectorstore.add_documents(chunks[i:i + BATCH_SIZE])
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks indexés")
        time.sleep(1)


def build_index(source_hash):
    documents, chunks = load_and_chunk_documents()
    print(f"{len(documents)} documents → {len(chunks)} chunks")

    # Construction dans un dossier neuf : pas de doublons avec un ancien index
    tmp_dir = INDEX_DIR + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    write_chunks(chunks, os.path.join(tmp_dir, "chunks.jsonl"))
    write_vectorstore(chunks, os.path.join(tmp_dir, "chroma"))

    manifest = {
        "source_hash": source_hash,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection_name": COLLECTION_NAME,
        "nb_documents": len(documents),
        "nb_chunks": len(chunks),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Échange : l'ancien index n'est supprimé qu'une fois le nouveau complet
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.replace(tmp_dir, INDEX_DIR)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Construit l'index RAG à partir des documents.")
    parser.add_argument("--force", action="store_true", help="réindexe même si les sources n'ont pas changé")
    args = parser.parse_args()

    source_hash = compute_source_hash()
    manifest = read_manifest()

    if not args.force and manifest and manifest.get("source_hash") == source_hash:
        print(f"Index à jour ({manifest['nb_chunks']} chunks, indexé le {manifest['indexed_at']}) — rien à faire.")
        return

    print("Construction de l'index...")
    manifest = build_index(source_hash)
    print(f"Index écrit dans {INDEX_DIR} ({manifest['nb_chunks']} chunks).")


if __name__ == "__main__":
    main()
