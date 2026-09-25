"""
ingest.py — Pipeline d'ingestion : documents → chunks → embeddings → index.

S'exécute à part de l'application, uniquement quand les documents changent :
    python -m rag_vehicules.ingest           # réindexe seulement si les sources ont changé
    python -m rag_vehicules.ingest --force   # réindexe dans tous les cas

Produit le dossier data/index/ lu par core.py :
    data/index/chroma/        base vectorielle
    data/index/chunks.jsonl   chunks bruts (pour reconstruire BM25 sans relire les sources)
    data/index/manifest.json  hash des sources + paramètres utilisés pour indexer

L'index est construit dans un dossier temporaire puis échangé d'un coup :
une ingestion qui échoue en cours de route ne laisse jamais un index à moitié écrit.
"""

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

from rag_vehicules.config import get_settings

BATCH_SIZE = 1000


def compute_source_hash():
    """Hash des fichiers sources ET des paramètres d'indexation :
    changer le chunking ou le modèle d'embedding force aussi une réindexation."""
    s = get_settings()
    hash_md5 = hashlib.md5()
    hash_md5.update(f"{s.chunk_size}_{s.chunk_overlap}_{s.embedding_model}".encode("utf-8"))

    for root, _, files in sorted(os.walk(s.docs_dir)):
        for filename in sorted(files):
            if filename.endswith(".md"):
                with open(os.path.join(root, filename), "rb") as f:
                    hash_md5.update(f.read())

    return hash_md5.hexdigest()


def read_manifest():
    path = get_settings().manifest_file
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_chunk_documents():
    s = get_settings()
    loader = DirectoryLoader(
        str(s.docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )
    return documents, splitter.split_documents(documents)


def write_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_vectorstore(chunks, chroma_dir):
    s = get_settings()
    embeddings = OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)
    vectorstore = Chroma(
        collection_name=s.collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        vectorstore.add_documents(chunks[i:i + BATCH_SIZE])
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks indexés")
        time.sleep(1)


def build_index(source_hash):
    s = get_settings()
    documents, chunks = load_and_chunk_documents()
    print(f"{len(documents)} documents → {len(chunks)} chunks")

    # Construction dans un dossier neuf : pas de doublons avec un ancien index
    tmp_dir = s.index_dir.with_name(s.index_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    write_chunks(chunks, tmp_dir / "chunks.jsonl")
    write_vectorstore(chunks, tmp_dir / "chroma")

    manifest = {
        "source_hash": source_hash,
        "embedding_model": s.embedding_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "collection_name": s.collection_name,
        "nb_documents": len(documents),
        "nb_chunks": len(chunks),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Échange : l'ancien index n'est supprimé qu'une fois le nouveau complet
    if s.index_dir.exists():
        shutil.rmtree(s.index_dir)
    os.replace(tmp_dir, s.index_dir)
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
    print(f"Index écrit dans {get_settings().index_dir} ({manifest['nb_chunks']} chunks).")


if __name__ == "__main__":
    main()
