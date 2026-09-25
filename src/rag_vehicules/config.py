"""
config.py — Configuration de l'application, lue depuis les variables d'environnement.

Toute valeur peut être surchargée par une variable d'environnement du même nom
en majuscules (ex. LLM_MODEL=gpt-4o, RETRIEVER_K=30). En local, elles viennent
du fichier .env ; en production, elles sont injectées par la plateforme
(Azure Container Apps, avec les secrets tirés d'Azure Key Vault) : le code ne
change pas d'un environnement à l'autre.

La configuration est validée au démarrage : une clé manquante ou une valeur
incohérente fait échouer le lancement immédiatement, avec un message clair,
plutôt qu'au milieu d'une requête utilisateur.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings

# Racine du projet : src/rag_vehicules/config.py -> remonte de deux niveaux
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Charge .env dans os.environ (sans écraser les variables déjà définies) :
# LangSmith lit ses propres variables LANGCHAIN_* directement dans l'environnement.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    # --- Secrets -------------------------------------------------------------
    # SecretStr : la valeur n'apparaît jamais dans un print, un log ou une trace.
    # min_length : une variable présente mais vide (OPENAI_API_KEY=) est refusée aussi.
    openai_api_key: SecretStr = Field(min_length=1)

    # --- Ingestion : changer une de ces valeurs impose de relancer l'ingestion --
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    collection_name: str = "vehicules"
    chunk_size: int = Field(600, gt=0)
    chunk_overlap: int = Field(60, ge=0)
    embedding_model: str = "text-embedding-3-large"

    # --- Requête : modifiables sans réindexer --------------------------------
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(0.3, ge=0, le=2)
    llm_timeout_s: float = Field(60, gt=0)
    llm_max_retries: int = Field(2, ge=0)
    retriever_k: int = Field(50, gt=0, le=200)
    bm25_weight: float = Field(0.5, ge=0, le=1)
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = Field(5, gt=0)
    rerank_threshold: float = 0.2
    http_timeout_s: float = Field(10, gt=0)

    @model_validator(mode="after")
    def _check_coherence(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP doit être inférieur à CHUNK_SIZE")
        if self.rerank_top_k > self.retriever_k:
            raise ValueError("RERANK_TOP_K ne peut pas dépasser RETRIEVER_K")
        return self

    # Chemins dérivés de index_dir
    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def chunks_file(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def manifest_file(self) -> Path:
        return self.index_dir / "manifest.json"


@lru_cache
def get_settings() -> Settings:
    """Lit et valide la configuration une seule fois par process."""
    return Settings()
