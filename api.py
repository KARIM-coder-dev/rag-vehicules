"""
api.py — API HTTP de l'agent RAG véhicules.

Lancement (dev) :
    uvicorn api:app --reload
Documentation interactive générée automatiquement : http://localhost:8000/docs

L'index doit avoir été construit au préalable (python ingest.py) : sinon
l'API refuse de démarrer plutôt que de servir des réponses vides.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from config import get_settings
from rag_core import ask_agent, build_agent, load_manifest, validate_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag_api")

MAX_HISTORY = 20


# ============================================================
# Schémas d'entrée / sortie — validés automatiquement par FastAPI
# ============================================================

class Message(BaseModel):
    # Seuls "user" et "assistant" sont acceptés : un client ne doit pas
    # pouvoir injecter un faux message "system" dans l'historique.
    role: Literal["user", "assistant"]
    content: str = Field(max_length=10_000)


class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY)
    session_id: str | None = Field(default=None, max_length=100)
    location: Location | None = None


class AskResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: int


# ============================================================
# Cycle de vie — l'agent est construit UNE fois au démarrage du process
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configuration validée avant tout : une variable manquante arrête le
    # démarrage ici, pas à la première requête d'un utilisateur.
    settings = get_settings()
    logger.info("Chargement de l'agent (LLM=%s, k=%d)...", settings.llm_model, settings.retriever_k)
    app.state.agent = build_agent(settings)
    app.state.manifest = load_manifest(settings)
    logger.info("Agent prêt (index du %s, %s chunks)",
                app.state.manifest["indexed_at"], app.state.manifest["nb_chunks"])
    yield


app = FastAPI(title="RAG Véhicules", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attribue un identifiant à chaque requête, renvoyé dans l'en-tête
    X-Request-ID : c'est ce que l'utilisateur transmet au support, et ce
    qu'on cherche ensuite dans les logs."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health(request: Request):
    """Sonde de disponibilité : répond seulement si l'agent est chargé, et
    indique quelle version de l'index est servie."""
    manifest = request.app.state.manifest
    return {
        "status": "ok",
        "index": {
            "indexed_at": manifest["indexed_at"],
            "nb_chunks": manifest["nb_chunks"],
            "embedding_model": manifest["embedding_model"],
        },
    }


# Fonction synchrone (def, pas async def) : agent.invoke est bloquant,
# FastAPI l'exécute donc dans un pool de threads sans bloquer les autres requêtes.
@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    request_id = request.state.request_id
    session_id = body.session_id or str(uuid.uuid4())

    try:
        question = validate_question(body.question)
    except ValueError as e:
        logger.warning("request_id=%s question refusée : %s", request_id, e)
        raise HTTPException(status_code=400, detail=str(e))

    location = (body.location.latitude, body.location.longitude) if body.location else None

    start = time.perf_counter()
    try:
        answer = ask_agent(
            request.app.state.agent,
            question,
            history=[m.model_dump() for m in body.history],
            session_id=session_id,
            location=location,
            channel=request.headers.get("X-Client", "api"),
        )
    except Exception:
        # Le détail technique va dans les logs, jamais dans la réponse au client.
        logger.exception("request_id=%s échec de l'agent", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne. Référence : {request_id}",
        )
    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info("request_id=%s session_id=%s latency_ms=%d", request_id, session_id, latency_ms)
    return AskResponse(answer=answer, session_id=session_id, latency_ms=latency_ms)
