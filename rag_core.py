"""
rag_core.py — Côté requête du pipeline RAG (retrieval, reranking, agent).

Ce module NE construit PAS l'index : il lit celui produit par ingest.py
(dossier index/). Si l'index est absent ou incompatible avec la config,
build_agent() échoue immédiatement avec un message explicite plutôt que
de réindexer en silence au démarrage de l'application.

Il ne dépend d'aucune interface : api.py l'importe, tout autre client
pourrait le faire de la même façon.
"""

import json
import logging
import os

import requests
from geopy.distance import geodesic
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder
from langgraph.prebuilt import create_react_agent

from config import Settings, get_settings

logger = logging.getLogger(__name__)


# ============================================================
# Fonctions internes — lecture de l'index produit par ingest.py
# ============================================================

class IndexNotReadyError(RuntimeError):
    """L'index est absent ou a été construit avec une autre configuration."""


def load_manifest(settings: Settings) -> dict:
    """Lit le manifest de l'index et vérifie qu'il est compatible avec la config."""
    if not settings.manifest_file.exists():
        raise IndexNotReadyError(
            "Index introuvable. Lance d'abord l'ingestion : python ingest.py"
        )
    with open(settings.manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Interroger un index avec un autre modèle d'embedding que celui qui l'a
    # construit ne plante pas : ça renvoie juste de mauvais résultats. On bloque.
    for key in ("embedding_model", "collection_name"):
        if manifest.get(key) != getattr(settings, key):
            raise IndexNotReadyError(
                f"Index construit avec {key}={manifest.get(key)}, "
                f"config actuelle : {getattr(settings, key)}. Relance : python ingest.py --force"
            )
    return manifest


def _load_chunks(settings: Settings):
    chunks = []
    with open(settings.chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(Document(page_content=record["page_content"], metadata=record["metadata"]))
    return chunks


def _load_vectorstore(settings: Settings, embeddings):
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def _sanitize_docs(docs):
    suspicious = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "reveal system prompt",
        "system message",
        "you are now"
    ]
    safe_docs = []
    for doc in docs:
        content = doc.page_content.lower()
        if any(pattern in content for pattern in suspicious):
            continue
        safe_docs.append(doc)
    return safe_docs


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _validate_output(answer):
    if not answer or not answer.strip():
        return "Je n'ai pas trouvé suffisamment d'informations."
    if len(answer) > 5000:
        return "Réponse trop longue."
    return answer


def validate_question(question: str) -> str:
    question = question.strip()

    if not question:
        raise ValueError("Question vide")
    if len(question) > 1000:
        raise ValueError("Question trop longue")

    suspicious_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "reveal your prompt",
        "show me your instructions",
        "forget your instructions"
    ]
    question_lower = question.lower()
    for pattern in suspicious_patterns:
        if pattern in question_lower:
            raise ValueError("Prompt injection détectée")

    return question


# ============================================================
# Point d'entrée principal — à appeler UNE SEULE FOIS par process
# (api.py le fait au démarrage, dans son lifespan)
# ============================================================

def build_agent(settings: Settings | None = None):
    """
    Construit le pipeline de requête (retrievers, reranker, tools, agent) à
    partir de l'index existant et retourne l'agent prêt à l'emploi.

    Ne fait aucun appel d'embedding sur le corpus : l'indexation est le rôle
    de ingest.py. Reste coûteux (charge le modèle de reranking) — à appeler
    une seule fois par process.
    """
    settings = settings or get_settings()

    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        logger.info("LangSmith activé — projet : %s", os.getenv("LANGCHAIN_PROJECT", "default"))
    else:
        logger.info("LangSmith désactivé (LANGCHAIN_TRACING_V2 != 'true')")

    load_manifest(settings)
    chunks = _load_chunks(settings)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    vectorstore = _load_vectorstore(settings, embeddings)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.retriever_k
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[settings.bm25_weight, 1 - settings.bm25_weight]
    )

    reranker = CrossEncoder(settings.reranker_model)

    def rerank_documents(question, docs, top_k, threshold):
        if not docs:
            return []
        pairs = [[question, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        filtered_docs = [doc for doc, score in ranked_docs if score >= threshold]
        return filtered_docs[:top_k]

    def retrieve_and_rerank(question):
        docs = ensemble_retriever.invoke(question)
        docs = _sanitize_docs(docs)
        docs = rerank_documents(
            question, docs, top_k=settings.rerank_top_k, threshold=settings.rerank_threshold
        )
        return _format_docs(docs)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_template("""
Tu réponds uniquement à partir du contexte fourni ci-dessous.
Si l'information n'est pas présente dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :
""")

    rag_chain = (
        {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | _validate_output
    )

    @tool
    def search_vehicules(question: str) -> str:
        """Recherche des informations sur les véhicules d'occasion (prix, modèle, année,
        kilométrage, motorisation) dans la base de données scrapée d'un site de vente automobile.
        Utilise ce tool pour toute question sur des véhicules spécifiques ou leurs caractéristiques."""
        question = validate_question(question)
        return rag_chain.invoke(question)

    @tool
    def get_weather(latitude: float, longitude: float) -> dict:
        """Retourne la météo actuelle (température, humidité, vitesse du vent) pour des
        coordonnées GPS données (latitude, longitude)."""
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"
        }
        response = requests.get(url, params=params, timeout=settings.http_timeout_s)
        response.raise_for_status()
        return response.json()

    @tool
    def get_price_petrol(user_lat: float = None, user_lon: float = None, rayon_km: float = 25) -> list:
        """Retourne les stations-service avec leurs prix (gazole, SP95, SP98), horaires
        d'ouverture, et distance par rapport à l'utilisateur. Si user_lat/user_lon sont
        fournis, filtre uniquement les stations dans le rayon donné (25km par défaut),
        triées de la plus proche à la plus éloignée."""
        
        url = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/prix-des-carburants-en-france-flux-instantane-v2/records"
        params = {"limit": 100}  # augmente l'échantillon pour avoir des résultats dans le rayon
        response = requests.get(url, params=params, timeout=settings.http_timeout_s)
        response.raise_for_status()
        data = response.json()

        resultats = []
        for s in data["results"]:
            geom = s.get("geom") or {}
            station_lat = geom.get("lat")
            station_lon = geom.get("lon")

            distance_km = None
            if user_lat is not None and user_lon is not None and station_lat and station_lon:
                distance_km = geodesic((user_lat, user_lon), (station_lat, station_lon)).km
                if distance_km > rayon_km:
                    continue  # hors du rayon demandé, on ignore

            resultats.append({
                "ville": s.get("ville"),
                "adresse": s.get("adresse"),
                "prix_gazole": s.get("gazole_prix"),
                "prix_sp95": s.get("sp95_prix"),
                "prix_sp98": s.get("sp98_prix"),
                "carburants_disponibles": s.get("carburants_disponibles"),
                "horaires_24_24": s.get("horaires_automate_24_24"),
                "horaires_detail": s.get("horaires"),
                "distance_km": round(distance_km, 1) if distance_km is not None else None,
            })

        if user_lat is not None and user_lon is not None:
            resultats.sort(key=lambda x: x["distance_km"] if x["distance_km"] is not None else float("inf"))

        return resultats

    tools = [search_vehicules, get_weather, get_price_petrol]

    system_prompt = """Tu es un assistant qui répond à des questions sur des véhicules
d'occasion, la météo, et le prix du carburant.

Règle stricte n°1 — résolution des références :
Avant d'appeler search_vehicules, si la question contient une référence implicite
à un véhicule déjà mentionné plus haut dans la conversation ("la première voiture",
"son kilométrage", "celui-ci", "ce modèle"...), tu dois d'abord identifier de quel
véhicule précis il s'agit à partir de l'historique, puis reformuler une requête de
recherche EXPLICITE et AUTONOME contenant le nom réel du véhicule.

Exemple :
- Historique : tu as cité "Porsche Cayenne Electric" et "Mercedes-Benz EQE SUV"
- Question : "donne-moi la fiche technique de la première voiture"
- Mauvais appel : search_vehicules("fiche technique de la première voiture")
- Bon appel : search_vehicules("fiche technique Porsche Cayenne Electric")

Règle stricte n°2 — jamais de réponse sans recherche :
Pour TOUTE question concernant un véhicule (prix, marque, modèle, année,
kilométrage, fiche technique...), tu dois TOUJOURS appeler search_vehicules avec
une requête explicite (jamais de référence implicite dans l'appel lui-même).
Ne réponds JAMAIS à une question sur un véhicule à partir de ta seule mémoire de
la conversation, même si l'information semble déjà connue.

Règle stricte n°3 — pas d'invention :
Si l'outil ne retourne aucune information pertinente, dis clairement que tu ne l'as
pas trouvée. N'invente jamais une marque, un prix, ou une caractéristique.

Règle stricte n°4 — réutiliser l'historique quand c'est pertinent :
Si la question porte sur un sujet déjà discuté plus haut dans la conversation
(un véhicule déjà cité, une réponse déjà donnée), tu peux répondre directement à
partir de l'historique SANS rappeler search_vehicules, sauf si un détail précis
manque et nécessite une nouvelle recherche.

Règle stricte n°5 — demandes de résumé global :
Si l'utilisateur demande un résumé, un récapitulatif, ou un "bref" de "la
discussion"/"tout ça"/"la conversation" sans préciser un sujet particulier,
tu dois résumer L'ENSEMBLE des échanges précédents de la conversation (tous
les sujets abordés : véhicules, météo, carburant...), pas seulement la
dernière réponse donnée. 

Règle stricte n°6 :
Note technique : les questions peuvent contenir en fin de message une ligne du type
"(Position actuelle de l'utilisateur : latitude=..., longitude=...)". C'est une
information technique fournie automatiquement pour te permettre de localiser
l'utilisateur (ex: pour get_price_petrol) — ne la répète JAMAIS dans tes réponses,
ne mentionne pas les coordonnées GPS brutes à l'utilisateur. Utilise plutôt un nom
de ville/lieu si tu peux le déduire, ou dis simplement "votre position actuelle"
sans donner les chiffres."""

    agent = create_react_agent(llm, tools, prompt=system_prompt)

    return agent


def ask_agent(
    agent,
    question: str,
    history: list | None = None,
    session_id: str = "default",
    location: tuple[float, float] | None = None,
    channel: str = "api",
) -> str:
    """
    Pose une question à l'agent déjà construit et retourne uniquement le
    texte de la réponse finale.

    `history` : liste de messages précédents au format
    [{"role": "user"/"assistant", "content": "..."}], pour que l'agent
    garde le contexte de la conversation (sinon chaque appel est traité
    isolément, sans mémoire des échanges précédents).

    `location` : (latitude, longitude) de l'utilisateur si connue, transmise
    au LLM pour get_price_petrol (voir règle n°6 du system prompt).

    `channel` : origine de l'appel ("api", "streamlit"...), pour filtrer les
    traces LangSmith.
    """
    contenu = question
    if location is not None:
        lat, lon = location
        contenu += f"\n\n(Position actuelle de l'utilisateur : latitude={lat}, longitude={lon})"

    messages = (history or []) + [{"role": "user", "content": contenu}]

    result = agent.invoke(
        {"messages": messages},
        config={
            "run_name": "rag-agent-query",
            "tags": ["rag-vehicules", "agent", channel],
            "metadata": {"question": question, "session_id": session_id},
        }
    )
    return result["messages"][-1].content