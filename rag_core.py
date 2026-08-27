"""
rag_core.py — Cœur du pipeline RAG (chargement, indexation, retrieval, agent).

Ce module ne s'exécute pas directement : il expose des fonctions/objets
que app.py (Streamlit) importe et réutilise. Toute la logique lourde
(chargement, embeddings, indexation) est encapsulée dans build_agent(),
pensée pour être appelée UNE SEULE FOIS grâce au cache Streamlit.
"""

from dotenv import load_dotenv
load_dotenv()

import os

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print(f"LangSmith activé — projet : {os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("LangSmith désactivé (LANGCHAIN_TRACING_V2 absent ou différent de 'true' dans le .env)")

import os
import time
import hashlib
import requests

from geopy.distance import geodesic
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder
from langgraph.prebuilt import create_react_agent


# ============================================================
# Configuration
# ============================================================

DOSSIER_DOCS = DOSSIER_DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA_TEST")
CHROMA_DIR = "./chroma_db"
HASH_FILE = os.path.join(CHROMA_DIR, "source_hash.txt")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60
EMBEDDING_MODEL = "text-embedding-3-large"


# ============================================================
# Fonctions internes — chargement, chunking, cache, indexation
# ============================================================

def _get_folder_hash(folder_path, extension=".md", chunk_size=600, chunk_overlap=60, embedding_model="text-embedding-3-large"):
    hash_md5 = hashlib.md5()
    config = f"{chunk_size}_{chunk_overlap}_{embedding_model}"
    hash_md5.update(config.encode("utf-8"))

    for root, _, files in sorted(os.walk(folder_path)):
        for filename in sorted(files):
            if filename.endswith(extension):
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    hash_md5.update(f.read())

    return hash_md5.hexdigest()


def _load_and_chunk_documents():
    loader = DirectoryLoader(
        DOSSIER_DOCS,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def _build_vectorstore(chunks, embeddings):
    current_hash = _get_folder_hash(
        DOSSIER_DOCS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL
    )

    needs_reindex = True
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            if f.read().strip() == current_hash:
                needs_reindex = False

    if needs_reindex:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            time.sleep(1)

        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
    else:
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

    return vectorstore


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
# Point d'entrée principal — à appeler UNE SEULE FOIS
# (Streamlit le mettra en cache avec @st.cache_resource)
# ============================================================

def build_agent():
    """
    Construit tout le pipeline RAG (chargement, indexation, retrievers,
    reranker, tools, agent) et retourne l'agent prêt à l'emploi.

    Coûteux à l'exécution (charge le modèle de reranking, indexe si besoin) —
    ne doit être appelé qu'une seule fois par session d'application.
    """
    chunks = _load_and_chunk_documents()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = _build_vectorstore(chunks, embeddings)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 50
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    def rerank_documents(question, docs, top_k=5, threshold=0.2):
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
        docs = rerank_documents(question, docs, top_k=5, threshold=0.2)
        return _format_docs(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

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
        response = requests.get(url, params=params)
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
        response = requests.get(url, params=params)
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


def ask_agent(agent, question: str, history: list | None = None, session_id: str = "default") -> str:
    """
    Pose une question à l'agent déjà construit et retourne uniquement le
    texte de la réponse finale.

    `history` : liste de messages précédents au format
    [{"role": "user"/"assistant", "content": "..."}], pour que l'agent
    garde le contexte de la conversation (sinon chaque appel est traité
    isolément, sans mémoire des échanges précédents).
    """
    messages = (history or []) + [{"role": "user", "content": question}]

    result = agent.invoke(
        {"messages": messages},
        config={
            "run_name": "rag-agent-query",
            "tags": ["rag-vehicules", "agent", "streamlit"],
            "metadata": {"question": question, "session_id": session_id},
        }
    )
    return result["messages"][-1].content