from dotenv import load_dotenv
load_dotenv()

import os
import time
import hashlib
import requests
import pandas as pd

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print(f"LangSmith activé — projet : {os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("LangSmith désactivé (LANGCHAIN_TRACING_V2 absent ou différent de 'true' dans le .env)")

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_core.documents import Document
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
# 1. Chargement des documents
# ============================================================

loader = DirectoryLoader(
    "/Users/karim/Downloads/DATA_TEST",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()


# ============================================================
# 2. Chunking
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=35,
)
chunks = splitter.split_documents(documents)


# ============================================================
# 3. Vérification si les documents/config ont changé (cache)
# ============================================================

def get_folder_hash(
    folder_path,
    extension=".md",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-3-small"
):
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


chroma_dir = "./chroma_db"
hash_file = os.path.join(chroma_dir, "source_hash.txt")
current_hash = get_folder_hash(
    "/Users/karim/Downloads/DATA_TEST",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-3-small"
)

needs_reindex = True
if os.path.exists(hash_file):
    with open(hash_file, "r") as f:
        if f.read().strip() == current_hash:
            needs_reindex = False


# ============================================================
# 4. Embeddings + indexation (par lots)
# ============================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if needs_reindex:
    print("Dossier modifié — réindexation...")
    os.makedirs(chroma_dir, exist_ok=True)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=chroma_dir)

    batch_size = 350
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Lot {i // batch_size + 1} indexé ({i + len(batch)}/{len(chunks)})")
        time.sleep(1)

    with open(hash_file, "w") as f:
        f.write(current_hash)
    print("Indexation terminée")
else:
    print("Dossier inchangé — cache réutilisé")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=chroma_dir)


# ============================================================
# 5. Retrievers : vectoriel + BM25 (hybrid search)
# ============================================================

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 20
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


# ============================================================
# 6. Reranking
# ============================================================

reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank_documents(question, docs, top_k=5, threshold=0.2):
    if not docs:
        return []

    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    filtered_docs = [doc for doc, score in ranked_docs if score >= threshold]

    return filtered_docs[:top_k]


# ============================================================
# 7. Sécurité — sanitization et validation
# ============================================================

def sanitize_docs(docs):
    safe_docs = []

    suspicious = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "reveal system prompt",
        "system message",
        "you are now"
    ]

    for doc in docs:
        content = doc.page_content.lower()
        if any(pattern in content for pattern in suspicious):
            continue
        safe_docs.append(doc)

    return safe_docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def validate_output(answer):
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


def retrieve_and_rerank(question):
    docs = ensemble_retriever.invoke(question)
    docs = sanitize_docs(docs)
    docs = rerank_documents(question, docs, top_k=5, threshold=0.2)
    return format_docs(docs)


# ============================================================
# 8. Chaîne RAG (utilisée à l'intérieur du tool véhicules)
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

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
    | validate_output
)


# ============================================================
# 9. Définition des tools pour l'agent
# ============================================================

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
def get_price_petrol() -> list:
    """Retourne les prix actuels du carburant (gazole) dans différentes villes de France,
    à partir des données officielles data.economie.gouv.fr."""
    url = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/prix-des-carburants-en-france-flux-instantane-v2/records"
    params = {"limit": 20}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return [
        {"ville": s.get("ville"), "prix_gazole": s.get("gazole_prix")}
        for s in data["results"]
    ]


# ============================================================
# 10. Agent — choisit automatiquement le bon tool
# ============================================================

tools = [search_vehicules, get_weather, get_price_petrol]
agent = create_react_agent(llm, tools)


# ============================================================
# 11. Utilisation
# ============================================================

if __name__ == "__main__":
    question = "Tu peux me dire le temps qu'il fait aujour'dhui a Paris et dis moi le pris de carburant de la ville Paris"

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print(result["messages"][-1].content)