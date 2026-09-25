
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv
load_dotenv()

import os
import time
import hashlib
import requests
import pandas as pd







from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder



loader = DirectoryLoader(
    "/Users/karim/Downloads/DATA_TEST",           # chemin vers ton dossier
    glob="**/*.md",          # tous les fichiers .md, y compris dans les sous-dossiers
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()




def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


def get_price_petrol():
    url = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/prix-des-carburants-en-france-flux-instantane-v2/records"

    params = {
        "limit": 20
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    results = []

    for station in data["results"]:
        results.append({
            "ville": station.get("ville"),
            "prix_gazole": station.get("gazole_prix")
        })

    return results


# --- 2. Chunking ---

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      # taille approximative d'un chunk (en caractères)
    chunk_overlap=60,    # chevauchement pour ne pas couper le contexte
)

chunks = splitter.split_documents(documents)
# # print(f"{len(chunks)} chunks créés")

# # ------------------verification du fichier changé ou pas-----------

def get_folder_hash(
    folder_path,
    extension=".md",
    chunk_size=600,
    chunk_overlap=60,
    embedding_model="text-embedding-3-large"
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
current_hash = current_hash = get_folder_hash(
    "/Users/karim/Downloads/DATA_TEST",
    chunk_size=600,
    chunk_overlap=60,
    embedding_model="text-embedding-3-large"
)

needs_reindex = True
if os.path.exists(hash_file):
    with open(hash_file, "r") as f:
        if f.read().strip() == current_hash:
            needs_reindex = False

# # --- 3. Embeddings + indexation (par lots) ---

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if needs_reindex:
    print("Dossier modifié — réindexation...")
    os.makedirs(chroma_dir, exist_ok=True)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=chroma_dir)

    batch_size = 1000
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


# print("Base vectorielle créée et persistée")


# # --- 4. Retrievers : vectoriel + BM25, puis combinés (hybrid search) ---

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

    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # On garde uniquement les documents suffisamment pertinents
    filtered_docs = [
        doc
        for doc, score in ranked_docs
        if score >= threshold
    ]

    return filtered_docs[:top_k]


# # --- 5. Prompt + LLM ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_template("""
Tu réponds uniquement à partir du contexte fourni ci-dessous.
Si l'information n'est pas présente dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :
""")

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

def retrieve_and_rerank(question):

    # 1. Hybrid retrieval
    docs = ensemble_retriever.invoke(question)

    # 2. Protection contre les documents contenant des injections
    docs = sanitize_docs(docs)

    # 3. Reranking
    docs = rerank_documents(
        question,
        docs,
        top_k=5,
        threshold = 0.2
    )

    # 4. Construction du contexte
    return format_docs(docs)

rag_chain = (
    {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | validate_output
)

# Utilisation

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


question = validate_question(
    "cites moi tous les véhicules électriques dont le prix dépasse 250 000 DT"
)

reponse = rag_chain.invoke(question)
print(reponse)





# chroma_dir = "./chroma_db"
# hash_file = os.path.join(chroma_dir, "source_hash.txt")

# remote_path = "hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet"
# local_path = "passages.parquet"

# if not os.path.exists(local_path):
#     df = pd.read_parquet(remote_path)
#     df.to_parquet(local_path)
# else:
#     df = pd.read_parquet(local_path)

# # Utilise ensuite local_path pour le hash
# parquet_path = local_path
# # -------------- 1 chargement------------
# df = pd.read_parquet(remote_path)
# documents = []
# for _, row in df.iterrows():
#     contenu = row["passage"]  # adapte au nom de ta colonne
#     documents.append(Document(
#         page_content=contenu,
#         metadata={"source": str(row.get("id", ""))}
#     ))

# loader = PyPDFLoader("PDF.pdf")
# documents = loader.load()
# print(f"{len(documents)} page(s) chargée(s)")




# resultats = ensemble_retriever.invoke("cites moi tous les véhicules électriques dont le prix dépasse 250 000 DT")
# for r in resultats:
#     print(r.page_content, "\n---")

# # # Test rapide
# # resultats = retriever.invoke("Quelle est la question à tester ?")
# # for r in resultats:
# #     print(r.page_content[:200], "\n---")