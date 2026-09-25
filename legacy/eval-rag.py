from dotenv import load_dotenv
load_dotenv()

import os
import random
import json

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from main_tools import rag_chain, ensemble_retriever


# ============================================================
# 1. Générer un jeu de test (question + réponse de référence)
#    à partir d'un échantillon de tes documents .md
# ============================================================

DOSSIER_DOCS = "/Users/karim/Downloads/DATA_TEST"
NB_EXEMPLES = 10  # nombre de questions à générer pour le jeu de test

loader = DirectoryLoader(
    DOSSIER_DOCS,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

# Échantillon aléatoire de documents pour générer les questions
echantillon = random.sample(documents, min(NB_EXEMPLES, len(documents)))

llm_generator = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

generation_prompt = ChatPromptTemplate.from_template("""
Voici la fiche d'un véhicule extraite d'un site de vente automobile :

{document}

Génère une question précise et factuelle qu'un utilisateur pourrait poser
sur ce véhicule (prix, modèle, année, kilométrage, etc.), ainsi que la
réponse exacte attendue, basée uniquement sur les informations ci-dessus.

Réponds strictement au format JSON suivant, sans texte additionnel :
{{"question": "...", "ground_truth": "..."}}
""")

parser = JsonOutputParser()
generation_chain = generation_prompt | llm_generator | parser

testset = []

print(f"Génération de {len(echantillon)} question(s) de test...")

for doc in echantillon:
    try:
        result = generation_chain.invoke({"document": doc.page_content})
        testset.append({
            "question": result["question"],
            "ground_truth": result["ground_truth"]
        })
        print(f"  - {result['question']}")
    except Exception as e:
        print(f"  Erreur sur un document, ignoré : {e}")

# Sauvegarde du jeu de test généré (réutilisable sans regénérer à chaque fois)
with open("testset_ragas.json", "w", encoding="utf-8") as f:
    json.dump(testset, f, ensure_ascii=False, indent=2)

print(f"\n{len(testset)} question(s) générée(s) et sauvegardée(s) dans testset_ragas.json")


# ============================================================
# 2. Faire tourner le RAG sur chaque question du jeu de test
#    et collecter : réponse générée + contexte récupéré
# ============================================================

print("\nExécution du RAG sur le jeu de test...")

testset = json.load(open("testset_ragas.json"))
resultats = []

for item in testset:
    question = item["question"]
    docs = ensemble_retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    reponse = rag_chain.invoke(question)

    resultats.append({
        "question": question,
        "answer": reponse,
        "contexts": contexts,
        "ground_truth": item["ground_truth"]
    })

json.dump(resultats, open("eval_data.json", "w"), ensure_ascii=False, indent=2)
print("Données d'évaluation sauvegardées dans eval_data.json")
