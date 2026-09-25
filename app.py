"""
app.py — Interface Streamlit pour discuter avec l'agent RAG véhicules.

Simple client de l'API (api.py) : aucune logique RAG ici.

Lancement :
    uvicorn api:app          # dans un terminal
    streamlit run app.py     # dans un autre

L'adresse de l'API se règle avec la variable d'environnement API_URL
(http://localhost:8000 par défaut).
"""

import os
import uuid

import requests
import streamlit as st
from streamlit_js_eval import get_geolocation

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT_S = 120

position = get_geolocation()

st.set_page_config(page_title="Assistant Véhicules", page_icon="🚗", layout="centered")


def appeler_api(question, historique, session_id, position):
    payload = {"question": question, "history": historique, "session_id": session_id}
    if position and position.get("coords"):
        payload["location"] = {
            "latitude": position["coords"]["latitude"],
            "longitude": position["coords"]["longitude"],
        }

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json=payload,
            headers={"X-Client": "streamlit"},
            timeout=API_TIMEOUT_S,
        )
    except requests.Timeout:
        return "Le service met trop de temps à répondre. Réessaie dans un instant."
    except requests.ConnectionError:
        return "Le service est injoignable pour le moment."

    if response.status_code == 400:
        return f"Question invalide : {response.json()['detail']}"
    if response.status_code == 422:
        return "Question invalide (trop longue ou mal formée)."
    if not response.ok:
        return f"Une erreur est survenue. Référence : {response.headers.get('X-Request-ID', 'inconnue')}"
    return response.json()["answer"]


# ============================================================
# État de session — historique de conversation + id de session
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ============================================================
# Interface
# ============================================================

st.title("🚗 Assistant Véhicules")
st.caption("Pose une question sur les véhicules, la météo, ou le prix du carburant.")

# Affiche l'historique existant
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
question = st.chat_input("Écris ta question ici...")

if question:
    # Affiche immédiatement le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Historique précédent (hors le nouveau message, déjà ajouté juste au-dessus),
    # limité aux 20 derniers messages acceptés par l'API
    historique_precedent = st.session_state.messages[:-1][-20:]

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            reponse = appeler_api(
                question,
                historique_precedent,
                st.session_state.session_id,
                position,
            )
        st.markdown(reponse)

    st.session_state.messages.append({"role": "assistant", "content": reponse})


# ============================================================
# Sidebar — infos utiles
# ============================================================

with st.sidebar:
    st.subheader("À propos")
    st.markdown(
        "Cet assistant peut répondre à des questions sur :\n"
        "- 🚙 Les véhicules d'occasion (prix, modèle, année...)\n"
        "- 🌦️ La météo actuelle\n"
        "- ⛽ Le prix du carburant en France"
    )

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()
