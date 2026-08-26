"""
app.py — Interface Streamlit pour discuter avec l'agent RAG véhicules.

Lancement :
    streamlit run app.py
"""

import uuid
import streamlit as st
from rag_core import build_agent, ask_agent

st.set_page_config(page_title="Assistant Véhicules", page_icon="🚗", layout="centered")


# ============================================================
# Chargement de l'agent — UNE SEULE FOIS par session serveur
# ============================================================

@st.cache_resource(show_spinner="Chargement de l'agent (indexation si nécessaire)...")
def get_agent():
    return build_agent()


agent = get_agent()


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

    historique_precedent = st.session_state.messages[:-1]

    # Appelle l'agent et affiche la réponse
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            try:
                reponse = ask_agent(agent, question, session_id=st.session_state.session_id)
            except ValueError as e:
                reponse = f"Question invalide : {e}"
            except Exception as e:
                reponse = f"Une erreur est survenue : {e}"

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