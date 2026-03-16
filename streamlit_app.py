import streamlit as st
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipediaapi

# Configuration de la page
st.set_page_config(page_title="Kraken_bot", page_icon="🐙")
st.title("🐙 Kraken_bot - Assistant intelligent qui apprend (version légère)")

# Fichier de connaissances
KNOWLEDGE_FILE = "knowledge.json"

def load_knowledge():
    """Charge la base de connaissances depuis le JSON."""
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as f:
            return json.load(f)
    return {"questions": [], "reponses": []}

def save_knowledge(knowledge):
    """Sauvegarde la base de connaissances."""
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(knowledge, f)

knowledge = load_knowledge()

# Initialisation du vectoriseur TF‑IDF (en cache)
@st.cache_resource
def init_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), analyzer="char", lowercase=True)

vectorizer = init_vectorizer()

# Variable pour stocker la matrice TF‑IDF des questions connues
if "tfidf_matrix" not in st.session_state:
    if knowledge["questions"]:
        st.session_state.tfidf_matrix = vectorizer.fit_transform(knowledge["questions"])
    else:
        st.session_state.tfidf_matrix = None

# API Wikipedia
wiki = wikipediaapi.Wikipedia('Kraken_bot (contact@example.com)', 'fr')

def rechercher_wikipedia(question):
    """Extrait un sujet de la question et retourne un résumé Wikipedia."""
    mots_interrogatifs = ['qui', 'est', 'que', 'qu', 'quoi', 'comment', 'pourquoi', 'où', 'quand']
    mots = question.lower().split()
    sujet = None
    for mot in mots:
        if mot not in mots_interrogatifs and len(mot) > 2:
            sujet = mot
            break
    if not sujet:
        return None
    page = wiki.page(sujet.capitalize())
    if page.exists():
        return f"🔍 Selon Wikipedia : {page.summary[:500]}..."
    return None

def get_best_answer(question, threshold=0.5):
    """Cherche la question la plus similaire dans la base."""
    if not knowledge["questions"] or st.session_state.tfidf_matrix is None:
        return None
    # Transformer la nouvelle question
    q_vec = vectorizer.transform([question])
    # Calculer la similarité avec toutes les questions connues
    sim = cosine_similarity(q_vec, st.session_state.tfidf_matrix)[0]
    best_idx = np.argmax(sim)
    if sim[best_idx] >= threshold:
        return knowledge["reponses"][best_idx]
    return None

def learn_new(question, answer):
    """Ajoute une nouvelle question/réponse à la base et met à jour la matrice TF‑IDF."""
    knowledge["questions"].append(question)
    knowledge["reponses"].append(answer)
    # Recalculer la matrice TF‑IDF avec toutes les questions (y compris la nouvelle)
    st.session_state.tfidf_matrix = vectorizer.fit_transform(knowledge["questions"])
    save_knowledge(knowledge)

# Interface de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Saisie utilisateur
if prompt := st.chat_input("Pose ta question à Kraken_bot"):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Kraken_bot réfléchit..."):
            # Chercher une réponse dans la base
            answer = get_best_answer(prompt)
            if answer:
                final_response = answer
            else:
                # Sinon, interroger Wikipedia
                wiki_res = rechercher_wikipedia(prompt)
                if wiki_res:
                    final_response = wiki_res
                    # Apprendre cette nouvelle information
                    learn_new(prompt, wiki_res)
                else:
                    final_response = "Je ne sais pas répondre à ça pour l'instant. Tu peux m'apprendre en modifiant le code ou en utilisant la version locale."
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
