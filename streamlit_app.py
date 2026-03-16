import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import wikipediaapi
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
st.set_page_config(page_title="Kraken_bot", page_icon="🐙")
st.title("🐙 Kraken_bot - Assistant intelligent qui apprend")

# Initialisation du modèle d'embedding (mis en cache)
@st.cache_resource
def load_encoder():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Modèle multilingue

encoder = load_encoder()

# Initialisation de la base de connaissances (fichier JSON)
KNOWLEDGE_FILE = "knowledge.json"

def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as f:
            return json.load(f)
    else:
        return {"q": [], "a": []}  # Listes de questions et réponses

def save_knowledge(knowledge):
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(knowledge, f)

knowledge = load_knowledge()

# Pré-calcul des embeddings pour les questions connues (pour accélérer)
if "embeddings" not in st.session_state:
    if knowledge["q"]:
        st.session_state.embeddings = encoder.encode(knowledge["q"])
    else:
        st.session_state.embeddings = np.array([])

# API Wikipedia
wiki = wikipediaapi.Wikipedia('Kraken_bot (contact@example.com)', 'fr')

def rechercher_wikipedia(question):
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

def get_best_answer(question, threshold=0.7):
    if not knowledge["q"]:
        return None
    # Encoder la question
    q_emb = encoder.encode([question])
    # Calculer similarité avec toutes les questions connues
    sim = cosine_similarity(q_emb, st.session_state.embeddings)[0]
    best_idx = np.argmax(sim)
    if sim[best_idx] >= threshold:
        return knowledge["a"][best_idx]
    else:
        return None

def learn_new(question, answer):
    knowledge["q"].append(question)
    knowledge["a"].append(answer)
    # Mettre à jour les embeddings
    new_emb = encoder.encode([question])
    if st.session_state.embeddings.size == 0:
        st.session_state.embeddings = new_emb
    else:
        st.session_state.embeddings = np.vstack([st.session_state.embeddings, new_emb])
    save_knowledge(knowledge)

# Interface de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pose ta question à Kraken_bot"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Kraken_bot réfléchit..."):
            # Chercher une réponse connue
            answer = get_best_answer(prompt)
            if answer:
                final_response = answer
            else:
                # Sinon, chercher sur Wikipedia
                wiki_res = rechercher_wikipedia(prompt)
                if wiki_res:
                    final_response = wiki_res
                    # Apprendre cette nouvelle info
                    learn_new(prompt, wiki_res)
                else:
                    # Demander à l'utilisateur d'enseigner (simulé ici)
                    final_response = "Je ne sais pas encore répondre à ça. Tu peux m'apprendre en modifiant le code ou en utilisant la version locale."
                    # Pour un apprentissage interactif, il faudrait un mécanisme plus complexe
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
