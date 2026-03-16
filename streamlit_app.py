import streamlit as st
import logging
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import wikipediaapi
import os

# Configuration
logging.basicConfig(level=logging.INFO)

# Titre de l'application
st.set_page_config(page_title="Kraken_bot", page_icon="🐙")
st.title("🐙 Kraken_bot - Assistant intelligent qui apprend")

# Initialisation du bot (mise en cache pour ne pas le recharger à chaque interaction)
@st.cache_resource
def init_bot():
    bot = ChatBot(
        'Kraken_bot',
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///kraken_database.sqlite3',  # Fichier local
        logic_adapters=[
            'chatterbot.logic.BestMatch',
            'chatterbot.logic.MathematicalEvaluation',
            'chatterbot.logic.TimeLogicAdapter'
        ],
        read_only=False
    )
    # Entraînement initial (si la base est vide)
    trainer = ChatterBotCorpusTrainer(bot)
    try:
        trainer.train("chatterbot.corpus.french")
    except:
        trainer.train("chatterbot.corpus.english")
    return bot

kraken = init_bot()

# Initialisation de Wikipedia API
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

# Gestion de l'historique de la conversation dans st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Pose ta question à Kraken_bot"):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Obtenir la réponse du bot
    with st.chat_message("assistant"):
        with st.spinner("Kraken_bot réfléchit..."):
            response = kraken.get_response(prompt)
            if response.confidence < 0.5:
                wiki_reponse = rechercher_wikipedia(prompt)
                if wiki_reponse:
                    final_response = wiki_reponse
                    # Apprendre cette nouvelle info
                    kraken.learn_response(wiki_reponse, prompt)
                else:
                    # Demander à l'utilisateur d'enseigner
                    final_response = "Je ne sais pas répondre à ça. Tu peux m'apprendre en modifiant le code ou en utilisant la version locale."
                    # Ici on pourrait stocker la question pour apprentissage différé
            else:
                final_response = response.text
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
