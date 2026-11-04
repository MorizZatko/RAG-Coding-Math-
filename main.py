"""Main entry point for the RAG Streamlit pipeline.

This module initializes the Streamlit page configuration, manages
session states for the vector store, RAG engine, import process,
and handles the sidebar navigation and page routing.
"""

import torch.nn as nn
import sys
# Workaround to enable GPU usage
sys.modules['torch.nn'] = nn
import streamlit as st
from rag.vectors_storage import VectorStoreManager
from rag.engine import RAG_Engine
from rag.chat_storage import ChatStore
from language import UI
from views import import_page, chat_page, manage_page


# Initialize the Streamlit page layout and title
st.set_page_config(page_title="Coding & Math RAG", layout="wide")

# Initialize vector store manager
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

# Initialize RAG engine
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAG_Engine()

# Initialize persistent chat history storage
if "chat_store" not in st.session_state:
    st.session_state.chat_store = ChatStore()

# Initialize import process and control flags
if "import_state" not in st.session_state:
    st.session_state.import_state = "IDLE"
if "cancel_import" not in st.session_state:
    st.session_state.cancel_import = False
if "import_messages" not in st.session_state:
    st.session_state.import_messages = []

# Check if an import process is currently active
import_running = st.session_state.import_state in ("START", "RUNNING")

# Render sidebar title
st.sidebar.title("Coding & Math RAG")

# Render language selection in the sidebar
language_label = st.sidebar.radio("Language / Sprache", ["Deutsch", "English"], key="language_label", disabled=import_running)
language = "de" if language_label == "Deutsch" else "en"
lang = UI[language]

# Render sidebar navigation controls
st.sidebar.title(lang["nav_title"])
page = st.sidebar.radio(
    lang["nav_label"],
    ["import", "chat", "manage"],
    format_func=lambda k: lang[f"nav_{k}"],
    key="page",
    disabled=import_running
)

# Display a sidebar notification if navigation is locked due to an active import
if import_running:
    st.sidebar.info(lang["nav_locked"])

# Route the user to the selected view page
if page == "import":
    import_page.render(lang, import_running)
elif page == "chat":
    chat_page.render(lang, language)
elif page == "manage":
    manage_page.render(lang)