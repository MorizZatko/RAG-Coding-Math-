"""Dataset management view module for the RAG Streamlit application.

This module provides user interface controls for inspecting stored documents,
sorting the documents list, downloading or opening original files, deleting individual documents,
and performing complete database resets.
"""

import os
from urllib.parse import quote
import streamlit as st
from utils import format_date


def _sorted_documents(sort_key):
    """Retrieves and sorts stored documents based on specified criterion.
    
    Args:
        sort_key (str): The sorting key ('date_desc', 'date_asc', 'name' or 'tag').

    Returns:
        list: A sorted list of document metadata dictionaries.
    """
    documents = st.session_state.vector_store.list_documents()

    if sort_key == "date_desc":
        documents.sort(key=lambda d: d["uploaded_at"], reverse=True)
    elif sort_key == "date_asc":
        documents.sort(key=lambda d: d["uploaded_at"])
    elif sort_key == "name":
        documents.sort(key=lambda d: d["name"].lower())
    else:
        documents.sort(key=lambda d: (d["tag"].lower(), d["name"].lower()))

    return documents


def _confirm_delete_document(lang):
    """Renders confirmation controls before deleting a single document.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    pending = st.session_state.get("doc_to_delete")
    if not pending:
        return

    st.warning(lang["doc_confirm"].format(name=pending))
    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button(lang["btn_yes_doc"], key="doc_del_yes"):
            st.session_state.vector_store.delete_document(pending)
            st.session_state.doc_to_delete = None
            st.rerun()

    with col_no:
        if st.button(lang["btn_no"], key="doc_del_no"):
            st.session_state.doc_to_delete = None
            st.rerun()


def _file_cell(doc, index, lang):
    """Renders an open link for PDF files or a download button for other formats.
    
    Args:
        doc (dict): Document metadata dictionary.
        index (int): Unique index for button key generation.
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    path = doc["original_path"]

    if path and os.path.exists(path):
        if path.lower().endswith(".pdf"):
            url = f"/app/static/documents/{quote(os.path.basename(path))}"
            st.markdown(f"[{lang['open']}]({url})")
        else:
            with open(path, "rb") as f:
                st.download_button(lang["download"], f.read(), file_name=doc["name"], key=f"dl_{index}")
    else:
        st.caption("-")


def _render_documents(lang):
    """Renders the document directory table with sorting and deletion controls.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    st.subheader(lang["docs_header"])

    sort_key = st.selectbox(
        lang["sort_label"],
        ["date_desc", "date_asc", "name", "tag"],
        format_func=lambda k: lang[f"sort_{k}"]
    )

    _confirm_delete_document(lang)

    documents = _sorted_documents(sort_key)

    if not documents:
        st.info(lang["no_docs"])
        return

    widths = [4, 2, 3, 2, 2]

    header = st.columns(widths)
    for col, key in zip(header, ["col_name", "col_tag", "col_date", "col_file", ""]):
        col.markdown(f"**{lang[key]}**" if key else "")
    st.markdown("---")

    for i, doc in enumerate(documents):
        c_name, c_tag, c_date, c_file, c_del = st.columns(widths)

        c_name.write(doc["name"])
        c_tag.write(doc["tag"])
        c_date.write(format_date(doc["uploaded_at"]))

        with c_file:
            _file_cell(doc, i, lang)

        with c_del:
            if st.button(lang["btn_del_doc"], key=f"del_{i}"):
                st.session_state.doc_to_delete = doc["name"]
                st.rerun()


def _render_reset(lang):
    """Renders dataset reset controls and confirmation dialogs.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    st.subheader(lang["reset_header"])

    if st.session_state.get("reset_success_msg"):
        st.success(lang["reset_success"])
        st.session_state.reset_success_msg = False

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button(lang["btn_delete"]):
            st.session_state.confirm_reset = True
            st.rerun()
        return

    st.warning(lang["confirm_warning"])
    col_btn_1, col_btn_2 = st.columns(2)

    with col_btn_1:
        if st.button(lang["btn_yes"]):
            st.session_state.vector_store.clear_database()
            st.session_state.confirm_reset = False
            st.session_state.reset_success_msg = True
            st.rerun()

    with col_btn_2:
        if st.button(lang["btn_no"]):
            st.session_state.confirm_reset = False
            st.rerun()


def render(lang):
    """Renders the complete dataset management page view.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    st.title(lang["manage_title"])

    col_docs, col_reset = st.columns([1, 1])

    with col_docs:
        _render_documents(lang)

    with col_reset:
        _render_reset(lang)