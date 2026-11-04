"""Import page view module for the RAG Streamlit application.

This module provides user interface controls and background process logic
for uploading, converting and indexing documents (PDF, DOCX, MD) and Python scripts
into the vector database with live progress bars and cancelation controls.
"""

import time
from pathlib import Path
import streamlit as st
from pipeline.converter import _convert_pdf


def _make_progress_callback(lang, widgets):
    """Creates a callback function to update progress bars and estimated time remaining.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
        widgets (dict): Dictionary containing Streamlit UI elements for progress rendering.

    Returns:
        callable: A function that updates the progress bars and messages for text and image conversion phases.
    """
    start_time = None

    def update_progress(phase, progress, msg, current_img=0, total_imgs=0):
        nonlocal start_time
        if st.session_state.cancel_import:
            raise InterruptedError("Import canceled")

        progress = max(0.0, min(float(progress), 1.0))

        if phase == "text":
            widgets["text_bar"].progress(progress)
            widgets["text_status"].text(msg)
        elif phase == "images":
            widgets["img_bar"].progress(progress)
            widgets["img_status"].text(msg)

            if total_imgs > 0:
                if start_time is None:
                    start_time = time.time()
                if current_img > 1:
                    elapsed = time.time() - start_time
                    avg_per_img = elapsed / (current_img - 1)
                    remaining_seconds = int(avg_per_img * (total_imgs - current_img + 1))
                else:
                    remaining_seconds = total_imgs * 8

                hours, rest = divmod(remaining_seconds, 3600)
                mins, secs = divmod(rest, 60)
                if hours > 0:
                    time_str = lang["eta_hour"].format(hours=hours, mins=mins, secs=secs)
                elif mins > 0:
                    time_str = lang["eta_min"].format(mins=mins, secs=secs)
                else:
                    time_str = lang["eta_sec"].format(secs=secs)

                widgets["eta"].info(time_str)

    return update_progress


def _process_file(uploaded_file, custom_tag, padding_size, update_progress, status, lang):
    """Parses and indexes a single uploaded file into the vector store.
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file object.
        custom_tag (str): Optional category tag provided by the user.
        padding_size (int): Padding in pixels added around extracted images.
        update_progress (callable): Callback function for updating progress UI.
        status (StatusContainer): Streamlit status container for rendering state messages.
        lang (dict): The UI dictionary containing bilingual text strings.

    Returns: 
        int: The number of text/code chunks successfully created and stored.
    """
    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    store = st.session_state.vector_store

    # Handle Python scripts directly without Docling or visual image extraction
    if suffix == ".py":
        update_progress("text", 0.5, lang["splitting_py"])
        code = file_bytes.decode("utf-8", errors="replace")

        num_chunks = store.add_code_with_chunking(
            code=code,
            file_name=uploaded_file.name,
            file_bytes=file_bytes,
            custom_tag=custom_tag
        )

        update_progress("text", 1.0, lang["split_done"])
        update_progress("images", 1.0, lang["no_images_py"])
        return num_chunks

    # Convert and analyze PDF, DOCX and MD files via Docling pipeline
    doc_obj, image_infos = _convert_pdf(
        file_bytes,
        file_name=uploaded_file.name,
        status_callback=update_progress,
        padding=padding_size
    )

    status.update(label=lang["saving"].format(name=uploaded_file.name), state="running")

    return store.add_document_with_chunking(
        doc=doc_obj,
        file_name=uploaded_file.name,
        image_infos=image_infos,
        file_bytes=file_bytes,
        custom_tag=custom_tag
    )


def _run_import(uploaded_files, custom_tag, padding_size, lang):
    """Iterates over all uploaded files to process and store them sequentially.
    
    Args:
        uploaded_files (list): List of uploaded file objects from Streamlit.
        custom_tag (str): Optional category tag to assign to all files in the batch, provided by the user.
        padding_size (int): Padding around extracted images in pixels.
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    st.session_state.import_state = "RUNNING"
    total_chunks = 0
    message_collector = []

    try:
        for uploaded_file in uploaded_files:
            if st.session_state.cancel_import:
                message_collector.append(("info", lang["canceled"]))
                break

            st.markdown(f"### {lang['doc_file']}: '{uploaded_file.name}'")

            with st.status(lang["processing"].format(name=uploaded_file.name), expanded=True) as status:
                # Initialize UI progress tracking elements
                st.caption(lang["step_text"])
                text_bar = st.progress(0.0)
                text_status = st.empty()

                st.caption(lang["step_images"])
                img_bar = st.progress(0.0)
                img_status = st.empty()
                eta = st.empty()

                update_progress = _make_progress_callback(lang, {
                    "text_bar": text_bar,
                    "text_status": text_status,
                    "img_bar": img_bar,
                    "img_status": img_status,
                    "eta": eta
                })

                try:
                    num_chunks = _process_file(uploaded_file, custom_tag, padding_size, update_progress, status, lang)

                except Exception as e:
                    status.update(label=lang["error_label"].format(name=uploaded_file.name), state="error")
                    err_msg = lang["error_file"].format(name=uploaded_file.name, err=e)
                    st.error(err_msg)
                    message_collector.append(("error", err_msg))
                else:
                    total_chunks += num_chunks
                    eta.empty()
                    status.update(label=lang["done"].format(name=uploaded_file.name), state="complete", expanded=False)
                    succ_msg = lang["success_file"].format(name=uploaded_file.name, n=num_chunks)
                    st.success(succ_msg)
                    message_collector.append(("success", succ_msg))

        tot_msg = lang["total"].format(n=total_chunks)
        st.info(tot_msg)
        message_collector.append(("info", tot_msg))

    finally:
        st.session_state.import_messages = message_collector
        st.session_state.import_state = "IDLE"
        st.session_state.cancel_import = False
        st.rerun()


def render(lang, import_running):
    """Renders the document import page view and manages file submission events.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
        import_running (bool): Whether an import process is currently active or not.
    """
    st.title(lang["import_title"])
    st.write(lang["import_intro"])

    col_1, col_2 = st.columns([1, 1])

    with col_1:
        st.subheader(lang["import_param"])
        padding_size = st.slider(lang["padding_size"], min_value=5, max_value=100, value=25, step=1, disabled=import_running)
        custom_tag = st.text_input(lang["tag_label"], value="", disabled=import_running)

    with col_2:
        st.subheader(lang["upload_header"])
        uploaded_files = st.file_uploader(
            lang["upload_label"],
            type=["pdf", "docx", "py", "md"],
            accept_multiple_files=True,
            disabled=import_running
        )

        col_btn_process, col_btn_cancel = st.columns([1, 1])

        with col_btn_process:
            process_clicked = st.button(lang["btn_process"], disabled=import_running)
        with col_btn_cancel:
            cancel_clicked = st.button(lang["btn_cancel"], disabled=not import_running)

        if cancel_clicked:
            st.session_state.cancel_import = True
            st.session_state.import_state = "IDLE"
            st.session_state.import_messages = [("warning", lang["canceled"])]
            st.rerun()

        if process_clicked:
            if not uploaded_files:
                st.warning(lang["no_files"])
            else:
                st.session_state.cancel_import = False
                st.session_state.import_messages = []
                st.session_state.import_state = "START"
                st.rerun()

    # Display status messages from the completed import run
    if st.session_state.import_state == "IDLE":
        for msg_type, msg_text in st.session_state.import_messages:
            getattr(st, msg_type)(msg_text)

    # Execute active import batch task
    if st.session_state.import_state in ("START", "RUNNING"):
        _run_import(uploaded_files, custom_tag, padding_size, lang)