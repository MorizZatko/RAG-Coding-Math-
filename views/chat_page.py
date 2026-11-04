"""Chat view module for the RAG Streamlit application.

This module handles the interactive chat interface, including chat history,
model settings configuration in the sidebar, RAG retrieval processing,
answer generation, context source display and associated visual image display.
"""

import os
import time
import streamlit as st
from utils import fix_latex
from config import MAX_DIST, IMAGE_MAX_DIST, N_RESULTS, N_RESULTS_WITH_HISTORY

def _focus_chat_input():
    """Moves the cursor back to the chat-input-field after an answer."""
    st.iframe(
        f"""
        <script>
        // {time.time()} 
        const focusInput = () => {{
            const el = window.parent.document.querySelector('[data-testid="stChatInput"] textarea');
            if (el) el.focus();
        }};
        focusInput();
        setTimeout(focusInput, 150);
        </script>
        """,
        height=1
    )


def _open_chat(chat_id):
    """Switches to a saved chat (or to a fresh, not yet saved chat if chat_id is None).

    Used as button callback, so it runs before the script is re-executed.

    Args:
        chat_id (str | None): The id of the chat to open.
    """
    store = st.session_state.chat_store
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = store.load_messages(chat_id) if chat_id else []
    st.session_state.chat_to_delete = None


def _request_delete(chat_id):
    """Marks a chat for deletion, so a confirmation is shown."""
    st.session_state.chat_to_delete = chat_id


def _cancel_delete():
    """Cancels a pending chat deletion."""
    st.session_state.chat_to_delete = None


def _confirm_delete(chat_id):
    """Deletes a chat and switches to a fresh chat if it was the open one."""
    st.session_state.chat_store.delete_chat(chat_id)
    if st.session_state.get("current_chat_id") == chat_id:
        _open_chat(None)
    st.session_state.chat_to_delete = None


def _rename_chat(chat_id):
    """Applies the title entered in the rename text field to a chat."""
    st.session_state.chat_store.rename_chat(chat_id, st.session_state.get(f"rename_{chat_id}", ""))


def _init_chat_state():
    """Initializes chat state and reopens the most recent chat after an application restart."""
    if "chat_to_delete" not in st.session_state:
        st.session_state.chat_to_delete = None

    if "current_chat_id" not in st.session_state:
        chats = st.session_state.chat_store.list_chats()
        _open_chat(chats[0]["id"] if chats else None)

    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_chat_list(container, lang):
    """Renders the list of saved chats with new, rename and delete controls in the sidebar.

    Args:
        container: A Streamlit container (reserved in the sidebar) to render into.
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    store = st.session_state.chat_store
    chats = store.list_chats()
    current = st.session_state.current_chat_id

    with container:
        st.header(lang["chats_header"])
        st.button(lang["new_chat"], key="chat_new", width="stretch", on_click=_open_chat, args=(None,))

        if not chats:
            st.caption(lang["no_chats"])

        # Scrollable box only for long lists; height=None is not accepted by newer Streamlit versions
        list_box = st.container(height=300) if len(chats) > 6 else st.container()
        with list_box:
            for chat in chats:
                c_open, c_del = st.columns([5, 1])
                c_open.button(
                    chat["title"] or lang["untitled_chat"],
                    key=f"chat_open_{chat['id']}",
                    type="primary" if chat["id"] == current else "secondary",
                    width="stretch",
                    on_click=_open_chat,
                    args=(chat["id"],)
                )
                c_del.button("✕", key=f"chat_del_{chat['id']}", help=lang["delete_chat"], on_click=_request_delete, args=(chat["id"],))

        pending = st.session_state.chat_to_delete
        pending_chat = store.get_chat(pending) if pending else None

        if pending_chat:
            st.warning(lang["chat_confirm"].format(title=pending_chat["title"] or lang["untitled_chat"]))
            col_yes, col_no = st.columns(2)
            col_yes.button(lang["btn_yes_doc"], key="chat_del_yes", on_click=_confirm_delete, args=(pending,))
            col_no.button(lang["btn_no"], key="chat_del_no", on_click=_cancel_delete)

        current_chat = store.get_chat(current) if current else None
        if current_chat:
            with st.expander(lang["rename_chat"]):
                st.text_input(lang["rename_label"], value=current_chat["title"], key=f"rename_{current}")
                st.button(lang["btn_save"], key=f"rename_save_{current}", on_click=_rename_chat, args=(current,))


def _show_history(lang):
    """Renders the existing chat message history from session state.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(fix_latex(message["content"]))
            for info in message.get("images", []):
                if isinstance(info, dict) and os.path.exists(info["path"]):
                    st.image(info["path"], caption=info["caption"], width=640)
            if message.get("context"):
                with st.expander(lang["sources"]):
                    st.text(message["context"])


def _collect_images(retrieved_chunks, lang):
    """Filters and collects relevant images associated with the top retrieved text chunks.
    
    Args:
        retrieved_chunks (list): A list of retrieved vector store chunk dictionaries.
        lang (dict): The UI dictionary containing bilingual text strings.

    Returns:
        list: A deduplicated list of dictionaries containing image paths and captions.
    """
    # Seperate text chunks from image chunks to identify the top text source
    text_chunks = [c for c in retrieved_chunks if c["metadata"].get("type") != "image"]
    top_source = text_chunks[0]["metadata"].get("source") if text_chunks else None

    image_infos = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        print(f"DEBUG: distance={chunk['distance']:.3f} type={meta.get('type')} source={meta.get('source')} page={meta.get('page')}")
        img_p = meta.get("image_path", "")

        # Include valid images that meet the distance threshold and match the top source
        if (img_p and os.path.exists(img_p) and chunk["distance"] < IMAGE_MAX_DIST and meta.get("source") == top_source):
            image_infos.append({
                "path": img_p,
                "caption": f"{meta.get('source')} - {lang['page']} {meta.get('page')}"
            })

    # Remove duplicate entries while preserving order
    return list({i["path"]: i for i in image_infos}.values())


def _handle_prompt(prompt, lang, language, model_name, temperature, num_ctx):
    """Answers a new user question, displays the answer and saves the exchange to the chat history.

    Args:
        prompt (str): The user's question.
        lang (dict): The UI dictionary containing bilingual text strings.
        language (str): The active language code ('de' or 'en').
        model_name (str): The selected Ollama model.
        temperature (float): Sampling temperature for generation.
        num_ctx (int): Context window size in tokens.
    """
    engine = st.session_state.rag_engine

    # Previous messages of this chat give the model and the retrieval their conversational context
    history = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(lang["spinner"]):
            try:
                # Turn follow-up questions into standalone search queries (no-op without history)
                search_query = engine.rewrite_query(
                    prompt, history, model_name=model_name, language=language, num_ctx=num_ctx
                )

                # Retrieve similar chunks from the vector store and filter by maximum distance
                n_results = N_RESULTS_WITH_HISTORY if history else N_RESULTS
                retrieved_chunks = st.session_state.vector_store.search_similar(search_query, n_results=n_results)
                retrieved_chunks = [c for c in retrieved_chunks if c["distance"] is not None and c["distance"] < MAX_DIST]

                # Collect relevant visual images for the response
                image_infos = _collect_images(retrieved_chunks, lang)

                # Filter out image-chunks whose figures are not actually displayed
                valid_img_paths = {info["path"] for info in image_infos}
                active_chunks = [
                    c for c in retrieved_chunks
                    if c["metadata"].get("type") != "image" or c["metadata"].get("image_path") in valid_img_paths
                ]

                # Generate RAG answer using the engine
                answer, context_str = engine.get_answer(
                    prompt=prompt,
                    retrieved_chunks=active_chunks,
                    model_name=model_name,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    language=language,
                    has_images=bool(image_infos),
                    history=history
                )

                # Render the answer markdown with fixed LaTeX math formatting
                st.markdown(fix_latex(answer))

                # Display visual sources if available
                if image_infos:
                    st.markdown("---")
                    st.subheader(lang["visual"])
                    for info in image_infos:
                        st.image(info["path"], caption=info["caption"], width=640)

                # Provide an expander to inspect raw context sources
                with st.expander(lang["sources"]):
                    st.text(context_str)

                # Save question and answer together, so a failed answer never leaves a lone question behind
                chat_id = st.session_state.current_chat_id
                if chat_id is None:
                    chat_id = st.session_state.chat_store.create_chat(model=model_name)
                    st.session_state.current_chat_id = chat_id
                st.session_state.chat_store.append_exchange(
                    chat_id, prompt, answer, images=image_infos, context=context_str
                )

                # Append assistant response, images and context sources to the displayed history
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "images": image_infos, "context": context_str
                })

            except Exception as e:
                # No answer was saved, so the question is removed from the history as well
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
                st.error(str(e))


def render(lang, language):
    """Renders the main RAG chat page view and handles user interaction.
    
    Args:
        lang (dict): The UI dictionary containing bilingual text strings.
        language (str): The active language code ('de' or 'en').
    
    """
    st.title(lang["chat_title"])

    _init_chat_state()

    # Reserve the top of the sidebar for the chat list, which is filled after the answer was saved
    chat_list_slot = st.sidebar.container()

    # Configure model parameters via sidebar widgets
    with st.sidebar:
        st.header(lang["model_settings"])
        model_name = st.selectbox(lang["ollama_model"], ["qwen2.5-coder:7b", "qwen3:8b", "qwen3:14b", "gemma2:9b", ], index=0)
        temperature = st.slider(lang["temperature"], min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        num_ctx = st.select_slider(lang["context_window"], options=[2048, 4096, 8192, 16384], value=8192)

    # Display previous chat messages
    _show_history(lang)

    # Process new user prompt input
    prompt = st.chat_input(lang["input"])
    if prompt:
        _handle_prompt(prompt, lang, language, model_name, temperature, num_ctx)

    # Render the chat list last, so a chat created by this answer is already listed
    _render_chat_list(chat_list_slot, lang)

    if prompt:
        _focus_chat_input()