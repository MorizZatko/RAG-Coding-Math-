"""RAG engine module for prompt construction and Ollama LLM interactions.

This module defines system prompts, localization hints and the RAG_Engine class
responsible for formatting retrieved context chunks and querying local Ollama models
to generate accurate answers.
"""
import re
import ollama
from typing import List, Dict
from config import HISTORY_MAX_MESSAGES, HISTORY_TOKEN_SHARE, HISTORY_MESSAGE_MAX_CHAR

PROMPTS = {
     "de": """Du bist ein präziser Wissens- und Code-Assistent.
        Beantworte die Frage des Nutzers ausschließlich basierend auf dem folgenden Kontext aus seinen Dokumenten und Skripten.
        Nutze kein eigenes Vorwissen. Falls die Information nicht im Kontext steht, sage ehrlich, dass du es nicht weißt.
        Falls Diagramme oder Bilder im Kontext vorhanden sind, erkläre deren Inhalt im Text.
        Erfinde keine Fakten. Antworte immer auf Deutsch, auch wenn der Kontext englisch ist.
        Schreibe mathematische Formeln in LaTeX mit $...$ (inline) und $$...$$ (eigene Zeile). Verwende nie \\[ \\] oder eckige Klammern für Formeln.
        {image_hint}
        Bildbeschreibungen stammen aus einer automatischen Bildanalyse und können ungenau sein. Text und Bildunterschriften haben Vorrang.
        Nenne keine Quellennummern, sondern verweise mit Dateiname und Seite.
        Behalte englische Fachbegriffe bei (z. B. bandwidth, acquisition function).
        Ein vorheriger Gesprächsverlauf dient nur dazu, Folgefragen zu verstehen. Fakten dürfen ausschließlich aus dem folgenden Kontext stammen.
        Kontext: {context_str}""",
    "en": """You are a precise knowledge and code assistant.
        Answer the user's question exclusively based on the following context from their documents and scripts.
        Do not use your own prior knowledge. If the information is not in the context, honestly say that you don't know.
        If diagrams or images are present in the context, explain their content in the text.
        Do not invent facts. Always answer in English, even if the context is in German.
        Write math formulas in LaTeX using $...$ (inline) and $$...$$ (own line). Never use \\[ \\] or plain brackets for formulas.
        {image_hint}
        Image descriptions are generated via automated image analysis and may be inaccurate. The text and image captions take precedence.
        Do not cite source numbers; instead, refer to the file name and page.
        A previous conversation history only serves to understand follow-up questions. Facts must come exclusively from the following context.
        Context: {context_str}"""
}

IMAGE_HINT = {
    "de": "Unter deiner Antwort wird automatisch die passende Abbildung angezeigt. Sage nie, dass du keine Bilder zeigen kannst, sondern verweise auf die Abbildung unter der Antwort.",
    "en": "The matching figure is automatically shown below your answer. Never say that you cannot show images; refer to the figure below your answer instead."
}

NO_IMAGE_HINT = {
    "de": "Zu dieser Antwort wird keine Abbildung angezeigt. Wenn der Nutzer ein Bild möchte, sage, dass dazu keine passende Abbildung gefunden wurde.",
    "en": "No figure is shown for this answer. If the user asks for an image, say that no matching figure was found."
}

NO_RESULT = {
    "de": "Dazu habe ich in den Dokumenten nichts gefunden.",
    "en": "I couldn't find anything about this in the documents."
}

OLLAMA_ERROR = {
    "de": "Ollama-Fehler: {err}. Prüfe, ob Ollama installiert ist und läuft.",
    "en": "Ollama error: {err}. Check whether Ollama is installed and running."
}

REWRITE_PROMPT = {
    "de": """Formuliere die letzte Frage des Nutzers zu einer eigenständigen Suchanfrage um, die ohne den Gesprächsverlauf verständlich ist.
        Löse Verweise wie "das", "es", "dort", "Bild 2" oder "die Funktion" anhand des Verlaufs auf.
        Ändere die Bedeutung nicht und beantworte die Frage nicht. Gib ausschließlich die umformulierte Frage in einer Zeile aus.
        Gesprächsverlauf:
        {history}
        Letzte Frage: {question}
        Eigenständige Frage:""",
    "en": """Rewrite the user's last question into a standalone search query that can be understood without the conversation history.
        Resolve references such as "it", "that", "there", "figure 2" or "the function" using the history.
        Do not change the meaning and do not answer the question. Output only the rewritten question on a single line.
        Conversation history:
        {history}
        Last question: {question}
        Standalone question:"""
}

ROLE_LABEL = {
    "de": {"user": "Nutzer", "assistant": "Assistent"},
    "en": {"user": "User", "assistant": "Assistant"}
}

# Reasoning models (e.g. qwen3) may put their thoughts into the answer text
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)

# Rough character-per-token ratio used to size the chat history budget
_CHARS_PER_TOKEN = 3.5


def strip_think(text: str):
    """Removes <think>...</think> reasoning blocks from a model output.

    Args:
        text (str): The raw model output.

    Returns:
        str: The output without reasoning blocks. If nothing would be left,
            the original text is returned, so an answer is never lost completely.
    """
    text = text or ""
    cleaned = text
    # Some chat templates only emit the closing tag, because the opening tag is part of the prompt
    if "</think>" in cleaned and "<think>" not in cleaned.lower():
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    cleaned = _THINK_BLOCK.sub("", cleaned).strip()
    return cleaned or text.strip()


def _truncate(text: str, max_chars: int):
    """Shortens a text to a maximum length and marks the cut with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def _clean_history(history: List[Dict]):
    """Reduces stored chat messages to plain role/content pairs without reasoning blocks.

    Args:
        history (list): Message dictionaries (role, content, optional images/context).

    Returns:
        list: A list of {"role", "content"} dictionaries with non-empty content.
    """
    cleaned = []
    for msg in history or []:
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = strip_think(msg.get("content", ""))
        if content:
            cleaned.append({"role": msg["role"], "content": content})
    return cleaned


def build_history_messages(history: List[Dict], num_ctx: int, max_messages: int = HISTORY_MAX_MESSAGES):
    """Selects the most recent messages that fit into the history budget of the context window.

    The history gets only a fraction of the context window, because the retrieved
    chunks and the system prompt must stay complete. Ollama would otherwise silently
    cut off the beginning of the prompt, including the system instructions.

    Args:
        history (list): Previous messages of the chat, oldest first.
        num_ctx (int): Context window size in tokens.
        max_messages (int): Maximum number of previous messages to include.

    Returns:
        list: Messages for the Ollama chat API, starting with a user and ending with an assistant message.
    """
    budget = int(num_ctx * HISTORY_TOKEN_SHARE * _CHARS_PER_TOKEN)
    messages = _clean_history(history)[-max_messages:]

    selected = []
    used = 0
    # Newest messages are the most relevant ones, so fill the budget from the end
    for msg in reversed(messages):
        remaining = budget - used
        if remaining < 80:
            break
        content = _truncate(msg["content"], min(HISTORY_MESSAGE_MAX_CHAR, remaining))
        selected.append({"role": msg["role"], "content": content})
        used += len(content)
    selected.reverse()

    # A valid history starts with the user and ends with an answer, because the new question follows
    while selected and selected[0]["role"] != "user":
        selected.pop(0)
    while selected and selected[-1]["role"] != "assistant":
        selected.pop()
    return selected


class RAG_Engine:
    """Manage prompt construction and communication with local Ollama models."""
    def __init__(self, default_model: str = "qwen2.5-coder:7b"):
        self.default_model = default_model

    def rewrite_query(self, prompt: str, history: List[Dict], model_name: str = None, language: str = "de", num_ctx: int = 4096):
        """Rewrites a follow-up question into a standalone search query using the chat history.
    
        Without this step, questions like "Explain that in more detail" would produce
        useless vector search results. Without history the question is returned unchanged.
        On any failure the original question is used, so retrieval keeps working.
    
        Args:
            prompt (str): The user's current question.
            history (list): Previous messages of the chat, oldest first.
            model_name (str, optional): Name of the Ollama model to use.
            language (str): Language code ('de' or 'en').
            num_ctx (int): Context window size. Must match the answer call, because a different value makes Ollama reload the model.
    
        Returns:
            str: The standalone search query.
        """
        model = model_name or self.default_model
        language = language if language in REWRITE_PROMPT else "de"
    
        turns = _clean_history(history)[-4:]
        if not turns:
            return prompt
    
        transcript = "\n".join(
            f"{ROLE_LABEL[language][t['role']]}: {_truncate(t['content'], 400)}" for t in turns
        )

        instruction = REWRITE_PROMPT[language].format(history=transcript, question=prompt)
        messages = [{"role": "user", "content": instruction}]
        options = {"temperature": 0, "num_predict": 200, "num_ctx": num_ctx}
    
        try:
            try:
                # Switch off reasoning, so the short token budget is not used up by thoughts
                response = ollama.chat(model=model, messages=messages, options=options, think=False)
            except TypeError:
                # Older versions of the ollama package do not know the 'think' parameter
                response = ollama.chat(model=model, messages=messages, options=options)
            text = strip_think(response["message"]["content"])
        except Exception as e:
            print(f"DEBUG: Query rewrite failed: {e}")
            return prompt
    
        # Output that consists only of reasoning is useless as a search query
        if "think>" in text.lower():
            return prompt
    
        lines = text.splitlines()
        rewritten = lines[0].strip().strip("\"'`“”„") if lines else ""
        if not rewritten or len(rewritten) > 500:
            return prompt
    
        print(f"DEBUG: Query rewritten: '{prompt}' -> '{rewritten}'")
        return rewritten

    def get_answer(self, prompt: str, retrieved_chunks: List[Dict], model_name: str = None, temperature: float = 0.2, num_ctx: int = 4096, language: str = "de", has_images: bool = False, history: List[Dict] = None):
        """Builds context from ChromaDB chunks and queries the Ollama model for an answer.

        Args:
            prompt (str): The user's input question.
            retrieved_chunks (list): List of retrieved chunk dictionaries.
            model_name (str, optional): Name of the Ollama model to use.
            temperature (float): Sampling temperature for generation.
            num_ctx (int): Context window token size.
            language (str): Language code ('de' or 'en').
            has_images (bool): Whether visual images are available for the response.
            history (list, optional): Previous messages of the chat, oldest first.

        Returns:
            tuple: A tuple containing (answer_string, context_string).
        """
        model = model_name or self.default_model
        language = language if language in PROMPTS else "de"

        image_hint = (IMAGE_HINT if has_images else NO_IMAGE_HINT)[language]

        # Return early if no relevant context chunks were retrieved
        if not retrieved_chunks:
            return NO_RESULT[language], ""

        # Construct the context string from retrieved chunks
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "?")
            context_str += f"\n--- {source}, page: {page} ---\n{chunk['content']}\n"

        system_prompt = PROMPTS[language].format(context_str=context_str, image_hint=image_hint)
        history_messages = build_history_messages(history, num_ctx)

        # Send request to Ollama chat model
        try:
            response = ollama.chat(
                model=model,
                options={
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "repeat_penalty": 1.1
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history_messages,
                    {"role": "user", "content": prompt}
                ]
            )
            answer = strip_think(response["message"]["content"])
            return answer, context_str

        except Exception as e:
            raise RuntimeError(OLLAMA_ERROR[language].format(err=e))

