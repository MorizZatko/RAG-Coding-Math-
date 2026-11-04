"""Vector store management module for ChromaDB indexing and retrieval.

This module handles vector database initialization, orphan segment cleanup,
original file storage, Python code parsing and chunking, hybrid document chunking via Docling,
similarity searching and complete database reset.
"""

import chromadb
import ast
import glob
import os
import shutil
import gc
import time
import sqlite3
from datetime import datetime
from docling.chunking import HybridChunker
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

DOCS_DIR = os.path.join("static", "documents")
IMAGES_DIR = "extracted_images"


def _safe_filename(name: str):
    """Replaces all characters except letters, digits, '.', '_', '-', for safe paths and URLs.
    
    Args:
        name (str): The original filename string.

    Returns:
        str: The sanitized filename string.
    """
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _rmtree_retry(path: str, tries: int = 5, delay: float = 0.5):
    """Deletes a directory recursivly with retries to handle file locks on Windows.
    
    Args:
        path (str): The file path of the directory to remove.
        tries (int): Maximum number of deletion attempts.
        delay (float): Delay in seconds between retry attempts.

    Returns:
        bool: True if deletion succeeded or path was missing, False otherwise.
    """
    for _ in range(tries):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except Exception:
            gc.collect()
            time.sleep(delay)
    print(f"DEBUG: Could not delete folder: {path}")
    return False


def _split_by_lines(text: str, max_chars: int, overlap_lines: int = 3):
    """Splits text line by line into smaller chunks as a fallback mechanism.

    Args:
        text (str): The input text string to split.
        max_chars (int): Maximum character length per chunk.
        overlap_lines (int): Number of overlapping lines between consecutive chunks.

    Returns:
        list: A list of text chunk strings.
    """
    parts = []
    current = []
    size = 0

    for line in text.splitlines():
        if size + len(line) + 1 > max_chars and current:
            parts.append("\n".join(current))
            current = current[-overlap_lines:]
            size = sum(len(l) + 1 for l in current)
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def _chunk_python_code(code: str, max_chars: int = 1500):
    """Parses and splits Python code into logical chunks using AST analysis.

    Args:
        code (str): The Python source code string.
        max_chars (int): Maximum character length threshold per chunk.

    Returns:
        list: A list of tuples containing (chunk_label, chunk_text, start_line).
    """
    lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [("code", part, 1) for part in _split_by_lines(code, max_chars)]

    chunks = []
    func_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    def segment(node):
        """Determines the start and end line numbers of a node, including decorators and comments."""
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        while start > 1 and lines[start - 2].strip().startswith('#'):
            start -= 1
        return start, node.end_lineno

    def rest_text(start, end, segments):
        """Extracts unsegmented lines within a given range."""
        covered = set()
        for s, e in segments:
            covered.update(range(s, e + 1))
        return "\n".join(lines[i - 1] for i in range(start, end + 1) if i not in covered).strip()

    def add(label, text, start):
        """Adds a chunk, splitting it if the text exceeds the maximum length."""
        parts = _split_by_lines(text, max_chars)
        for i, part in enumerate(parts, 1):
            name = label if len(parts) == 1 else f"{label} (part {i}/{len(parts)})"
            chunks.append((name, part, start))

    top = [(n, *segment(n)) for n in tree.body if isinstance(n, func_types + (ast.ClassDef,))]

    # Extract module-level imports, constants and global code outside of functions and classes
    module_text = rest_text(1, len(lines), [(s, e) for _, s, e in top])
    if module_text:
        add("module level", module_text, 1)

    for node, start, end in top:
        text = "\n".join(lines[start - 1:end])
        is_class = isinstance(node, ast.ClassDef)
        kind = "class" if is_class else "function"

        methods = [(m, *segment(m)) for m in node.body if isinstance(m, func_types)] if is_class else []

        if is_class and len(text) > max_chars and methods:
            # Handle large classes by splitting header and individual methods
            class_head = lines[node.lineno - 1]
            head_text = rest_text(start, end, [(s, e) for _, s, e in methods])
            add(f"class {node.name} (header & attributes)", head_text, start)

            for m, m_start, m_end in methods:
                m_text = "\n".join(lines[m_start - 1:m_end])
                add(f"method {node.name}.{m.name}", f"{class_head}\n    ...\n{m_text}", m_start)
        else:
            add(f"{kind} {node.name}", text, start)

    return chunks


# Define embedding model and tokenizer for hybrid chunking
model = 'BAAI/bge-m3'
chunk_tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(model),
    max_tokens=512
)

class VectorStoreManager:
    def __init__(self, collection_name: str = 'RAG'):
        self.collection_name = collection_name
        self.persist_directory = "./chroma_db"

        # Initialize local ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Define sentence transformer embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model,
            device="cpu",
            )

        # Clean up leftovers from earlier resets
        self._remove_orphan_segments()

        # Load or create the ChromaDB collection
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)

    def _remove_orphan_segments(self):
        """Remove database segment folders that are not longer referenced in SQlite metadata."""
        db_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        if not os.path.exists(db_path):
            return

        try:
            con = sqlite3.connect(db_path)
            used = {row[0] for row in con.execute("SELECT id FROM segments")}
            con.close()
        except Exception as e:
            print(f"DEBUG: Segments cant be read: {e}")
            return

        for entry in os.listdir(self.persist_directory):
            folder = os.path.join(self.persist_directory, entry)
            if os.path.isdir(folder) and entry not in used:
                _rmtree_retry(folder)


    def _save_original(self, file_name: str, file_bytes: bytes):
        """Stores the original uploaded file on disk for later retrieval or download.

        Args:
            file_name (str): The name of the file.
            file_bytes (bytes): The raw binary content of the file.

        Returns:
            str: The saved file path.
        """
        os.makedirs(DOCS_DIR, exist_ok=True)
        path = os.path.join(DOCS_DIR, _safe_filename(file_name))

        with open(path, "wb") as f:
            f.write(file_bytes)
        return path


    def list_documents(self):
        """Returns a list of unique stored documents grouped from chunk metadata.

        Returns:
            list: A list of document metadata dictionaries.
        """
        data = self.collection.get(include=["metadatas"])

        docs = {}
        for meta in data.get("metadatas") or []:
            name = meta.get("source", "Unknown")
            docs.setdefault(name, {
                "name": name,
                "tag": meta.get("tag", "default"),
                "uploaded_at": meta.get("uploaded_at", ""),
                "original_path": meta.get("original_path", "")
            })

        return list(docs.values())


    def delete_document(self, file_name: str):
        """Removes all vector chunks, extracted images, and original files associated with a document.

        Args:
            file_name (str): The name of the document to delete.
        """
        existing = self.collection.get(where={"source": file_name}, include=["metadatas"])
        metas = existing.get("metadatas") or []

        # Delete extracted image folder
        img_root = os.path.abspath(IMAGES_DIR)
        img_dirs = set()

        for meta in metas:
            img_p = meta.get("image_path", "")
            if img_p:
                img_dirs.add(os.path.abspath(os.path.dirname(img_p)))

        for img_dir in img_dirs:
            if img_dir.startswith(img_root + os.sep) and os.path.isdir(img_dir):
                shutil.rmtree(img_dir, ignore_errors=True)

        # Delete orignal file
        for original in {m.get("original_path", "") for m in metas}:
            if original and os.path.exists(original):
                try:
                    os.remove(original)
                except Exception:
                    pass

        if metas:
            self.collection.delete(where={"source": file_name})


    def add_document_with_chunking(self, doc, file_name: str, image_infos: list, file_bytes: bytes, custom_tag: str = ""):
        """Processes and indexes a document (PDF, DOCX, MD) with hybrid chunking and visual descriptions.

        Args:
            doc: The parsed Docling document object.
            file_name (str): The name of the file.
            image_infos (list): List of extracted image description dictionaries.
            file_bytes (bytes): Raw binary content of the file.
            custom_tag (str): Optional category tag for the document.

        Returns:
            int: Total number of chunks successfully added.
        """
        self.delete_document(file_name)
        original_path = self._save_original(file_name, file_bytes)
        uploaded_at = datetime.now().isoformat(timespec="seconds")

        chunker = HybridChunker(tokenizer=chunk_tokenizer)
        chunk_iter = chunker.chunk(doc)

        documents = []
        metadata = []
        ids = []

        # Process text chunks
        for idx, chunk in enumerate(chunk_iter):
            page_no = 1
            for item in chunk.meta.doc_items:
                if item.prov:
                    page_no = item.prov[0].page_no
                    break

            documents.append(chunk.text)
            ids.append(f"{file_name}_chunk_{idx}")
            metadata.append({
                "source": file_name,
                "chunk_index": idx,
                "page": page_no,
                "type": "text", 
                "tag": custom_tag if custom_tag else "default",
                "image_path": "",
                "original_path": original_path,
                "uploaded_at": uploaded_at
            })

        # Process image analysis descriptions
        for img_idx, info in enumerate(image_infos):
            if not info["description"]:
                continue

            caption = info.get("caption", "")
            documents.append(f"Figure / diagram on page {info['page']}:\nCaption: {caption}\nAutomatic image analysis: {info['description']}")
            ids.append(f"{file_name}_image_{img_idx}")
            metadata.append({
                "source": file_name,
                "chunk_index": img_idx,
                "page": info["page"],
                "type": "image", 
                "tag": custom_tag if custom_tag else "default",
                "image_path": info["path"],
                "original_path": original_path,
                "uploaded_at": uploaded_at
            })

        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadata,
                ids=ids
            )

        return len(documents)

    def add_code_with_chunking(self, code: str, file_name: str, file_bytes: bytes, custom_tag: str = ""):
        """Splits and indexes a Python script file into code symbol chunks.

        Args:
            code (str): The Python source code string.
            file_name (str): The name of the file.
            file_bytes (bytes): Raw binary content of the file.
            custom_tag (str): Optional category tag for the script.

        Returns:
            int: Total number of code chunks successfully added.
        """
        self.delete_document(file_name)
        original_path = self._save_original(file_name, file_bytes)
        uploaded_at = datetime.now().isoformat(timespec="seconds")

        documents = []
        metadata = []
        ids = []

        for idx, (label, text, start_line) in enumerate(_chunk_python_code(code)):
            documents.append(f"# File: {file_name} | {label}\n{text}")
            ids.append(f"{file_name}_code_{idx}")
            metadata.append({
                "source": file_name,
                "chunk_index": idx,
                "symbol": label,
                "start_line": start_line,
                "type": "code",
                "tag": custom_tag if custom_tag else "default",
                "image_path": "",
                "original_path": original_path,
                "uploaded_at": uploaded_at
            })

        if documents:
            self.collection.upsert(documents=documents, metadatas=metadata, ids=ids)

        return len(documents)

    def search_similar(self, query: str, n_results: int = 3):
        """Searches the vector store for chunks most similar to the query string.

        Args:
            query (str): The search query string.
            n_results (int): Maximum number of results to retrieve.

        Returns:
            list: A list of dictionaries containing retrieved chunk content, metadata, and distance.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)

        retrieved_chunks = []

        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [None] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                retrieved_chunks.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return retrieved_chunks


    def clear_database(self):
        """Clears the entire ChromaDB collection, extracted images, temporary files, and stored documents."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print("DEBUG: ChromaDB collection deleted!")
        except Exception as e:
            print(f"DEBUG: Collection cant be deleted: {e}")

        self.collection = None
        gc.collect()
        self._remove_orphan_segments()

        img_dir = "extracted_images"
        if os.path.exists(img_dir):
            try:
                shutil.rmtree(img_dir)
                os.makedirs(img_dir, exist_ok=True)
                print("DEBUG: Images successfully deleted!")
            except Exception as e:
                print(f"DEBUG: Image deletion error: {e}")

        garbage_dir = "garbage"
        if os.path.exists(garbage_dir):
            try:
                shutil.rmtree(garbage_dir)
                os.makedirs(garbage_dir, exist_ok=True)
                print("DEBUG: Garbage successfully deleted!")
            except Exception as e:
                print(f"DEBUG: Garbage deletion error: {e}")

        for temp_file in glob.glob("temp????????.*"):
            try:
                os.remove(temp_file)
            except Exception:
                pass

        if os.path.exists(DOCS_DIR):
            shutil.rmtree(DOCS_DIR, ignore_errors=True)

        self.collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=self.embedding_fn)
        print(f"DEBUG: Chunks after reset: {self.collection.count()}")