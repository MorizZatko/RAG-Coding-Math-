"""Utility functions for data formatting and LaTeX conversion.

This module provides helper functions to format ISO timestamps into
readable strings and to parse and convert LaTeX expressions into
Streamlit-compatible Markdown formats while preserving code blocks.
"""

import re
from datetime import datetime


def format_date(iso: str):
    """Formats an ISO timestamp string into a readable date & time string.
    
    Args:
        iso (str): The ISO timestamp in string format.

    Retuns:
        str: The formatted date string in 'dd.mm.yyyy HH:MM' format.
    """
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return "-"


def _convert_math(text: str):
    """Converts standard LaTeX into Streamlit-compatible Markdown math formats.
    
    Args:
        text (str): The raw text string containing LaTeX expressions.

    Returns:
        str: The processed text string with updated math format.
    """
    # Convert block-level LaTeX brackets to double-dollar to display math syntax
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$${m.group(1).strip()}$$", text, flags=re.DOTALL)

    # Convert inline LaTeX parantheses to single-dollar inline math syntax
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.DOTALL)

    # Convert bare brackets on their own line containing LaTeX commands to display math
    text = re.sub(r"(?m)^[ \t]*\[\s*([^\n]*\\[a-zA-Z]+[^\n]*?)\s*\][ \t]*$", lambda m: f"$${m.group(1).strip()}$$", text)

    # Convert bare parentheses starting with LaTeX commands to inline math
    text = re.sub(r"\(\s*(\\[a-zA-Z]+[^()\n]*)\)", lambda m: f"${m.group(1).strip()}$", text)
    return text


def fix_latex(text: str):
    """Process text so that LaTeX expressions from the model render correctly in Streamlit Markdown.
    
    Args:
        text (str): The input text containing potential LaTeX and code blocks.

    Returns:
        str: The processed text with converted math formulas outside of code blocks.
    """
    # Split text to isolate code blocks from regular prose and math
    parts = re.split(r"(```.*?```|`[^`\n]+`)", text, flags=re.DOTALL)

    # Process only non-code text segments for math conversions
    for i in range(0, len(parts), 2):
        parts[i] = _convert_math(parts[i])
    return "".join(parts)