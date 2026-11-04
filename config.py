"""Configuration constants module for vector distance thresholds.

This module defines global threshold constants used by the chat view
to determine similarity matches for text chunks and visual images.
(lower distance = more similarity).
"""

# Maximum distance threshold for text chunk vector similarity matching
MAX_DIST = 0.64

# Maximum distance threshold for image vector similarity matching
IMAGE_MAX_DIST = 0.5

# Number of chunks retrieved for the first question in the chat
N_RESULTS = 5

# Fewer chunks for follow-up questions to ensure the chat history fits into the context window
N_RESULTS_WITH_HISTORY = 4

# Maximum of previous messages sent to the model as chat history
HISTORY_MAX_MESSAGES = 6

# Share of the context window in tokens
HISTORY_TOKEN_SHARE = 0.2

# Maximum length of a single history message in characters
HISTORY_MESSAGE_MAX_CHAR = 1200