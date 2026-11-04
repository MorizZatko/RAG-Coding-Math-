@echo off
echo Starting RAG-System
call rag\Scripts\activate
streamlit run main.py
pause