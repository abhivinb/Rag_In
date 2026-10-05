"""Small local UI for manually exercising the RAG HTTP endpoint."""

import os
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Enterprise RAG", layout="centered")
st.title("Enterprise RAG Knowledge Assistant")
st.caption("Ask a question against the documents.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
api_key = os.getenv("API_KEY", "")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

uploaded_file = st.file_uploader("Index a document", type=["pdf", "docx", "txt"])

if st.button("Upload and index", disabled=uploaded_file is None):
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        with st.spinner("Indexing document..."):
            response = httpx.post(
                f"{api_url}/documents",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                headers=headers,
                timeout=180.0,
            )
        if response.is_success:
            payload = response.json()
            st.success(
                f"Indexed {payload['file_name']} with {payload['chunk_count']} chunks."
            )
        else:
            st.error(f"API error {response.status_code}: {response.text}")
    except httpx.HTTPError as error:
        st.error(f"Could not reach the API: {error}")

query = st.chat_input("Ask a question about your indexed documents")

if query:
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        with st.spinner("Searching the knowledge base..."):
            response = httpx.post(
                f"{api_url}/chat",
                json={
                    "query": query,
                    "conversation_id": st.session_state.conversation_id,
                    "messages": st.session_state.messages,
                },
                headers=headers,
                timeout=120.0,
            )
        if response.is_success:
            payload = response.json()
            st.session_state.conversation_id = payload["conversation_id"]
            st.session_state.messages = payload["messages"]
            st.session_state.last_sources = payload.get("sources", [])
            st.rerun()
        else:
            st.error(f"API error {response.status_code}: {response.text}")
    except httpx.HTTPError as error:
        st.error(f"Could not reach the API: {error}")

if st.session_state.last_sources:
    with st.expander("Latest answer sources"):
        for source in st.session_state.last_sources:
            st.write(
                f"{source['document_id']} / {source['chunk_id']} "
                f"(score: {source['retrieval_score']:.3f})"
            )
