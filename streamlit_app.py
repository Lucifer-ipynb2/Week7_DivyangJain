"""
streamlit_app.py
-----------------
Simple web UI for the RAG system.

Run with:
    streamlit run streamlit_app.py
"""

import os
import streamlit as st

import config
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Question Answering (RAG)")
st.caption("Ask questions grounded in your own PDFs/text files, powered by OpenRouter.")


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


pipeline = get_pipeline()

with st.sidebar:
    st.header("⚙️ Index")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(config.DOCS_FOLDER, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(config.DOCS_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to '{config.DOCS_FOLDER}/'.")

    if st.button("🔨 Build / Rebuild Index"):
        with st.spinner("Building index..."):
            pipeline.build_index(config.DOCS_FOLDER)
        st.success("Index built!")

    index_ready = os.path.exists(os.path.join(config.INDEX_DIR, "index.faiss"))
    if index_ready and pipeline.store.index is None:
        try:
            pipeline.load_index()
        except Exception:
            pass

    st.write(f"Model: `{config.OPENROUTER_MODEL}`")
    st.write(f"Top-K retrieved chunks: {config.TOP_K}")

st.divider()

question = st.text_input("Ask a question about your documents:")

if st.button("Get Answer") and question:
    if pipeline.store.index is None:
        st.warning("Please build the index first (see sidebar).")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = pipeline.answer(question)
            except Exception as e:
                st.error(str(e))
                result = None

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            if result["sources"]:
                st.subheader("Sources")
                for s in result["sources"]:
                    st.write(f"- **{s['source']}** (chunk {s['chunk_id']}, similarity {s['score']})")
