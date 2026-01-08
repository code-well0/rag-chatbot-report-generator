import asyncio
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------- EVENT LOOP FIX ----------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------------- ENV ----------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ---------------- CONSTANTS ----------------
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- STREAMLIT ----------------
st.set_page_config(page_title="RAG Chatbot + Report Generator", page_icon="🤖")
st.title("📄 RAG Chatbot & Report Generator")

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.
<context>
{context}
</context>

Question: {input}
""")

# ---------------- VECTOR DB ----------------
def create_vector_db(uploaded_files=None):
    if "vectors" in st.session_state:
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    docs = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
    else:
        loader = PyPDFDirectoryLoader("papers")
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)

    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
    st.session_state.docs = final_docs

    st.success("✅ Vector database ready!")

# ---------------- PDF UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs (optional, overrides default papers folder)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    create_vector_db(uploaded_files)
elif st.button("Create Vector Store from Default PDFs"):
    create_vector_db()

# ---------------- CHAT ----------------
query = st.text_input("Ask a question from your documents:")

if query and "vectors" in st.session_state:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    start = time.process_time()
    response = rag_chain.invoke({"input": query})

    st.write("⏱ Response Time:", round(time.process_time() - start, 2), "seconds")
    st.subheader("🤖 Answer")
    st.write(response["answer"])

    with st.expander("📄 Retrieved Context"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("-----")

# ---------------- REPORT GENERATOR ----------------
if st.button("Generate Summary Report"):
    if "docs" not in st.session_state:
        st.warning("Create the vector store first!")
    else:
        report_prompt = ChatPromptTemplate.from_template("""
Create a structured summary report from the following documents.
Include headings, key insights, and conclusions.

<context>
{context}
</context>
""")

        report_chain = create_stuff_documents_chain(llm, report_prompt)

        # 🔥 invoke returns STRING
        report_text = report_chain.invoke({
            "context": st.session_state.docs
        })

        st.subheader("📑 Summary Report")
        st.write(report_text)

        st.download_button(
            label="Download Report as TXT",
            data=report_text,
            file_name="document_report.txt",
            mime="text/plain"
        )
