import asyncio
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
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

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DocChat AI",
    page_icon="public\chatbot-icon.svg",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("""
# DocChat AI: RAG-Powered Document Intelligence
Ask questions and generate reports from your documents using **Gemini + FAISS**
""")

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ---------------- PROMPTS ----------------
qa_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.
<context>
{context}
</context>

Question: {input}
""")

report_prompt = ChatPromptTemplate.from_template("""
Create a structured summary report from the following documents.
Include:
- Title
- Key Insights
- Important Findings
- Conclusion

<context>
{context}
</context>
""")

# ---------------- VECTOR DB ----------------
def create_vector_db(uploaded_files=None):
    if "vectors" in st.session_state:
        st.info("Vector database already exists.")
        return

    with st.spinner("Creating vector database..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        docs = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(path)
                docs.extend(loader.load())
        else:
            loader = PyPDFDirectoryLoader("papers")
            docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)

        st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
        st.session_state.docs = final_docs

    st.success("Vector database created successfully!")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("Document Setup")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base"):
        create_vector_db(uploaded_files)

    st.divider()

    if "vectors" in st.session_state:
        st.success("Vector Store Ready")
    else:
        st.warning("Vector Store Not Created")

# ================= MAIN TABS =================
tab1, tab2 = st.tabs(["Chatbot", "Report Generator"])

# ---------------- CHAT TAB ----------------
with tab1:
    st.subheader("Ask Questions from Documents")

    query = st.text_input("Enter your question")

    if query:
        if "vectors" not in st.session_state:
            st.warning("Please build the vector database first.")
        else:
            doc_chain = create_stuff_documents_chain(llm, qa_prompt)
            retriever = st.session_state.vectors.as_retriever()
            rag_chain = create_retrieval_chain(retriever, doc_chain)

            start = time.process_time()
            response = rag_chain.invoke({"input": query})

            st.markdown("### Answer")
            st.write(response["answer"])

            st.caption(f"⏱ Response Time: {round(time.process_time()-start, 2)} seconds")

            with st.expander("Retrieved Context"):
                for i, doc in enumerate(response["context"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.divider()

# ---------------- REPORT TAB ----------------
with tab2:
    st.subheader("Generate Structured Report")

    report_type = st.selectbox(
        "Select Report Type",
        ["Summary Report", "Technical Overview", "Research Insights"]
    )

    if st.button("Generate Report"):
        if "docs" not in st.session_state:
            st.warning("Please build the vector database first.")
        else:
            with st.spinner("Generating report..."):
                report_chain = create_stuff_documents_chain(llm, report_prompt)
                report_text = report_chain.invoke({
                    "context": st.session_state.docs
                })

            st.markdown("### Generated Report")
            st.write(report_text)

            st.download_button(
                "Download Report",
                data=report_text,
                file_name="rag_report.txt",
                mime="text/plain"
            )
