import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

TOP_K = 3

with open("eval_questions.json", "r") as f:
    eval_data = json.load(f)

PDF_DIR = "uploaded_pdfs"

if not os.path.exists(PDF_DIR) or len(os.listdir(PDF_DIR)) == 0:
    raise RuntimeError("No PDFs found in uploaded_pdfs/. Upload PDFs via the app first.")

loader = PyPDFDirectoryLoader(PDF_DIR)
docs = loader.load()

if len(docs) == 0:
    raise RuntimeError("PDFs found but no readable text (scanned PDFs not supported).")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_docs = splitter.split_documents(docs)

if len(final_docs) == 0:
    raise RuntimeError("Text splitting failed.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = FAISS.from_documents(final_docs, embeddings)

correct = 0

for item in eval_data:
    question = item["question"]
    expected_doc = item["expected_doc"]

    retrieved_docs = vector_store.similarity_search(question, k=TOP_K)

    retrieved_sources = [
        os.path.basename(doc.metadata.get("source", ""))
        for doc in retrieved_docs
    ]

    if expected_doc in retrieved_sources:
        correct += 1

    print(f"Q: {question}")
    print("Retrieved:", retrieved_sources)
    print("Expected:", expected_doc)
    print("-" * 50)

accuracy = correct / len(eval_data)
print(f"\n Top-{TOP_K} Retrieval Accuracy: {accuracy:.2%}")
