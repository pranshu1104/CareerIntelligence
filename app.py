
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# CONFIG


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=api_key)

# PAGE

st.set_page_config(page_title="Career RAG Assistant", page_icon="🚀")

st.title("🚀 Career RAG Assistant")
st.write("Upload your Resume and a Job Description, then ask questions using RAG.")


# FILE UPLOADS


resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])


# BUILD VECTOR DATABASES (SEPARATE)


def build_vector_db_for_file(uploaded_file, source_label):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # Tag every chunk with its source
    for doc in docs:
        doc.metadata["source"] = source_label

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectordb

# CREATE VECTOR STORES ONCE


if resume_file and jd_file:
    if "resume_db" not in st.session_state or "jd_db" not in st.session_state:
        with st.spinner("Creating embeddings and vector databases..."):
            st.session_state.resume_db = build_vector_db_for_file(resume_file, "resume")
            st.session_state.jd_db = build_vector_db_for_file(jd_file, "job_description")
        st.success("✅ Knowledge Base Created")


# USER QUERY


query = st.text_input(
    "Ask a question",
    placeholder="What skills am I missing for this role?"
)

# RAG PIPELINE

if st.button("Analyze"):

    if "resume_db" not in st.session_state or "jd_db" not in st.session_state:
        st.warning("Please upload both Resume and Job Description first.")

    elif not query:
        st.warning("Please enter a question.")

    else:
        with st.spinner("Retrieving relevant information..."):

            # Always pull from BOTH stores independently
            resume_docs = st.session_state.resume_db.similarity_search(query, k=4)
            jd_docs = st.session_state.jd_db.similarity_search(query, k=4)

            resume_context = "\n\n".join(doc.page_content for doc in resume_docs)
            jd_context = "\n\n".join(doc.page_content for doc in jd_docs)

            prompt = f"""
You are an expert AI Career Coach helping a candidate evaluate their fit for a job.

Use ONLY the retrieved context below. Do not make up information.

========================
CANDIDATE'S RESUME
========================

{resume_context}

========================
JOB DESCRIPTION
========================

{jd_context}

========================
QUESTION
========================

{query}

Provide a structured response with:

1. **Direct Answer** - Answer the question directly based on both documents.
2. **Matching Skills** - Skills/experience from the resume that match the JD requirements.
3. **Missing Skills** - Requirements from the JD that are NOT found in the resume.
4. **Resume Improvements** - Specific suggestions to better tailor the resume for this role.
5. **Interview Preparation Advice** - Key topics to prepare for based on the JD.
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("📊 Analysis Results")
            st.markdown(response.text)

            # if errror, run pip install sentence-transformers