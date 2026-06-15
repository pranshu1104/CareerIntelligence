""""import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini Client
if not api_key:
    st.error("Error: GEMINI_API_KEY missing from .env file!")
    st.stop()

client = genai.Client(api_key=api_key)

# Configure the Streamlit UI Dashboard
st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")
st.title("🤖 GenAI Portfolio Assistant")
st.write("Ask a research question to generate a structured synthesis report.")

# User Input Field
user_query = st.text_input("Enter your research topic:", placeholder="e.g., Explain Quantum Computing in 3 paragraphs")

# Button to Trigger the Generation
if st.button("Generate Report", type="primary"):
    if user_query:
        with st.spinner("Analyzing and drafting report..."):
            try:
                # API Call to Gemini
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Provide a structured professional breakdown regarding: {user_query}"
                )

                # Render the Output Nicely
                st.subheader("Results:")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"API Error: {str(e)}")
    else:
        st.warning("Please enter a topic first!")

import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini Client
if not api_key:
    st.error("Error: GEMINI_API_KEY missing from .env file!")
    st.stop()

client = genai.Client(api_key=api_key)

# Configure the Streamlit UI Dashboard
st.set_page_config(page_title="RAG Document Assistant", page_icon="📑")
st.title("📑 AI Smart Document Assistant (RAG)")
st.write("Upload a business document or text file, then ask questions about its specific contents.")

# 1. UI Element: File Uploader Component
uploaded_file = st.file_uploader("Upload your context document (.txt files)", type=["txt"])

# Global variable to store document content
document_context = ""

if uploaded_file is not None:
    # Read the text content of the uploaded file directly in memory
    document_context = uploaded_file.read().decode("utf-8")
    st.success("✅ Document processed successfully! Ready for queries.")

    # Optional: Visual Anchor showing a preview of the file text to the user
    with st.expander("👁️ View Extracted Document Text"):
        st.code(document_context)

# 2. UI Element: User Query Input Field
user_query = st.text_input("What do you want to find or synthesize from this document?",
                           placeholder="e.g., What is the protocol for password security?")

# 3. Execution Phase: Button to Trigger Generation
if st.button("Query Document", type="primary"):
    if user_query:
        if not document_context:
            st.warning("Please upload a context document first so the AI has reference data!")
        else:
            with st.spinner("Searching document layout and generating answer..."):
                try:
                    # Construct a secure RAG structured prompt template
                    rag_prompt = f

                    You are a professional enterprise research analyst. Answer the user's question using ONLY the provided reference text.
                    If the answer cannot be found in the text, politely state that the information is missing from the document.
                    
                    REFERENCE TEXT:
                    \"\"\"
                    {document_context}
                    \"\"\"
                    
                    USER QUESTION: {user_query}
                    
                    ANSWER:
                    "

                    # API Call to Gemini
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=rag_prompt
                    )

                    # Render the Output Nicely
                    st.subheader("Analysis Results:")
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"API Error: {str(e)}")
    else:
        st.warning("Please type a question regarding your document!")
"""
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# CONFIG
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=api_key)

# =========================
# PAGE
# =========================

st.set_page_config(page_title="Career RAG Assistant", page_icon="🚀")

st.title("🚀 Career RAG Assistant")
st.write("Upload your Resume and a Job Description, then ask questions using RAG.")

# =========================
# FILE UPLOADS
# =========================

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])

# =========================
# BUILD VECTOR DATABASES (SEPARATE)
# =========================

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

# =========================
# CREATE VECTOR STORES ONCE
# =========================

if resume_file and jd_file:
    if "resume_db" not in st.session_state or "jd_db" not in st.session_state:
        with st.spinner("Creating embeddings and vector databases..."):
            st.session_state.resume_db = build_vector_db_for_file(resume_file, "resume")
            st.session_state.jd_db = build_vector_db_for_file(jd_file, "job_description")
        st.success("✅ Knowledge Base Created")

# =========================
# USER QUERY
# =========================

query = st.text_input(
    "Ask a question",
    placeholder="What skills am I missing for this role?"
)

# =========================
# RAG PIPELINE
# =========================

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