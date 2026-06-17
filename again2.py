import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
loader1=PyPDFLoader("data/resume.pdf")
loader2=PyPDFLoader("data/jd.pdf")
doc1=loader1.load()
doc2=loader2.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks1=splitter.split_documents(doc1)
chunks2=splitter.split_documents(doc2)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vDB1=Chroma.from_documents(documents=chunks1,embedding=embeddings)
vDB2=Chroma.from_documents(documents=chunks2,embedding=embeddings)
userQuery=input("Enter your prompt: ")
retriever1=vDB1.as_retriever(search_kwargs={"k":5})
retriever2=vDB2.as_retriever(search_kwargs={"k":5})
rc1= retriever1.invoke(userQuery)
rc2= retriever2.invoke(userQuery)
context1="\n".join(doc.page_content for doc in rc1)
context2="\n".join(doc.page_content for doc in rc2)
prompt = f"""{context1}

{context2}

{userQuery}

You are an AI assistant who compares resumes and job descriptions. You are supposed to give detailed answers for the user's queries. If your data is not updated, ignore the dates. Today is 17 June 2026"""
client = genai.Client(api_key=api_key)
response = client.models.generate_content(model= "gemini-2.5-flash" ,contents=prompt)
print(response.text)