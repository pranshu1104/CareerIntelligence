import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
api_key= os.getenv("{GEMINI_API_KEY}")
doc11= PyPDFLoader("data/resume.pdf")
doc22=PyPDFLoader("data/jd.pdf")
doc1=doc11.load()
doc2=doc22.load()
splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk1=splitter.split_documents(doc1);
chunk2=splitter.split_documents(doc2);
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorDb1=Chroma.from_documents(documents=chunk1,embedding=embeddings)
vectorDb2=Chroma.from_documents(documents=chunk2,embedding=embeddings)
userQuery=input("Ask : ")
retriever1 = vectorDb1.as_retriever(search_kwargs={"k":5})
retriever2= vectorDb2.as_retriever(search_kwargs={"k":5})
re_chunk = retriever1.invoke(userQuery)
re_chunk1= retriever2.invoke(userQuery)
context1= "/n".join(doc.page_content for doc in re_chunk)
context2 = "/n".join(doc.page_content for doc in re_chunk1)
prompt = f"""{context1}  

{context2}

{userQuery}

You are an AI assistant who compares resumes and job descriptions, and give detailed analysis.
"""
client = genai.Client(api_key=api_key)
response=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
print(response.text)