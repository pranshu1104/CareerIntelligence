import os
from google import genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key=os.getenv("{GEMINI_API_KEY}")
loader= PyPDFLoader("data/resume.pdf")
docs=loader.load()
loader1=PyPDFLoader("data/jd.pdf")
docs1=loader1.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk1=splitter.split_documents(docs)
chunk2=splitter.split_documents(docs1)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorDBres=Chroma.from_documents(documents=chunk1,embedding=embeddings)
vectorDBjd=Chroma.from_documents(documents=chunk2,embedding=embeddings)
userQuery=input("Enter your query: ") #from frontend
retriever=vectorDBres.as_retriever(search_kwargs={"k":5})
retriever2=vectorDBjd.as_retriever(search_kwargs={"k":5})
retrieved_chunks11 = retriever.invoke(userQuery)
retrieved_chunks22 = retriever2.invoke(userQuery)
retrieved_chunks1="/n".join(doc.page_content for doc in retrieved_chunks11)
retrieved_chunks2="/n".join(doc.page_content for doc in retrieved_chunks22)
client = genai.Client(api_key=api_key)
#prompt=PromptTemplate(
 #   userQuery+retrieved_chunks1+retrieved_chunks2+""
#)
prompt = f" {userQuery}{retrieved_chunks1}{retrieved_chunks2} You are a career assistant who compares resumes and job descriptions to identify skill gaps, provide feedback and assist job candidates for better succcess"
response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
print(response.text)