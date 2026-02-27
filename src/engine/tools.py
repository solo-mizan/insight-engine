import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def ingest_pdfs(directory_path: str):
    """
    We use RecursiveCharacterTextSplitter because it preserves paragraph structure better than simple character splits.
    """
    # 1. load pdf
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()

    # 2. split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    splits = text_splitter.split_documents(docs)

    # 3. create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
        persist_directory="./chroma_db"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})