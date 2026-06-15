import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter # for splitting the documents into chunks
from langchain_openai import OpenAIEmbeddings  # to create embeddings for the documents
from langchain_chroma import Chroma 
from dotenv import load_dotenv


def load_docs(docs_path:str="./docs") -> list[Document]:

   if not os.path.exists(docs_path):
      raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")
   
   docs = []

   for file in Path(docs_path).rglob("*.txt"):
      docs.append(
         Document(
               page_content=file.read_text(),
               metadata={"source": str(file)},
         )
      )

   return docs 

def create_chunks(docs:list[Document], chunk_size:int=512, chunk_overlap:int=100) -> list[Document]:
   
   text_splitter = CharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
   )

   chunks = text_splitter.split_documents(docs)
   
   return chunks

def create_embeddings_and_store(chunks:list[Document], persist_directory:str="db/vector_store") -> Chroma:

   embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

   # create chroma db vector store
   vector_store = Chroma.from_documents(
      documents=chunks,
      embedding=embedding_model,
      persist_directory=persist_directory,
      collection_metadata={"hnsw":"cosine"},
   )

   return vector_store

def main():
   
   # 1. loading the files 
   docs = load_docs("./docs")
   # print(docs)

   # 2. create chunks from the documents
   chunks = create_chunks(docs)

   # 3. create embeddings & store them in a vector database
   vector_store = create_embeddings_and_store(chunks)
   

if __name__ == "__main__":
    main()
