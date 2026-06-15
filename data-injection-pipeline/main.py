import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter # for splitting the documents into chunks
from langchain_openai import OpenAIEmbeddings  # to create embeddings for the documents
from langchain_chroma import Chroma 
from dotenv import load_dotenv


def load_docs(docs_path:str="./data") -> list[Document]:

   if not os.path.exists(docs_path):
      raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")
   
   docs = []

   for file in Path(docs_path).rglob("*.txt"):
      text = file.read_text(encoding="utf-8")

      docs.append(
         Document(
               page_content=text,
               metadata={"source": str(file)}
         )
      )

   return docs

def main():
   
   # 1. loading the files 
   docs = load_docs("./data")
    


if __name__ == "__main__":
    main()
