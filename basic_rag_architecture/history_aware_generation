from __future__ import annotations

import os
from collections import deque
from typing import Deque

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)


class ConversationalRAG:
    """
    Production-friendly conversational RAG wrapper.

    Responsibilities:
    - Load vector database
    - Manage retrieval
    - Rewrite contextual questions
    - Maintain chat history
    - Generate grounded answers
    """

    def __init__(
        self,
        persist_directory: str,
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        history_size: int = 10,
        top_k: int = 5,
    ) -> None:

        self.chat_history: Deque = deque(maxlen=history_size)

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
        )

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
        )

        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 20,
            },
        )

        self.chain = self._build_chain()

    def _build_chain(self):
        """
        Creates:
        User Question
            ↓
        History-aware retriever
            ↓
        Chroma
            ↓
        Context
            ↓
        GPT-4o
        """

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Given the conversation history and the latest user question,
                    rewrite the question so it is fully standalone.

                    Return only the rewritten question.
                    """,
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=contextualize_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant.

                    Use ONLY the provided context to answer.

                    If the answer cannot be found in the context,
                    reply with:

                    "I don't know based on the provided documents."

                    Context:
                    {context}
                    """,
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
        )

        return create_retrieval_chain(
            history_aware_retriever,
            document_chain,
        )

    def ask(self, question: str) -> str:
        """
        Ask a question and receive a grounded answer.
        """

        response = self.chain.invoke(
            {
                "input": question,
                "chat_history": list(self.chat_history),
            }
        )

        answer = response["answer"]

        self.chat_history.append(
            HumanMessage(content=question)
        )

        self.chat_history.append(
            AIMessage(content=answer)
        )

        return answer

    def clear_history(self) -> None:
        self.chat_history.clear()


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment."
        )

    rag = ConversationalRAG(
        persist_directory="db/chroma_db",
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        history_size=10,
        top_k=5,
    )

    print("Conversational RAG Started")
    print("Type 'quit' to exit")
    print("Type 'clear' to reset chat history")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("\nGoodbye!")
            break

        if question.lower() == "clear":
            rag.clear_history()
            print("\nChat history cleared.")
            continue

        try:
            answer = rag.ask(question)
            print(f"\nAssistant: {answer}")

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()