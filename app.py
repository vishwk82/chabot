import os
import openai
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic import PrivateAttr


from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain.schema.retriever import BaseRetriever  # ✅ Correct import path
from langchain.schema import Document
from typing import List

from config import EMBEDDING_CONFIG, LLM_CONFIG, VECTOR_STORE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ubuntu Q&A Chatbot", version="1.0")


class QueryRequest(BaseModel):
    question: str

qa_chain = None
vectorstore = None

# ✅ Custom FullDocRetriever
class FullDocRetriever(BaseRetriever):
    _retriever: any = PrivateAttr()

    def __init__(self, wrapped_retriever):
        super().__init__()
        self._retriever = wrapped_retriever  # ✅ Use PrivateAttr

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self._retriever.get_relevant_documents(query)
        seen = set()
        deduped = []
        for doc in docs:
            full_text = doc.metadata.get("full_doc", doc.page_content)
            if full_text not in seen:
                seen.add(full_text)
                doc.page_content = full_text  # Replace chunk with full doc
                deduped.append(doc)
        return deduped

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self._retriever.aget_relevant_documents(query)
        seen = set()
        deduped = []
        for doc in docs:
            full_text = doc.metadata.get("full_doc", doc.page_content)
            if full_text not in seen:
                seen.add(full_text)
                doc.page_content = full_text
                deduped.append(doc)
        return deduped


@app.on_event("startup")
def startup_event():
    """
    Initialize vector store, reranker, and QA chain.
    """
    global qa_chain, vectorstore

    openai_api_key = EMBEDDING_CONFIG["openai_api_key"]
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in config.")

    logger.info("Setting OPENAI_API_KEY and initializing QA chain...")

    openai.api_key = openai_api_key

    embedding_model = OpenAIEmbeddings(
        api_key=openai_api_key,
        model=EMBEDDING_CONFIG["model"]
    )

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=VECTOR_STORE_PATH
    )

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    full_doc_retriever = FullDocRetriever(retriever)

    llm = ChatOpenAI(
        temperature=LLM_CONFIG["temperature"],
        api_key=LLM_CONFIG["openai_api_key"],
        model_name=LLM_CONFIG["model_name"]
    )

    STUFF_PROMPT = PromptTemplate(
        template="""
You are an Ubuntu Q&A Chatbot. You will be given a user question and some context. If you find the answer in the context provide the answer from the context.
Context:
{context}

Question: {question}
Answer:
""",
        input_variables=["context", "question"],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=full_doc_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": STUFF_PROMPT},
        return_source_documents=True,
        output_key="answer"
    )

    logger.info("QA chain initialized successfully.")


@app.post("/chat")
def chat_endpoint(query_request: QueryRequest):
    """
    POST /chat
    Body: { "question": "How do I update packages in Ubuntu?" }
    """
    global qa_chain

    if qa_chain is None:
        raise HTTPException(
            status_code=500,
            detail="QA chain not initialized."
        )
    if not query_request.question:
        raise HTTPException(
            status_code=400,
            detail="Invalid question."
        )

    try:
        result = qa_chain({"question": query_request.question})
        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        sources = [
            {
                "source": doc.metadata.get("source", "source unknown"),
            }
            for doc in source_documents
        ]

        if answer == "The answer is not there in the database.":
            sources = []

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
