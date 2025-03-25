import os
import openai
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Updated imports to match the latest library structure
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import EMBEDDING_CONFIG, LLM_CONFIG, VECTOR_STORE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ubuntu Q&A Chatbot", version="1.0")


class QueryRequest(BaseModel):
    question: str

qa_chain = None
vectorstore = None


@app.on_event("startup")
def startup_event():
    """
    Initialize the vector store, reranker, and QA chain at application startup.
    """
    global qa_chain, vectorstore

    openai_api_key = EMBEDDING_CONFIG["openai_api_key"]
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in config.")

    logger.info("Setting OPENAI_API_KEY and initializing QA chain...")

    openai.api_key = openai_api_key

    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings(
        api_key=openai_api_key,
        model=EMBEDDING_CONFIG["model"]
    )

    # Load the Chroma store with the embedding function
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=VECTOR_STORE_PATH
    )

    # Create the base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Set up the reranker using a Hugging Face cross-encoder model
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # Wrap retriever with contextual compression and reranker
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=LLM_CONFIG["temperature"],
        api_key=LLM_CONFIG["openai_api_key"],
        model_name=LLM_CONFIG["model_name"]
    )

    # Custom prompt for chain
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

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Build ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": STUFF_PROMPT},
        return_source_documents=True,
        output_key="answer"
    )

    logger.info("QA chain with reranker and memory initialized successfully.")


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
        # Memory-based chat with contextual retrieval
        result = qa_chain({"question": query_request.question})
        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        sources = [
            {
                "source": doc.metadata.get("source", "information not found in the data source"),
                "content": doc.page_content
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
