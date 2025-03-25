# build_vector_store.py

import logging
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from config import (
    EMBEDDING_CONFIG, CHUNK_CONFIG,
    DATA_FOLDER, VECTOR_STORE_PATH
)
from logger_config import setup_logging

def build_vector_store():
    try:
        setup_logging()
        logger = logging.getLogger(__name__)

        logger.info(f"Loading markdown documents from {DATA_FOLDER}...")
        loader = DirectoryLoader(DATA_FOLDER, glob="**/*.md")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CONFIG["chunk_size"],
            chunk_overlap=CHUNK_CONFIG["chunk_overlap"]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        logger.info(f"Creating embeddings using model: {EMBEDDING_CONFIG['model']}")
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_CONFIG["model"],
            openai_api_key=EMBEDDING_CONFIG["openai_api_key"]
        )

        logger.info(f"Building Chroma vector store at {VECTOR_STORE_PATH}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        vectorstore.persist()
        logger.info("Vector store persisted successfully!")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error building vector store: {e}", exc_info=True)

if __name__ == "__main__":
    build_vector_store()
