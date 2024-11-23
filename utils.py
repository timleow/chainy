from typing import List
from pathlib import Path

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

PDF_STORAGE_PATH = "./pdfs"
chainlit_md_path  = Path(__file__).parent / "chainlit.md"
WELCOME_MESSAGE = chainlit_md_path.read_text()

def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for pdf_path in pdf_directory.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    doc_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search