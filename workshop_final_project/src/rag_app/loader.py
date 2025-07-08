from pathlib import Path
from typing import Iterable

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

TMP_UPLOADS_PATH = Path("tmp_uploads")
TMP_UPLOADS_PATH.mkdir(exist_ok=True)


class DocumentLoader:
    """Carica file PDF e TXT in Document objects."""

    def load_uploaded(self, files: Iterable) -> list[Document]:
        documents = []
        for file in files:
            if file.name.endswith(".txt"):
                content = file.getvalue().decode("utf-8")
                documents.append(
                    Document(page_content=content, metadata={"file_name": file.name})
                )
            elif file.name.endswith(".pdf"):
                tmp_path = TMP_UPLOADS_PATH / file.name
                with open(tmp_path, "wb") as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(str(tmp_path))
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(pages)
                for doc in docs:
                    doc.metadata["file_name"] = file.name
                    documents.append(doc)
        return documents
