import chunk
import time
import asyncio
from pathlib import Path
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_ENV,
)

existing_index = [ index["name"] for index in pc.list_indexes() ]

if PINECONE_INDEX_NAME not in existing_index:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="dotproduct",
        spec=spec,
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


async def load_vectorstore(uploaded_files, role: str, doc_id: str):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = text_splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadata = [
            {
                "source_id": file.filename,
                "doc_id": doc_id,
                "role": role,
                "page": chunk.metadata.get("page", 0)
            }
            for i, chunk in enumerate(chunks)
        ]

        print(f"Embedding {len(texts)} Chunk...")

        embeddings = await asyncio.to_thread(embed_model.embed_documents, texts)

        print("Uploading to Pinecone...")

        with tqdm(total=len(embeddings), desc="Upserting To Pinecone") as progress:
            index.upsert(
                vectors=zip(ids, embeddings, metadata),
            )
            progress.update(len(embeddings))
        
        print(f"Uploaded {len(embeddings)} Vectors for {file.filename} To Pinecone")





