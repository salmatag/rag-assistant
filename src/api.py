from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from src.generation import generer_reponse

app = FastAPI(title="RAG Assistant")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("documents")


class Question(BaseModel):
    question: str
    backend: str = "groq"
    k: int = 3


class Reponse(BaseModel):
    reponse: str
    sources: list[str]


def rechercher(question, k):
    q = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


@app.get("/")
def racine():
    return {"message": "RAG Assistant API"}


@app.post("/ask", response_model=Reponse)
def ask(requete: Question):
    passages = rechercher(requete.question, requete.k)
    reponse = generer_reponse(requete.question, passages, backend=requete.backend)
    sources = sorted({src for _, src in passages})
    return Reponse(reponse=reponse, sources=sources)