import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
import chromadb
from src.generation import generer_reponse

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("documents")


def rechercher(question, k=3):
    q = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def rag(question, backend="groq", k=3):
    passages = [(doc, meta["source"]) for doc, meta in rechercher(question, k)]
    reponse = generer_reponse(question, passages, backend=backend)
    print("REPONSE:\n", reponse)
    print("\nSOURCES REELLES (issues du retrieval):")
    for _, source in passages:
        print("  -", source)


if __name__ == "__main__":
    question = "Quelle est la capital du japon ?"
    print("=== BACKEND GROQ (cloud) ===")
    rag(question, backend="groq")
    print("\n=== BACKEND OLLAMA (local) ===")
    rag(question, backend="ollama")