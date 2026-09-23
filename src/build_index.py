from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def decouper(texte, taille=60, chevauchement=15):
    mots = texte.split()
    morceaux, debut = [], 0
    while debut < len(mots):
        morceaux.append(" ".join(mots[debut:debut + taille]))
        debut += taille - chevauchement
    return morceaux


def construire_index():
    model = SentenceTransformer(MODEL_NAME)
    chunks, metadatas, ids = [], [], []
    for fichier in sorted(Path("data/documents").glob("*.txt")):
        for j, morceau in enumerate(decouper(fichier.read_text(encoding="utf-8"))):
            chunks.append(morceau)
            metadatas.append({"source": fichier.name})
            ids.append(f"{fichier.stem}-{j}")
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_or_create_collection("documents")
    if collection.count() == 0:
        collection.add(documents=chunks, embeddings=model.encode(chunks).tolist(),
                       metadatas=metadatas, ids=ids)
    print("Index :", collection.count(), "chunks")


if __name__ == "__main__":
    construire_index()