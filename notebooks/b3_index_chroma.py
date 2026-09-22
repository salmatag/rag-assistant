from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def decouper(texte, taille=30, chevauchement=8):
    mots = texte.split()
    morceaux, debut = [], 0
    while debut < len(mots):
        morceaux.append(" ".join(mots[debut:debut + taille]))
        debut += taille - chevauchement
    return morceaux

model = SentenceTransformer(MODEL_NAME)

chunks, metadatas, ids = [], [], []
for fichier in sorted(Path("data/documents").glob("*.txt")):
    for j, morceau in enumerate(decouper(fichier.read_text(encoding="utf-8"))):
        chunks.append(morceau)
        metadatas.append({"source": fichier.name})
        ids.append(f"{fichier.stem}-{j}")

vecteurs = model.encode(chunks, show_progress_bar=True)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("documents")
collection.add(documents=chunks, embeddings=vecteurs.tolist(),
               metadatas=metadatas, ids=ids)

print("Index cree :", collection.count(), "chunks")