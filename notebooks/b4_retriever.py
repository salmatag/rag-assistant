from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("documents")

def rechercher(question, k=3):
    q = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q, n_results=k)
    return zip(res["documents"][0], res["metadatas"][0], res["distances"][0])

question = " Que risque-t-on avec l'aspirine sur le long terme ?"
for doc, meta, distance in rechercher(question):
    print(f"[{meta['source']}]  distance={distance:.3f}")
    print("   ", doc)
    print()
