import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("documents")

jeu_test = [
    {"question": "Quel medicament est utilise pour le diabete de type 2 ?", "source": "metformine.txt"},
    {"question": "Quels sont les effets indesirables de la metformine ?", "source": "metformine.txt"},
    {"question": "Que risque-t-on avec l'aspirine sur le long terme ?", "source": "aspirine.txt"},
    {"question": "L'aspirine est-elle utilisee apres un infarctus ?", "source": "aspirine.txt"},
    {"question": "Qu'est-ce que l'hypertension arterielle ?", "source": "hypertension.txt"},
    {"question": "Comment se mesure la tension ?", "source": "hypertension.txt"},
    {"question": "Le diabete de type 2 est-il lie au surpoids ?", "source": "diabete.txt"},
    {"question": "Le traitement du diabete associe-t-il regime et activite physique ?", "source": "diabete.txt"},
]


def analyse(question, source, max_k=5):
    q = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q, n_results=max_k)
    sources = [m["source"] for m in res["metadatas"][0]]
    rang = sources.index(source) + 1 if source in sources else 0
    return sources, rang


for ex in jeu_test:
    sources, rang = analyse(ex["question"], ex["source"])
    statut = f"rang {rang}" if rang else "ABSENT"
    print(f"[{statut:>6}]  attendu={ex['source']}")
    print(f"          Q: {ex['question']}")
    print(f"          top5: {sources}")
    print()