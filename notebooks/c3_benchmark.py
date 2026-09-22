import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

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


def decouper(texte, taille, chevauchement):
    mots = texte.split()
    morceaux, debut = [], 0
    while debut < len(mots):
        morceaux.append(" ".join(mots[debut:debut + taille]))
        debut += taille - chevauchement
    return morceaux


def construire_index(taille, chevauchement):
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"docs_{taille}_{chevauchement}")
    chunks, metas, ids = [], [], []
    for f in sorted(Path("data/documents").glob("*.txt")):
        for j, m in enumerate(decouper(f.read_text(encoding="utf-8"), taille, chevauchement)):
            chunks.append(m)
            metas.append({"source": f.name})
            ids.append(f"{f.stem}-{j}")
    col.add(documents=chunks, embeddings=model.encode(chunks).tolist(),
            metadatas=metas, ids=ids)
    return col


def recall(col, k):
    bons = 0
    for ex in jeu_test:
        q = model.encode([ex["question"]]).tolist()
        res = col.query(query_embeddings=q, n_results=k)
        sources = [m["source"] for m in res["metadatas"][0]]
        bons += ex["source"] in sources
    return bons / len(jeu_test)


print(f"{'taille':>7} | {'chunks':>6} | {'Recall@1':>8} | {'Recall@3':>8}")
print("-" * 42)
for taille in [15,20,25, 30, 60]:
    chev = max(1, taille // 4)
    col = construire_index(taille, chev)
    print(f"{taille:>7} | {col.count():>6} | {recall(col, 1):>8.0%} | {recall(col, 3):>8.0%}")