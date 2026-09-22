import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from src.generation import generer_reponse

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("documents")
juge = Groq(api_key=os.environ["GROQ_API_KEY"])

jeu_test = [
    "Quel medicament est utilise pour le diabete de type 2 ?",
    "Quels sont les effets indesirables de la metformine ?",
    "Que risque-t-on avec l'aspirine sur le long terme ?",
    "L'aspirine est-elle utilisee apres un infarctus ?",
    "Qu'est-ce que l'hypertension arterielle ?",
    "Comment se mesure la tension ?",
    "Le diabete de type 2 est-il lie au surpoids ?",
    "Le traitement du diabete associe-t-il regime et activite physique ?",
]


def rechercher(question, k=3):
    q = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def juger(contexte, reponse):
    prompt = (
        f"Contexte:\n{contexte}\n\nReponse:\n{reponse}\n\n"
        "La reponse est-elle entierement soutenue par le contexte ci-dessus ? "
        "Reponds uniquement par OUI ou NON."
    )
    r = juge.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content.strip().upper().startswith("OUI")


for backend in ["groq", "ollama"]:
    fideles = 0
    for question in jeu_test:
        passages = rechercher(question)
        contexte = "\n".join(texte for texte, _ in passages)
        reponse = generer_reponse(question, passages, backend=backend)
        ok = juger(contexte, reponse)
        fideles += ok
        print(f"[{'FIDELE' if ok else 'NON FIDELE'}] {question}")
    print(f"--> backend {backend} : fidelite = {fideles}/{len(jeu_test)} = {fideles/len(jeu_test):.0%}\n")