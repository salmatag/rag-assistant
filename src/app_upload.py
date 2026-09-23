from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gradio as gr
from src.generation import generer_reponse

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

TAILLE_MAX_MO = 10
EXTENSIONS = (".pdf", ".txt")

_cache = {}


def verifier_fichier(chemin):
    p = Path(chemin)
    if p.suffix.lower() not in EXTENSIONS:
        return f"Type non autorise. Formats acceptes : {', '.join(EXTENSIONS)}"
    if p.stat().st_size > TAILLE_MAX_MO * 1024 * 1024:
        return f"Fichier trop volumineux (max {TAILLE_MAX_MO} Mo)."
    return None


def lire_fichier(chemin):
    if chemin.lower().endswith(".pdf"):
        reader = PdfReader(chemin)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return Path(chemin).read_text(encoding="utf-8", errors="ignore")


def decouper(texte, taille=120, chevauchement=30):
    mots = texte.split()
    morceaux, debut = [], 0
    while debut < len(mots):
        morceaux.append(" ".join(mots[debut:debut + taille]))
        debut += taille - chevauchement
    return morceaux


def indexer(chemin):
    chunks = decouper(lire_fichier(chemin))
    return chunks, model.encode(chunks)


def repondre(fichier, question, backend):
    if not fichier:
        return "Ajoute d'abord un fichier."
    erreur = verifier_fichier(fichier)
    if erreur:
        return erreur
    if not question.strip():
        return "Ecris une question."

    if backend == "groq":
        note = "**Mode cloud : le contenu est envoye a Groq.** Pour des donnees sensibles, choisis Ollama (local)."
    else:
        note = "Mode local : les donnees restent sur ta machine."

    if fichier not in _cache:
        _cache[fichier] = indexer(fichier)
    chunks, vecteurs = _cache[fichier]

    scores = cosine_similarity(model.encode([question]), vecteurs)[0]
    indices = np.argsort(scores)[::-1][:3]
    passages = [(chunks[i], Path(fichier).name) for i in indices]

    reponse = generer_reponse(question, passages, backend=backend)
    extraits = "\n\n".join(f"- {p[:180]}..." for p, _ in passages)
    return f"{reponse}\n\n**Passages sources :**\n\n{extraits}\n\n---\n{note}"


demo = gr.Interface(
    fn=repondre,
    inputs=[
        gr.File(label="Ton document (PDF ou TXT)", file_types=[".pdf", ".txt"]),
        gr.Textbox(label="Ta question"),
        gr.Dropdown(["groq", "ollama"], value="groq", label="Moteur"),
    ],
    outputs=gr.Markdown(label="Reponse"),
    title="RAG sur ton document",
)

demo.launch(server_name="0.0.0.0", server_port=7861)