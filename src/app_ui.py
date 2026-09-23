from sentence_transformers import SentenceTransformer
import chromadb
import gradio as gr
from src.generation import generer_reponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("documents")


def rechercher(question, k=3):
    q = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q, n_results=k)
    return [(doc, meta["source"]) for doc, meta in zip(res["documents"][0], res["metadatas"][0])]


def repondre(question, backend, k):
    if not question.strip():
        return "Ecris une question."
    passages = rechercher(question, int(k))
    reponse = generer_reponse(question, passages, backend=backend)
    sources = sorted({src for _, src in passages})
    return f"{reponse}\n\n**Sources :** {', '.join(sources)}"


demo = gr.Interface(
    fn=repondre,
    inputs=[
        gr.Textbox(label="Ta question"),
        gr.Dropdown(["groq", "ollama"], value="groq", label="Moteur"),
        gr.Slider(1, 5, value=3, step=1, label="k (nombre de passages)"),
    ],
    outputs=gr.Markdown(label="Reponse"),
    title="RAG Assistant",
    description="Pose une question sur les documents fournis.",
)

demo.launch()