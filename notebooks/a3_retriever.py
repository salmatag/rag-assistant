from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Le paracétamol est utilisé contre la douleur et la fièvre.",
    "L'aspirine peut réduire l'inflammation.",
    "La metformine est prescrite pour le diabète de type 2.",
    "L'ibuprofène est un anti-inflammatoire non stéroïdien.",
    "La tension artérielle se mesure en millimètres de mercure.",
]

def rechercher(question, documents, top_k=2):
    doc_vecs = model.encode(documents)
    q_vec = model.encode([question])
    scores = cosine_similarity(q_vec, doc_vecs)[0]
    indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[i], scores[i]) for i in indices]

question = "Quel médicament fait baisser le sucre dans le sang ?"
for doc, score in rechercher(question, documents):
    print(f"{score:.3f}  {doc}")