from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

phrases = [
    "Le chat dort sur le canapé.",
    "Un chat est un animal domestique.",
    "La voiture roule vite sur l'autoroute.",
    "Le train part à 8 heures.",
]

vecteurs = model.encode(phrases)
sims = cosine_similarity(vecteurs)

print("        " + "  ".join(f"P{j+1}" for j in range(len(phrases))))
for i, ligne in enumerate(sims):
    print(f"P{i+1}  " + "  ".join(f"{v:.2f}" for v in ligne))