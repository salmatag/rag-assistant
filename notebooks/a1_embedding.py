from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

phrase = "Le chat dort sur le canapé."
vecteur = model.encode(phrase)

print("Dimension (nombre de nombres) :", vecteur.shape)
print("Les 10 premières valeurs :", vecteur[:10])