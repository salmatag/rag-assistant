from pathlib import Path

def decouper(texte, taille=30, chevauchement=8):
    mots = texte.split()
    morceaux = []
    debut = 0
    while debut < len(mots):
        fin = debut + taille
        morceaux.append(" ".join(mots[debut:fin]))
        debut += taille - chevauchement
    return morceaux

dossier = Path("data/documents")
for fichier in sorted(dossier.glob("*.txt")):
    morceaux = decouper(fichier.read_text(encoding="utf-8"))
    print(f"{fichier.name} : {len(morceaux)} chunks")
    print("   chunk 1 :", morceaux[0][:90], "...")