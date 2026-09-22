from pathlib import Path

corpus = {
    "metformine.txt": (
        "La metformine est un medicament antidiabetique de la famille des biguanides. "
        "Elle est prescrite en premiere intention dans le diabete de type 2. "
        "Son role principal est de faire baisser la glycemie en diminuant la production "
        "de glucose par le foie et en ameliorant la sensibilite a l'insuline. "
        "Les effets indesirables les plus frequents sont les troubles digestifs."
    ),
    "aspirine.txt": (
        "L'aspirine est un medicament antalgique, antipyretique et anti-inflammatoire. "
        "A faible dose, elle est aussi utilisee comme antiagregant plaquettaire. "
        "Elle est prescrite apres un infarctus du myocarde pour prevenir les recidives. "
        "Son usage prolonge peut provoquer des saignements digestifs."
    ),
    "hypertension.txt": (
        "L'hypertension arterielle est une pression du sang trop elevee dans les arteres. "
        "Elle augmente le risque d'infarctus, d'accident vasculaire cerebral et d'insuffisance renale. "
        "Le traitement repose sur des mesures hygieno-dietetiques et des medicaments. "
        "La tension se mesure en millimetres de mercure."
    ),
    "diabete.txt": (
        "Le diabete de type 2 est une maladie chronique caracterisee par un exces de sucre dans le sang. "
        "Il est souvent lie au surpoids et a la sedentarite. "
        "Le traitement associe regime alimentaire, activite physique et medicaments hypoglycemiants. "
        "La metformine est le traitement de premiere intention."
    ),
}

dossier = Path("data/documents")
dossier.mkdir(parents=True, exist_ok=True)
for nom, contenu in corpus.items():
    (dossier / nom).write_text(contenu, encoding="utf-8")
    print("ecrit :", dossier / nom)

