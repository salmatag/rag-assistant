# RAG Assistant

Assistant de recherche documentaire (RAG) — Projet 1 du portfolio Data/IA.

Objectif : répondre à des questions en s'appuyant sur **tes propres documents**,
avec **citations des sources**, et mesurer la qualité des réponses.

## Feuille de route

- **Phase A** — Fondations : embeddings, similarité cosinus, base vectorielle
- **Phase B** — V1 qui marche : ingestion → découpage → recherche → réponse
- **Phase C** — Évaluation : Recall@k, fidélité des réponses
- **Phase D** — Industrialisation : API FastAPI → Docker → CI → déploiement

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Structure

```
rag-assistant/
|-- data/            # documents sources (non versionnés)
|-- src/             # code réutilisable
|-- notebooks/       # expérimentations
|-- tests/           # tests
|-- requirements.txt
|-- README.md
```
