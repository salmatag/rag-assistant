# RAG Assistant

Assistant de recherche documentaire (RAG) : répond à des questions à partir de
**documents fournis**, avec **citations des sources**, et **mesure** la qualité du retrieval
et de la génération. Fonctionne avec un LLM **local** (Ollama) ou **cloud** (Groq).

## Fonctionnalités

- Recherche sémantique (embeddings) dans tes propres documents
- Génération de réponses **ancrées dans le contexte**, avec **sources**
- Deux moteurs de génération interchangeables : local (Ollama) et cloud (Groq)
- Évaluation chiffrée : Recall@k (retrieval) et fidélité (anti-hallucination)

## Architecture

```
Documents → Découpage (chunks) → Embeddings → Base vectorielle (Chroma)
                                                          |
Question -------------------------------------------------+
                                                          v
                                        Retrieval (top-k passages)
                                                          |
                                                          v
                                  LLM (Ollama local / Groq cloud) → Réponse + sources
```

## Stack

- `sentence-transformers` : embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- `chromadb` : base vectorielle
- `groq` / `ollama` : génération (cloud / local)
- `FastAPI` : API REST
- `python-dotenv` : gestion des secrets

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Génération locale (optionnelle) :

```powershell
ollama pull llama3.2:1b
```

Clé cloud : crée un fichier `.env` (voir `.env.example`) :

```text
GROQ_API_KEY=ta_cle_ici
```

## Utilisation

Construire l'index :

```powershell
python notebooks\b3_index_chroma.py
```

Lancer le RAG en ligne de commande :

```powershell
python notebooks\b5_rag.py
```

Lancer l'API :

```powershell
python -m uvicorn src.api:app --reload
```

Puis ouvrir http://127.0.0.1:8000/docs pour tester `POST /ask`.

Exemple d'appel :

```json
POST /ask
{
  "question": "Quel medicament est utilise pour le diabete de type 2 ?",
  "backend": "groq",
  "k": 3
}
```

## Évaluation

**Protocole** : corpus de 4 documents biomédicaux ; chunks de 60 mots (chevauchement 15) ;
embeddings `paraphrase-multilingual-MiniLM-L12-v2` ; index Chroma ; température 0 pour l'évaluation ;
jeu de test de 8 questions avec le document attendu.

### Retrieval — Recall@k

| k | Recall@k |
|---|---|
| 1 | 88% |
| 2 | 100% |
| 3 | 100% |
| 5 | 100% |

### Génération — Fidélité (LLM-as-a-judge)

| Backend | Modèle | Température | Fidélité |
|---|---|---|---|
| Groq (cloud) | gpt-oss-20b (20B) | 0 | 100% |
| Ollama (local) | llama3.2:1b (1B) | 0 | 25% |

### Benchmark — taille des chunks

| Mots/chunk | Chunks | Recall@1 | Recall@3 |
|---|---|---|---|
| 15 | 17 | 62% | 88% |
| 20 | 14 | 75% | 88% |
| 25 | 12 | 75% | 88% |
| 30 | 9 | 62% | 100% |
| 60 | 6 | 88% | 100% |

### Conclusions

- Chunks trop petits (15 mots) dégradent le retrieval ; **60 mots est optimal** ici.
- Retrieval solide (Recall@2 = 100%) ; le Recall@1 (88%) révèle une confusion entre documents proches.
- La fidélité dépend fortement du modèle : **100% (20B cloud) vs 25% (1B local)**.

### Limites

- Jeu de test réduit (8 questions) → écarts peu significatifs.
- Juge = même modèle que le générateur (biais d'auto-évaluation).
- Corpus minuscule pour la démonstration.

## Structure

```
rag-assistant/
|-- data/            # corpus + index Chroma (non versionnés)
|-- notebooks/       # scripts du projet (A → D)
|-- src/
|   |-- generation.py  # génération (Groq / Ollama)
|   `-- api.py         # API FastAPI
|-- tests/
|-- requirements.txt
`-- README.md
```
