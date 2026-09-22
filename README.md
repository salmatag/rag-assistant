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