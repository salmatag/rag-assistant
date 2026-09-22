import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "Tu es un assistant documentaire. Reponds UNIQUEMENT a partir du contexte fourni. "
    "Cite la source de chaque information entre crochets, par exemple [metformine.txt]. "
    "Si la reponse ne figure pas dans le contexte, dis clairement que tu ne sais pas."
)


def construire_prompt(question, passages):
    contexte = "\n\n".join(f"[{source}] {texte}" for texte, source in passages)
    return f"Contexte:\n{contexte}\n\nQuestion: {question}"


def _groq(prompt):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    reponse = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return reponse.choices[0].message.content


def _ollama(prompt, model="llama3.2:1b"):
    import ollama
    reponse = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return reponse["message"]["content"]


def generer_reponse(question, passages, backend="groq"):
    prompt = construire_prompt(question, passages)
    if backend == "groq":
        return _groq(prompt)
    if backend == "ollama":
        return _ollama(prompt)
    raise ValueError(f"backend inconnu : {backend}")