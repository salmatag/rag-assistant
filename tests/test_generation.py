import pytest
from src.generation import construire_prompt, generer_reponse


def test_prompt_contient_la_question():
    passages = [("Le paracetamol calme la douleur.", "para.txt")]
    prompt = construire_prompt("Comment calmer la douleur ?", passages)
    assert "Comment calmer la douleur ?" in prompt


def test_prompt_contient_la_source_et_le_texte():
    passages = [("La metformine traite le diabete.", "metformine.txt")]
    prompt = construire_prompt("question", passages)
    assert "metformine.txt" in prompt
    assert "La metformine traite le diabete." in prompt


def test_backend_inconnu_leve_une_erreur():
    with pytest.raises(ValueError):
        generer_reponse("question", [("texte", "source.txt")], backend="inconnu")