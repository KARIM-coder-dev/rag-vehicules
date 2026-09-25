"""
Tests de la configuration : valeurs par défaut, surcharge par variable
d'environnement, et refus des configurations invalides au démarrage.
"""

import pytest
from pydantic import ValidationError

from rag_vehicules.config import Settings


@pytest.fixture(autouse=True)
def env_vierge(monkeypatch):
    """Isole chaque test de la config de la machine : sans ça, une ligne
    LLM_MODEL=... dans le .env local ferait échouer (ou réussir) les tests
    différemment chez chaque développeur et en CI."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-factice")
    return monkeypatch


def test_valeurs_par_defaut(env):
    s = Settings()
    assert s.llm_model == "gpt-4o-mini"
    assert s.retriever_k == 50
    assert s.manifest_file.name == "manifest.json"


def test_surcharge_par_variable_d_environnement(env):
    env.setenv("LLM_MODEL", "gpt-4o")
    env.setenv("RETRIEVER_K", "30")
    s = Settings()
    assert s.llm_model == "gpt-4o"
    assert s.retriever_k == 30


def test_cle_api_manquante_refusee(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="openai_api_key"):
        Settings()


def test_cle_api_vide_refusee(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValidationError, match="openai_api_key"):
        Settings()


def test_cle_api_jamais_affichee(env):
    s = Settings()
    assert "sk-test-factice" not in repr(s)
    assert "sk-test-factice" not in str(s.openai_api_key)


def test_valeur_hors_bornes_refusee(env):
    env.setenv("LLM_TEMPERATURE", "5")
    with pytest.raises(ValidationError):
        Settings()


def test_chunk_overlap_incoherent_refuse(env):
    env.setenv("CHUNK_SIZE", "100")
    env.setenv("CHUNK_OVERLAP", "200")
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings()
