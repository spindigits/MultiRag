"""
config_manager.py — CaféIA
Gère la configuration du couplage LLM + Embedding.
Supporte deux modèles séparés : indexation (local) vs query (local ou Mistral API).
"""

import json
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

CONFIG_FILE = ".cafeia_config.json"

RECOMMENDED_COUPLES = [
    {"id": "qualite",  "label": "🥇 Qualité maximale",         "llm": "qwen2.5:32b", "embedding": "bge-m3",           "embedding_dim": 1024, "note": "Specs officielles LightRAG — 32B min recommandé",          "min_vram_gb": 20},
    {"id": "equilibre","label": "🥈 Équilibre qualité/vitesse", "llm": "qwen2.5:14b", "embedding": "mxbai-embed-large","embedding_dim": 1024, "note": "Bon compromis — surpasse OpenAI text-embedding-3-large",      "min_vram_gb": 10},
    {"id": "demo",     "label": "🥉 Démo rapide",               "llm": "qwen2.5:7b",  "embedding": "nomic-embed-text", "embedding_dim": 768,  "note": "Vitesse avant qualité — idéal pour démos et workshops",     "min_vram_gb": 6},
    {"id": "custom",   "label": "⚙️ Configuration manuelle",    "llm": None,          "embedding": None,               "embedding_dim": None, "note": "Choisir manuellement parmi les modèles disponibles",        "min_vram_gb": 0},
]

QUERY_PROVIDERS = [
    {"id": "mistral",     "label": "⚡ Mistral API",                       "model": "mistral-small-latest", "note": "2-5 sec/réponse — souverain FR 🇫🇷",          "requires_key": True,  "base_url": "https://api.mistral.ai/v1"},
    {"id": "ollama_same", "label": "🖥️ Ollama — même modèle (indexation)", "model": None,                  "note": "Simple mais lent (40-60s avec 14b)",          "requires_key": False, "base_url": None},
    {"id": "ollama_7b",   "label": "🖥️ Ollama — qwen2.5:7b (rapide)",     "model": "qwen2.5:7b",          "note": "15-20 sec/réponse — bon compromis local",     "requires_key": False, "base_url": None},
]

CONTEXT_PROFILES = {
    # Vrais noms QueryParam v1.4.9.11 :
    # top_k=entités graph, chunk_top_k=chunks vectoriels,
    # max_entity_tokens, max_relation_tokens, max_total_tokens
    "lean": {
        "label": "🚀 Lean (rapide)",
        "top_k": 10, "chunk_top_k": 5,
        "max_entity_tokens": 2000, "max_relation_tokens": 2000, "max_total_tokens": 8000,
        "note": "~8k tokens — réponse rapide, moins exhaustif",
    },
    "balanced": {
        "label": "⚖️ Balanced (recommandé)",
        "top_k": 20, "chunk_top_k": 10,
        "max_entity_tokens": 3000, "max_relation_tokens": 4000, "max_total_tokens": 15000,
        "note": "~15k tokens — bon équilibre qualité/vitesse",
    },
    "full": {
        "label": "📚 Full (défauts LightRAG)",
        "top_k": 40, "chunk_top_k": 20,
        "max_entity_tokens": 6000, "max_relation_tokens": 8000, "max_total_tokens": 30000,
        "note": "~30k tokens — defaults LightRAG, lent en local",
    },
}


@dataclass
class CafeiaConfig:
    llm_model: str
    embedding_model: str
    embedding_dim: int
    couple_id: str
    query_provider: str = "ollama_same"
    query_model: str = ""
    mistral_api_key: str = ""
    context_profile: str = "balanced"
    configured: bool = True

    def get_query_model(self) -> str:
        if self.query_provider == "ollama_same":
            return self.llm_model
        return self.query_model or self.llm_model

    def get_context_params(self) -> dict:
        return CONTEXT_PROFILES.get(self.context_profile, CONTEXT_PROFILES["balanced"])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        defaults = {"query_provider": "ollama_same", "query_model": "", "mistral_api_key": "", "context_profile": "balanced"}
        for k, v in defaults.items():
            d.setdefault(k, v)
        return cls(**d)


def load_config() -> Optional[CafeiaConfig]:
    if not Path(CONFIG_FILE).exists():
        return None
    try:
        return CafeiaConfig.from_dict(json.loads(Path(CONFIG_FILE).read_text()))
    except Exception:
        return None


def save_config(config: CafeiaConfig):
    Path(CONFIG_FILE).write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))


def get_ollama_models() -> dict:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        embed_kw = ["embed", "bge", "nomic", "mxbai", "minilm", "e5"]
        emb = [m for m in models if any(k in m.lower() for k in embed_kw)]
        llm = [m for m in models if m not in emb]
        return {"llm": llm, "embedding": emb, "all": models}
    except Exception:
        return {"llm": [], "embedding": [], "all": [], "error": "Ollama inaccessible"}


def check_couple_availability(couple: dict, available_models: list) -> dict:
    if couple["id"] == "custom":
        return {**couple, "llm_ok": True, "embed_ok": True, "available": True}
    def avail(name, models):
        return any(name.lower() in m.lower() or m.lower().startswith(name.lower()) for m in models)
    llm_ok  = avail(couple["llm"],       available_models)
    emb_ok  = avail(couple["embedding"], available_models)
    return {**couple, "llm_ok": llm_ok, "embed_ok": emb_ok, "available": llm_ok and emb_ok}


def test_mistral_key(api_key: str) -> bool:
    try:
        r = requests.get("https://api.mistral.ai/v1/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False