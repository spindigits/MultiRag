"""
Configuration LightRAG + Ollama — syntaxe officielle README HKUDS
CORRECTION : ollama_embed est déjà décoré → ne pas le re-wrapper dans EmbeddingFunc()
Utiliser @wrap_embedding_func_with_attrs + ollama_embed.func
"""

import asyncio
import os
import numpy as np
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import wrap_embedding_func_with_attrs

WORKING_DIR = "./storage"
os.makedirs(WORKING_DIR, exist_ok=True)


# ─── Embedding : syntaxe officielle LightRAG ──────────────────────────────────
# ollama_embed est DÉJÀ décoré avec @wrap_embedding_func_with_attrs
# → on NE PAS le re-wrapper dans EmbeddingFunc()
# → on utilise ollama_embed.func pour accéder à la fonction brute

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=8192,
    model_name="nomic-embed-text"
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embed.func(
        texts,
        embed_model="nomic-embed-text",
        host="http://localhost:11434"
    )
# ──────────────────────────────────────────────────────────────────────────────


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_batch_num=1,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",
        llm_model_max_async=2,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768}
        },
        max_total_tokens=32768,
        chunk_token_size=1200,
        embedding_func=embedding_func,   # fonction décorée, pas EmbeddingFunc()
    )

    await rag.initialize_storages()

    try:
        from lightrag.operate import initialize_pipeline_status
        await initialize_pipeline_status(rag)
    except (ImportError, AttributeError):
        pass

    return rag


if __name__ == "__main__":
    rag = asyncio.run(initialize_rag())
    print("✅ GraphRAG initialisé avec succès")