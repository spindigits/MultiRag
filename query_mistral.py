#!/usr/bin/env python3
"""
query_mistral.py — Requêtes LightRAG via REST API + réponse Mistral

lightrag-server gère l'indexation (Ollama).
Ce script bypass le LLM du serveur pour la QUERY :
  1. Appel /query?only_need_context=true  → contexte graph assemblé
  2. Envoi du contexte à Mistral API       → réponse rapide (2-5 sec)

Usage :
    python query_mistral.py "Quels clients ont acheté Bio ?" --mode hybrid
    python query_mistral.py "..." --mode naive    # pour comparer
"""

import argparse
import os
import sys
import json
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
LIGHTRAG_HOST  = os.getenv("LIGHTRAG_HOST", "http://localhost:9621")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL  = "mistral-small-latest"

# ── Couleurs terminal ─────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"; RESET = "\033[0m"


def get_context(question: str, mode: str) -> str:
    """Récupère le contexte assemblé par LightRAG sans appel LLM."""
    url = f"{LIGHTRAG_HOST}/query"
    payload = {
        "query": question,
        "mode": mode,
        "only_need_context": True,
        "top_k": 40,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # lightrag-server retourne {"response": "...context..."} ou {"context": "..."}
    return data.get("response") or data.get("context") or str(data)


def query_mistral(context: str, question: str) -> str:
    """Envoie contexte + question à Mistral API."""
    if not MISTRAL_API_KEY:
        print(f"{R}MISTRAL_API_KEY non défini. Export la variable.{RESET}")
        sys.exit(1)

    client = OpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
    )

    system_prompt = (
        "Tu es un assistant expert pour la jardinerie Verde Occitanie. "
        "Réponds en français de manière précise et concise en te basant "
        "UNIQUEMENT sur le contexte fourni. "
        "Si la réponse nécessite de croiser plusieurs informations, "
        "explique les liens entre elles."
    )

    user_msg = f"""Contexte extrait du knowledge graph :

{context}

---
Question : {question}
"""

    response = client.chat.completions.create(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Query LightRAG + Mistral API")
    parser.add_argument("question", help="La question à poser")
    parser.add_argument("--mode", choices=["naive", "local", "global", "hybrid"],
                        default="hybrid", help="Mode de recherche (défaut: hybrid)")
    parser.add_argument("--show-context", action="store_true",
                        help="Afficher le contexte brut récupéré")
    args = parser.parse_args()

    print(f"\n{B}━━━ CaféIA Query ━━━{RESET}")
    print(f"{Y}Question :{RESET} {args.question}")
    print(f"{Y}Mode     :{RESET} {args.mode}")
    print()

    # Étape 1 : contexte graph
    print(f"{G}[1/2] Récupération contexte LightRAG ({args.mode})...{RESET}", end=" ", flush=True)
    try:
        context = get_context(args.question, args.mode)
        ctx_tokens = len(context.split())
        print(f"OK ({ctx_tokens} mots)")
    except requests.exceptions.ConnectionError:
        print(f"\n{R}Erreur : lightrag-server inaccessible sur {LIGHTRAG_HOST}{RESET}")
        print("Lance : lightrag-server --config cafeia.env")
        sys.exit(1)
    except Exception as e:
        print(f"\n{R}Erreur : {e}{RESET}")
        sys.exit(1)

    if args.show_context:
        print(f"\n{Y}── Contexte brut ──{RESET}")
        print(context[:2000], "..." if len(context) > 2000 else "")
        print()

    # Étape 2 : Mistral API
    print(f"{G}[2/2] Génération réponse Mistral API...{RESET}", end=" ", flush=True)
    answer = query_mistral(context, args.question)
    print("OK")

    print(f"\n{B}━━━ Réponse ━━━{RESET}")
    print(answer)
    print(f"\n{B}━━━━━━━━━━━━━━━{RESET}\n")


if __name__ == "__main__":
    main()
