#!/usr/bin/env python3
"""
fix_dimensions.py
─────────────────
Diagnostique et corrige le mismatch de dimensions d'embedding entre
Ollama/nomic-embed-text (768) et LightRAG/NanoVectorDB (défaut 1024).

Usage :
    python fix_dimensions.py           # diagnostic + fix auto
    python fix_dimensions.py --check   # diagnostic seul
"""

import os
import sys
import json
import glob
import shutil
import struct
import argparse
import requests
from pathlib import Path

STORAGE_DIR   = "./storage"
OLLAMA_HOST   = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
EXPECTED_DIM  = 768   # nomic-embed-text


# ── 1. Vérifier la dim réelle renvoyée par Ollama ─────────────────────────────

def get_real_embedding_dim() -> int:
    """Envoie un texte test à Ollama et mesure la dim réelle."""
    print(f"[1] Test embedding Ollama ({EMBED_MODEL})...")
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": "test"},
            timeout=30
        )
        r.raise_for_status()
        vec = r.json().get("embedding", [])
        dim = len(vec)
        print(f"    → Dimension réelle : {dim}")
        return dim
    except Exception as e:
        print(f"    ❌ Ollama inaccessible : {e}")
        return -1


# ── 2. Scanner le storage pour trouver les dims stockées ──────────────────────

def scan_storage_dims():
    """Cherche les fichiers NanoVectorDB et lit leur dimension déclarée."""
    print(f"\n[2] Scan du storage : {STORAGE_DIR}/")

    if not os.path.exists(STORAGE_DIR):
        print("    Storage inexistant — rien à corriger.")
        return {}

    dims_found = {}

    # NanoVectorDB stocke les vecteurs dans des fichiers .json ou binaires
    for path in Path(STORAGE_DIR).rglob("*"):
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # NanoVectorDB embedding_dim field
                if "embedding_dim" in data:
                    dims_found[str(path)] = data["embedding_dim"]
                    print(f"    {path.name}: embedding_dim = {data['embedding_dim']}")
                # matrix field : first vector length
                if "matrix" in data and isinstance(data["matrix"], list) and data["matrix"]:
                    first = data["matrix"][0]
                    if isinstance(first, list):
                        dims_found[str(path) + "#matrix"] = len(first)
                        print(f"    {path.name}: matrix vector len = {len(first)}")
            except Exception:
                pass

        elif path.suffix in (".npy", ".pkl"):
            print(f"    {path.name} : binaire ({path.suffix}) — ignoré dans le scan")

    if not dims_found:
        print("    Aucun fichier avec dimension trouvé (storage peut-être vide).")

    return dims_found


# ── 3. Patcher les fichiers JSON qui ont la mauvaise dim ──────────────────────

def patch_storage(real_dim: int):
    """Corrige embedding_dim dans tous les JSON du storage."""
    print(f"\n[3] Patch storage → forcer embedding_dim={real_dim}...")

    patched = 0
    for path in Path(STORAGE_DIR).rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            changed = False

            if "embedding_dim" in data and data["embedding_dim"] != real_dim:
                print(f"    PATCH {path.name}: {data['embedding_dim']} → {real_dim}")
                data["embedding_dim"] = real_dim
                changed = True

            # Si la matrice contient des vecteurs de mauvaise taille → vider
            if "matrix" in data and isinstance(data["matrix"], list) and data["matrix"]:
                first = data["matrix"][0]
                if isinstance(first, list) and len(first) != real_dim:
                    print(f"    RESET {path.name}: matrice dim={len(first)} ≠ {real_dim} → vidée")
                    data["matrix"] = []
                    data["data"] = []
                    if "embedding_dim" in data:
                        data["embedding_dim"] = real_dim
                    changed = True

            if changed:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                patched += 1

        except Exception as e:
            print(f"    ⚠️  {path.name} : {e}")

    print(f"    → {patched} fichier(s) patché(s)")
    return patched


# ── 4. Vérifier main.py ───────────────────────────────────────────────────────

def check_main_py(real_dim: int):
    """Vérifie que main.py ne contient pas de dim incorrecte."""
    print("\n[4] Vérification main.py...")
    main_path = Path("main.py")
    if not main_path.exists():
        print("    main.py introuvable.")
        return

    content = main_path.read_text()
    issues = []

    if "vector_db_storage_cls_kwargs" in content:
        issues.append("vector_db_storage_cls_kwargs présent → peut overrider la dim")

    for line in content.split("\n"):
        if "embedding_dim" in line and str(real_dim) not in line and "1024" in line:
            issues.append(f"embedding_dim=1024 détecté : {line.strip()}")

    if issues:
        print("    ⚠️  Problèmes détectés :")
        for i in issues:
            print(f"       - {i}")
        print("    → Corrigez main.py : supprimez vector_db_storage_cls_kwargs")
        print(f"    → Assurez-vous que embedding_dim={real_dim} dans EmbeddingFunc")
    else:
        print(f"    ✅ main.py correct (embedding_dim={real_dim}, pas de conflit)")


# ── 5. Option nuclear : purge totale ─────────────────────────────────────────

def nuke_storage():
    """Supprime tout le storage — solution ultime si patch insuffisant."""
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        os.makedirs(STORAGE_DIR)
        print(f"    💥 Storage purgé : {STORAGE_DIR}/")
    else:
        print(f"    Storage déjà vide.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fix LightRAG embedding dimension mismatch")
    parser.add_argument("--check", action="store_true", help="Diagnostic seul, sans modification")
    parser.add_argument("--nuke",  action="store_true", help="Purge totale du storage (dernier recours)")
    args = parser.parse_args()

    print("=" * 60)
    print("  fix_dimensions.py — LightRAG embedding dim fixer")
    print("=" * 60)

    real_dim = get_real_embedding_dim()
    if real_dim == -1:
        print("\n⚠️  Impossible de contacter Ollama. Vérifiez : ollama serve")
        print(f"   On assume la dim attendue : {EXPECTED_DIM}")
        real_dim = EXPECTED_DIM

    dims_in_storage = scan_storage_dims()
    check_main_py(real_dim)

    if args.check:
        print("\n[--check] Mode lecture seule — aucune modification.")
        sys.exit(0)

    if args.nuke:
        print("\n[--nuke] Purge totale demandée...")
        nuke_storage()
        print("\n✅ Storage purgé. Relancez streamlit run app.py et ré-indexez vos documents.")
        sys.exit(0)

    # Fix automatique
    mismatches = {k: v for k, v in dims_in_storage.items() if v != real_dim}

    if not mismatches and not dims_in_storage:
        print("\n✅ Storage vide ou correct — rien à corriger.")
        print("   Si l'erreur persiste, lancez : python fix_dimensions.py --nuke")
        sys.exit(0)

    if mismatches:
        print(f"\n[AUTO-FIX] {len(mismatches)} mismatch(es) détecté(s) → patch en cours...")
        patched = patch_storage(real_dim)
        if patched > 0:
            print(f"\n✅ {patched} fichier(s) corrigé(s). Relancez : streamlit run app.py")
        else:
            print("\n⚠️  Patch insuffisant — les vecteurs binaires ne peuvent pas être corrigés.")
            print("   Solution : python fix_dimensions.py --nuke")
    else:
        print("\n✅ Toutes les dimensions sont cohérentes.")

    print("=" * 60)


if __name__ == "__main__":
    main()