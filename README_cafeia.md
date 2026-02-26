# CaféIA — LightRAG Server Setup
## Remplacement Streamlit → lightrag-server natif

---

## Pourquoi ce changement

Streamlit tourne dans un thread avec son propre event loop asyncio.
LightRAG est entièrement async avec des locks internes.
Résultat : conflits permanents → `lock bound to a different event loop`.

lightrag-server est un FastAPI natif async → aucun conflit, et bonus :
- WebUI React intégrée avec visualisation du knowledge graph
- API REST complète (upload, query, status)
- Compatible Open WebUI / N8N directement

---

## Installation (dans le venv existant)

```bash
cd /home/mensaflow/Desktop/LightRAG
source venv/bin/activate

# Installer le module API (si pas déjà fait)
pip install "lightrag-hku[api]" --break-system-packages

# Copier la config
cp cafeia.env .env
```

---

## Lancement

```bash
# Variables d'environnement
export MISTRAL_API_KEY="ta-clé-mistral"

# Démarrer le serveur
lightrag-server

# WebUI disponible sur :
# http://localhost:9621
```

---

## Workflow complet

### 1. Indexation (WebUI ou curl)

**Via WebUI** : http://localhost:9621 → onglet "Documents" → Upload

**Via curl** (batch) :
```bash
# Upload d'un fichier
curl -X POST http://localhost:9621/documents/upload \
  -F "file=@fournisseurs_certifications.txt"

# Vérifier le statut
curl http://localhost:9621/documents
```

### 2. Query

**Option A — WebUI** : onglet "Query" → choisir mode (naive/local/global/hybrid)

**Option B — Script Mistral (2-5 sec)**  :
```bash
# Query hybrid avec réponse Mistral rapide
python query_mistral.py "Quels clients ont acheté des produits Bio ?" --mode hybrid

# Comparaison naive vs hybrid (démo workshop)
python query_mistral.py "Quels fournisseurs livrent le plus vite ?" --mode naive
python query_mistral.py "Quels fournisseurs livrent le plus vite ?" --mode hybrid

# Voir le contexte graph brut
python query_mistral.py "..." --mode hybrid --show-context
```

**Option C — API REST directe** :
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quels clients Bio ?", "mode": "hybrid"}'
```

---

## Questions de démo multi-hop (workshop UPVD)

Ces questions sont **impossibles en naive**, résolues en **hybrid** :

```
1. "Quels clients ont acheté des produits d'un fournisseur certifié AB ?"
   Hops : client → achat → produit → fournisseur → certification

2. "Quel client professionnel a le plus investi dans l'irrigation ?"
   Hops : client(pro) → achats → produits → fournisseur(AquaGarden)

3. "Quels fournisseurs ont à la fois la certification Bio et un délai < 72h ?"
   Hops : fournisseur → certif + délai (croisement 2 attributs)

4. "Qui a acheté des semences Demeter en Occitanie ?"
   Hops : client → achat → produit → fournisseur(GrainVivant) → certif(Demeter)
```

---

## Purge du storage (re-indexation propre)

```bash
# Supprimer uniquement le statut (garde le graph si déjà indexé)
rm ./rag_storage/kv_store_doc_status.json

# Reset complet
rm -rf ./rag_storage/
```

---

## Architecture finale

```
lightrag-server (FastAPI, port 9621)
│
├── Indexation  →  Ollama qwen2.5:14b  (local, one-time)
│                  + mxbai-embed-large (1024 dim)
│
├── WebUI React  →  http://localhost:9621
│                   upload / query / graph viz
│
├── REST API  →  /documents/upload
│                /query
│                /documents (status)
│
└── query_mistral.py  →  context only_need_context=True
                          → Mistral API mistral-small-latest
                          → réponse en 2-5 sec
```

---

## Streamlit app.py

Le fichier app.py Streamlit reste pour **Phase 02 MensaFlow** (Qdrant + Neo4j hybrid routing).
Pour LightRAG standalone → lightrag-server only.
