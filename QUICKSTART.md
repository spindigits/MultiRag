# 🚀 Guide de Démarrage Rapide - CaféIA

## Prérequis rapides

### 1. Vérifier Ollama
```bash
# Vérifier qu'Ollama est installé
ollama --version

# Lancer Ollama (dans un terminal séparé)
ollama serve
```

### 2. Télécharger les modèles (une seule fois)
```bash
# Modèle LLM (~8 Go)
ollama pull qwen2.5:14b

# Modèle d'embeddings (~500 Mo)
ollama pull nomic-embed-text
```

## Lancement de l'application

### Méthode 1 : Script automatique (recommandé)
```bash
./run_app.sh
```

### Méthode 2 : Lancement manuel
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'application
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à : **http://localhost:8501**

## Utilisation rapide

### Étape 1 : Uploader des documents
1. Allez dans l'onglet **"📤 Upload Documents"**
2. Glissez-déposez vos fichiers (PDF, DOCX, XLSX, TXT)
3. Cliquez sur **"🚀 Indexer les documents"**
4. Attendez la confirmation d'indexation

### Étape 2 : Poser des questions
1. Allez dans l'onglet **"💬 Interroger le RAG"**
2. Tapez votre question
3. Choisissez le mode (hybrid recommandé)
4. Cliquez sur **"🔍 Rechercher"**

### Étape 3 : Consulter l'historique
- Onglet **"📜 Historique"** pour voir toutes vos requêtes précédentes

## Modes de recherche

| Mode | Description | Quand l'utiliser |
|------|-------------|------------------|
| **Naive** | RAG classique simple | Questions simples sur un document |
| **Local** | Relations proches (1 hop) | Questions sur des entités spécifiques |
| **Global** | Patterns globaux | Questions sur des tendances générales |
| **Hybrid** | Combinaison local + global | Questions complexes multi-hop (recommandé) |

## Résolution rapide de problèmes

### Ollama n'est pas accessible
```bash
# Dans un nouveau terminal, lancez :
ollama serve
```

### Les modèles ne sont pas téléchargés
```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### Erreur de dépendances Python
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Réinitialiser le knowledge graph
```bash
rm -rf storage/
# Puis ré-indexez vos documents
```

## Scripts CLI disponibles

### Insérer des documents en ligne de commande
```bash
python insert_docs.py
```

### Tester les requêtes en CLI
```bash
python query_demo.py
```

## Support

Pour plus d'informations, consultez le [README.md](README.md) complet.

---

**☕ Bon café et bonnes recherches !**
