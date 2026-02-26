# ☕ CaféIA - GraphRAG avec Ollama

Interface graphique Streamlit pour interroger un LLM local via Ollama et alimenter un système RAG (Retrieval-Augmented Generation) avec différents types de documents.

## 🚀 Fonctionnalités

- 📤 **Upload de documents** multiples formats : PDF, DOCX, XLSX, TXT
- 🤖 **Interrogation LLM** via Ollama (modèle qwen2.5:14b)
- 🔍 **4 modes de recherche** :
  - **Naive** : RAG classique, recherche simple
  - **Local** : Entités et relations proches (1 hop)
  - **Global** : Patterns globaux du knowledge graph
  - **Hybrid** : Combinaison local + global (recommandé pour multi-hop)
- 📜 **Historique** des requêtes avec horodatage
- 🎨 **Interface moderne** et intuitive

## 📋 Prérequis

### 1. Ollama installé et lancé

```bash
# Installer Ollama (si pas déjà fait)
curl -fsSL https://ollama.com/install.sh | sh

# Lancer Ollama
ollama serve
```

### 2. Télécharger les modèles nécessaires

```bash
# Modèle LLM (32K context)
ollama pull qwen2.5:14b

# Modèle d'embeddings
ollama pull nomic-embed-text
```

### 3. Python 3.11+

Vérifiez votre version de Python :
```bash
python3 --version
```

## 🛠️ Installation

### 1. Activer l'environnement virtuel

```bash
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 🎯 Utilisation

### Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse : `http://localhost:8501`

### Workflow recommandé

1. **Uploader des documents** (onglet "📤 Upload Documents")
   - Glissez-déposez vos fichiers PDF, DOCX, XLSX ou TXT
   - Cliquez sur "🚀 Indexer les documents"
   - Attendez la confirmation d'indexation

2. **Interroger le RAG** (onglet "💬 Interroger le RAG")
   - Saisissez votre question
   - Choisissez le mode de recherche (hybrid recommandé)
   - Cliquez sur "🔍 Rechercher"

3. **Consulter l'historique** (onglet "📜 Historique")
   - Visualisez toutes vos requêtes précédentes
   - Consultez les réponses obtenues

## 📁 Structure du projet

```
CaféIA/
├── app.py                  # Interface Streamlit principale
├── main.py                 # Configuration LightRAG + Ollama
├── document_processor.py   # Extraction de texte (PDF, DOCX, XLSX, TXT)
├── insert_docs.py          # Script d'insertion de documents (CLI)
├── query_demo.py           # Script de démonstration des requêtes (CLI)
├── requirements.txt        # Dépendances Python
├── storage/                # Dossier du knowledge graph (créé automatiquement)
└── venv/                   # Environnement virtuel Python
```

## 🔧 Configuration avancée

Vous pouvez modifier les paramètres dans [main.py](main.py) :

- **Modèle LLM** : `llm_model_name="qwen2.5:14b"`
- **Context window** : `llm_model_max_token_size=32768`
- **Parallélisme** : `llm_model_max_async=2`
- **Dimensions embeddings** : `embedding_dim=768`

## 🐛 Dépannage

### Erreur "Connection refused" lors du lancement

- Vérifiez qu'Ollama est bien lancé : `ollama serve`
- Vérifiez que les modèles sont téléchargés : `ollama list`

### Erreur lors de l'extraction de documents

- Assurez-vous que toutes les dépendances sont installées :
  ```bash
  pip install PyPDF2 python-docx openpyxl
  ```

### Réinitialiser le knowledge graph

Si vous voulez repartir de zéro :
```bash
rm -rf storage/
```

## 🚀 Scripts CLI (optionnels)

### Insérer des documents en ligne de commande

```bash
python insert_docs.py
```

### Tester les différents modes de requête

```bash
python query_demo.py
```

## 📚 Documentation

- [LightRAG Documentation](https://github.com/HKUDS/LightRAG)
- [Ollama Documentation](https://ollama.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📝 Licence

Ce projet est fourni à titre éducatif et de démonstration.

---

**☕ Développé avec passion pour CaféIA**
