#!/bin/bash

# Script de lancement de l'application CaféIA

echo "🚀 Lancement de CaféIA - GraphRAG avec Ollama"
echo "=============================================="

# Activer l'environnement virtuel
source venv/bin/activate

# Vérifier qu'Ollama est lancé
echo "📡 Vérification qu'Ollama est disponible..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  ATTENTION: Ollama ne semble pas être lancé!"
    echo "   Lancez Ollama avec: ollama serve"
    echo ""
    read -p "Voulez-vous continuer quand même ? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Ollama est disponible"
fi

# Lancer Streamlit
echo ""
echo "🌟 Lancement de l'interface Streamlit..."
echo "   L'application va s'ouvrir dans votre navigateur"
echo "   URL: http://localhost:8501"
echo ""
echo "Pour arrêter l'application, appuyez sur Ctrl+C"
echo ""

streamlit run app.py
