"""
CaféIA - GraphRAG avec Ollama + Mistral API
Wizard de configuration au premier lancement :
  - Couplage indexation (LLM local + embedding)
  - Provider de query (Mistral API ou Ollama)
  - Profil de contexte (lean / balanced / full)
"""

import streamlit as st
import os
import threading
import time
from pathlib import Path
from datetime import datetime

import nest_asyncio
nest_asyncio.apply()
import asyncio

from config_manager import (
    load_config, save_config, get_ollama_models,
    check_couple_availability, RECOMMENDED_COUPLES,
    QUERY_PROVIDERS, CONTEXT_PROFILES, CafeiaConfig,
    test_mistral_key
)
from lightrag import QueryParam

st.set_page_config(page_title="CaféIA - GraphRAG", page_icon="☕", layout="wide")

st.markdown("""<style>
/* ── Taille de base globale ── */
html, body, [class*="css"] {
    font-size: 18px !important;
}

/* ── Contenu principal ── */
.main .block-container p,
.main .block-container li,
.main .block-container label,
.main .block-container .stMarkdown,
.main .block-container .stText {
    font-size: 1.15rem !important;
    line-height: 1.7 !important;
}

/* ── Inputs, selects, textareas ── */
.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    font-size: 1.1rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* ── Boutons ── */
.stButton > button {
    font-size: 1.1rem !important;
    padding: 0.5rem 1.2rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    font-size: 1.05rem !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    font-size: 1.05rem !important;
}

/* ── Headers custom ── */
.main-header{font-size:3rem!important;font-weight:700!important;text-align:center!important;color:#1f77b4!important;margin-bottom:0.1rem!important}
.sub-header{font-size:1.15rem;text-align:center;color:#666;margin-bottom:1.5rem}
.section-title{font-size:1.25rem;font-weight:600;margin:1rem 0 .4rem 0;color:#333}
.provider-box{border:2px solid #4CAF50;border-radius:10px;padding:.8rem 1rem;background:#f9fff9;margin:.3rem 0}
.warn-box{border:2px solid #FF9800;border-radius:10px;padding:.8rem 1rem;background:#fffbf0;margin:.3rem 0}
</style>""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── RAG instances — deux caches séparés (indexation vs query) ───────────────

@st.cache_resource(show_spinner="⏳ Init knowledge graph (indexation)...")
def get_index_rag(llm_model: str, embedding_model: str, embedding_dim: int):
    """Instance RAG pour l'indexation — LLM local."""
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192, model_name=embedding_model)
    async def _embed(texts):
        return await ollama_embed.func(texts, embed_model=embedding_model, host="http://localhost:11434")

    async def _init():
        os.makedirs("./storage", exist_ok=True)
        rag = LightRAG(
            working_dir="./storage",
            embedding_batch_num=1, embedding_func_max_async=1,
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model, llm_model_max_async=1,
            llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
            max_total_tokens=32768, chunk_token_size=1200,
            embedding_func=_embed,
        )
        await rag.initialize_storages()
        try:
            from lightrag.operate import initialize_pipeline_status
            await initialize_pipeline_status(rag)
        except (ImportError, AttributeError):
            pass
        return rag

    return run_async(_init())


@st.cache_resource(show_spinner="⏳ Init knowledge graph (query)...")
def get_query_rag(query_provider: str, query_model: str,
                  embedding_model: str, embedding_dim: int,
                  mistral_api_key: str = ""):
    """
    Instance RAG pour la query — peut utiliser Mistral API ou Ollama.
    Cache séparé de l'indexation → les deux coexistent sans conflit.
    """
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192, model_name=embedding_model)
    async def _embed(texts):
        return await ollama_embed.func(texts, embed_model=embedding_model, host="http://localhost:11434")

    async def _init():
        os.makedirs("./storage", exist_ok=True)

        if query_provider == "mistral":
            from lightrag.llm.openai import openai_complete_if_cache as _oai
            _api_key = mistral_api_key
            _qmodel  = query_model

            async def _mistral_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
                # model est passé en 1er arg positionnel — NE PAS le mettre dans kwargs
                kwargs.pop("model", None)
                return await _oai(
                    _qmodel,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    api_key=_api_key,
                    base_url="https://api.mistral.ai/v1",
                    **kwargs
                )

            llm_func   = _mistral_llm
            llm_kwargs = {}
        else:
            from lightrag.llm.ollama import ollama_model_complete
            llm_func   = ollama_model_complete
            llm_kwargs = {"host": "http://localhost:11434", "options": {"num_ctx": 32768}}

        rag = LightRAG(
            working_dir="./storage",
            embedding_batch_num=1, embedding_func_max_async=1,
            llm_model_func=llm_func,
            llm_model_name=query_model, llm_model_max_async=2,
            llm_model_kwargs=llm_kwargs,
            max_total_tokens=32768, chunk_token_size=1200,
            embedding_func=_embed,
        )
        await rag.initialize_storages()
        return rag

    return run_async(_init())


def insert_document(text: str, cfg: CafeiaConfig):
    rag = get_index_rag(cfg.llm_model, cfg.embedding_model, cfg.embedding_dim)
    run_async(rag.ainsert(text))


def query_rag(question: str, mode: str, cfg: CafeiaConfig) -> str:
    rag = get_query_rag(
        cfg.query_provider, cfg.get_query_model(),
        cfg.embedding_model, cfg.embedding_dim,
        cfg.mistral_api_key
    )
    ctx = cfg.get_context_params()
    result = run_async(rag.aquery(
        question,
        param=QueryParam(
            mode=mode,
            top_k=ctx["top_k"],
            chunk_top_k=ctx["chunk_top_k"],
            max_entity_tokens=ctx["max_entity_tokens"],
            max_relation_tokens=ctx["max_relation_tokens"],
            max_total_tokens=ctx["max_total_tokens"],
            enable_rerank=False,   # pas de rerank model configuré
        )
    ))
    return result or "⚠️ Aucune réponse. Vérifiez que des documents sont indexés."


# ─── Session state ────────────────────────────────────────────────────────────

for key, default in [
    ('uploaded_files_count', 0),
    ('uploaded_files_list', []),
    ('query_history', []),
    ('show_wizard', False),
    ('wizard_selected', None),
    ('mistral_key_valid', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# WIZARD — ÉTAPE 1 : Couplage indexation
# ══════════════════════════════════════════════════════════════════════════════

def wizard_step1_indexing(ollama_info):
    st.markdown("### 1️⃣ Modèle d'indexation — extraction du knowledge graph")
    st.caption("Ce modèle est appelé ~80-100 fois par document pour extraire entités et relations.")

    available = ollama_info["all"]
    couples_status = [check_couple_availability(c, available) for c in RECOMMENDED_COUPLES]

    selected_id = st.session_state.wizard_selected

    for c in couples_status:
        is_custom = c["id"] == "custom"
        col_sel, col_info = st.columns([1, 10])
        with col_sel:
            btn_label = "✓" if selected_id == c["id"] else "○"
            btn_type  = "primary" if selected_id == c["id"] else "secondary"
            if st.button(btn_label, key=f"sel_{c['id']}", type=btn_type):
                st.session_state.wizard_selected = c["id"]
                st.rerun()
        with col_info:
            if is_custom:
                st.markdown(f"**{c['label']}** — {c['note']}")
            else:
                llm_badge = "✅" if c["llm_ok"]  else "⬇️"
                emb_badge = "✅" if c["embed_ok"] else "⬇️"
                st.markdown(
                    f"**{c['label']}** — {c['note']}  \n"
                    f"LLM: `{c['llm']}` {llm_badge} &nbsp;|&nbsp; "
                    f"Embedding: `{c['embedding']}` {emb_badge} &nbsp;|&nbsp; dim={c['embedding_dim']}"
                )

    # Config custom
    custom = {"llm": None, "embedding": None, "embedding_dim": 768}
    if selected_id == "custom":
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            custom["llm"] = st.selectbox("LLM", options=ollama_info["llm"] or ["qwen2.5:7b"])
        with c2:
            custom["embedding"] = st.selectbox("Embedding", options=ollama_info["embedding"] or ["nomic-embed-text"])
        with c3:
            custom["embedding_dim"] = st.selectbox("Dim embedding", [768, 1024, 1536], index=0)

    # Avertissement modèles manquants
    if selected_id and selected_id != "custom":
        chosen = next(c for c in couples_status if c["id"] == selected_id)
        if not chosen["available"]:
            st.warning("⚠️ Modèles à télécharger avant indexation :")
            if not chosen["llm_ok"]:
                st.code(f"ollama pull {chosen['llm']}")
            if not chosen["embed_ok"]:
                st.code(f"ollama pull {chosen['embedding']}")

    return selected_id, couples_status, custom


# ══════════════════════════════════════════════════════════════════════════════
# WIZARD — ÉTAPE 2 : Provider de query
# ══════════════════════════════════════════════════════════════════════════════

def wizard_step2_query(indexing_llm: str):
    st.markdown("### 2️⃣ Modèle de query — génération des réponses")
    st.caption("Ce modèle est appelé UNE SEULE fois par question, mais avec un contexte de 4-15k tokens.")

    selected_provider = st.session_state.get("wizard_query_provider", "mistral")
    api_key_input = ""

    for p in QUERY_PROVIDERS:
        label_extra = f" (= `{indexing_llm}`)" if p["id"] == "ollama_same" else ""
        col_sel, col_info = st.columns([1, 10])
        with col_sel:
            btn_type = "primary" if selected_provider == p["id"] else "secondary"
            if st.button("✓" if selected_provider == p["id"] else "○",
                         key=f"qp_{p['id']}", type=btn_type):
                st.session_state.wizard_query_provider = p["id"]
                st.rerun()
        with col_info:
            model_str = f"`{p['model']}`" if p["model"] else f"`{indexing_llm}`"
            st.markdown(f"**{p['label']}**{label_extra}  \n{model_str} — {p['note']}")

    # Clé Mistral si sélectionné
    if selected_provider == "mistral":
        st.markdown("---")
        st.markdown("**Clé API Mistral :**")
        col_key, col_test = st.columns([3, 1])
        with col_key:
            api_key_input = st.text_input(
                "MISTRAL_API_KEY",
                type="password",
                placeholder="Saisir ou laisser vide pour utiliser la variable d'environnement",
                value=os.getenv("MISTRAL_API_KEY", "")
            )
        with col_test:
            st.write("")
            if st.button("🔑 Tester", use_container_width=True):
                key_to_test = api_key_input or os.getenv("MISTRAL_API_KEY", "")
                if key_to_test:
                    with st.spinner("Test..."):
                        valid = test_mistral_key(key_to_test)
                    st.session_state.mistral_key_valid = valid
                    if valid:
                        st.success("✅ Clé valide")
                    else:
                        st.error("❌ Clé invalide ou réseau inaccessible")
                else:
                    st.warning("Saisissez une clé.")

        if st.session_state.mistral_key_valid is True:
            st.success("✅ Mistral API opérationnelle — query time ~2-5 secondes")
        elif st.session_state.mistral_key_valid is False:
            st.error("❌ Clé invalide — choisissez un provider Ollama en fallback")

    return selected_provider, api_key_input


# ══════════════════════════════════════════════════════════════════════════════
# WIZARD — ÉTAPE 3 : Profil de contexte
# ══════════════════════════════════════════════════════════════════════════════

def wizard_step3_context(query_provider: str):
    st.markdown("### 3️⃣ Profil de contexte injecté")
    st.caption("Contrôle le volume de tokens envoyés au LLM à chaque query. Impact direct sur la vitesse.")

    # Recommandation automatique
    if query_provider == "mistral":
        default_profile = "balanced"
        st.info("💡 Mistral API → **Balanced** recommandé (tokens peu coûteux, réponse rapide)")
    else:
        default_profile = "lean"
        st.info("💡 Ollama local → **Lean** recommandé (réduit le temps de génération de 3x)")

    selected_profile = st.radio(
        "Profil :",
        options=list(CONTEXT_PROFILES.keys()),
        format_func=lambda k: f"{CONTEXT_PROFILES[k]['label']} — {CONTEXT_PROFILES[k]['note']}",
        index=list(CONTEXT_PROFILES.keys()).index(default_profile),
        key="wizard_context_profile"
    )

    ctx = CONTEXT_PROFILES[selected_profile]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("top_k", ctx["top_k"])
    c2.metric("text_unit", f"{ctx['max_token_for_text_unit']}tk")
    c3.metric("global_ctx", f"{ctx['max_token_for_global_context']}tk")
    c4.metric("local_ctx",  f"{ctx['max_token_for_local_context']}tk")

    return selected_profile


# ══════════════════════════════════════════════════════════════════════════════
# WIZARD PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def show_setup_wizard():
    st.markdown("---")
    st.markdown("## ⚙️ Configuration initiale CaféIA")
    st.markdown("Sauvegardée dans `.cafeia_config.json` — ne sera plus demandée au prochain lancement.")
    st.markdown("---")

    with st.spinner("🔍 Détection des modèles Ollama..."):
        ollama_info = get_ollama_models()

    if "error" in ollama_info:
        st.error("❌ Ollama inaccessible — lancez `ollama serve`")
        st.stop()

    st.success(f"✅ Ollama — {len(ollama_info['all'])} modèle(s) détecté(s)")
    with st.expander("📋 Modèles disponibles"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**LLMs**")
            for m in ollama_info["llm"]: st.caption(f"• {m}")
        with c2:
            st.markdown("**Embedding**")
            for m in ollama_info["embedding"]: st.caption(f"• {m}")

    st.markdown("---")

    # Étape 1 — Indexation
    selected_id, couples_status, custom = wizard_step1_indexing(ollama_info)

    # Récupérer le LLM d'indexation pour afficher dans étape 2
    if selected_id and selected_id != "custom":
        chosen_couple = next(c for c in couples_status if c["id"] == selected_id)
        indexing_llm = chosen_couple["llm"]
    else:
        indexing_llm = custom.get("llm") or "qwen2.5:7b"

    st.markdown("---")

    # Étape 2 — Query
    selected_provider, api_key_input = wizard_step2_query(indexing_llm)

    st.markdown("---")

    # Étape 3 — Contexte
    selected_profile = wizard_step3_context(selected_provider)

    st.markdown("---")

    # ── Boutons Save / Skip ───────────────────────────────────────────────────
    col_save, col_skip = st.columns([3, 1])

    with col_save:
        can_save = selected_id is not None
        if selected_provider == "mistral":
            key_ok = (api_key_input or os.getenv("MISTRAL_API_KEY", "")) != ""
            if not key_ok:
                st.warning("⚠️ Renseignez une clé Mistral API pour utiliser ce provider.")
            can_save = can_save and key_ok

        if st.button("💾 Sauvegarder et démarrer", type="primary",
                     disabled=not can_save, use_container_width=True):

            # Construire la config
            if selected_id == "custom":
                llm_model  = custom["llm"]
                emb_model  = custom["embedding"]
                emb_dim    = custom["embedding_dim"]
                couple_id  = "custom"
            else:
                chosen = next(c for c in couples_status if c["id"] == selected_id)
                llm_model  = chosen["llm"]
                emb_model  = chosen["embedding"]
                emb_dim    = chosen["embedding_dim"]
                couple_id  = chosen["id"]

            # Query model
            provider_info = next(p for p in QUERY_PROVIDERS if p["id"] == selected_provider)
            query_model = provider_info["model"] or llm_model
            mistral_key = api_key_input if selected_provider == "mistral" else ""

            cfg = CafeiaConfig(
                llm_model=llm_model,
                embedding_model=emb_model,
                embedding_dim=emb_dim,
                couple_id=couple_id,
                query_provider=selected_provider,
                query_model=query_model,
                mistral_api_key=mistral_key,
                context_profile=selected_profile,
            )
            save_config(cfg)
            st.session_state.show_wizard = False
            st.success("✅ Configuration sauvegardée !")
            time.sleep(0.8)
            st.rerun()

    with col_skip:
        if st.button("⏭️ Défaut (7b + Mistral)", use_container_width=True):
            cfg = CafeiaConfig(
                llm_model="qwen2.5:7b", embedding_model="nomic-embed-text",
                embedding_dim=768, couple_id="demo",
                query_provider="ollama_same", query_model="qwen2.5:7b",
                context_profile="lean",
            )
            save_config(cfg)
            st.session_state.show_wizard = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config()

    if cfg is None or st.session_state.show_wizard:
        show_setup_wizard()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if os.path.exists("IMG/upvd_logo.png"):
            st.image("IMG/upvd_logo.png", width=150)
    with c2:
        st.markdown('<p class="main-header">☕ CaféIA - GraphRAG</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Interface de gestion documentaire et interrogation LLM avec Ollama</p>', unsafe_allow_html=True)
    with c3:
        if os.path.exists("IMG/mensaflow_logo.jpg"):
            st.image("IMG/mensaflow_logo.jpg", width=150)

    # ── Ollama check ──────────────────────────────────────────────────────────
    if not st.session_state.get('ollama_ok', False):
        try:
            import requests
            if requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200:
                st.session_state.ollama_ok = True
            else:
                st.error("Ollama ne répond pas"); st.stop()
        except Exception:
            st.error("❌ Ollama inaccessible sur localhost:11434"); st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Config active")

        couple_labels = {"qualite":"🥇 Qualité","equilibre":"🥈 Équilibre","demo":"🥉 Démo","custom":"⚙️ Custom"}
        query_labels  = {"mistral":"⚡ Mistral API","ollama_same":"🖥️ Ollama (same)","ollama_7b":"🖥️ Ollama 7b"}
        ctx           = cfg.get_context_params()

        st.info(f"""
**Indexation**
`{cfg.llm_model}`
Embedding: `{cfg.embedding_model}` dim={cfg.embedding_dim}

**Query**
{query_labels.get(cfg.query_provider,'?')}: `{cfg.get_query_model()}`

**Contexte**
{ctx['label']} — top_k={ctx['top_k']}

**Storage**
{'✅ Actif' if os.path.exists('./storage') else '❌ Vide'}
        """)

        if st.button("🔄 Reconfigurer", use_container_width=True):
            st.session_state.show_wizard = True
            st.rerun()

        with st.expander("📖 Modes de recherche"):
            st.markdown("""
| Mode | Multi-hop | Usage |
|------|-----------|-------|
| **naive** | ❌ | RAG classique |
| **local** | 1 hop | Entités proches |
| **global** | ✅ | Patterns globaux |
| **hybrid** | ✅✅ | **Recommandé** |
""")

        with st.expander("⚡ Vitesse estimée"):
            if cfg.query_provider == "mistral":
                st.success("Mistral API : ~2-5 sec")
            elif cfg.query_provider == "ollama_7b":
                st.warning("Ollama 7b : ~15-20 sec")
            else:
                st.warning(f"Ollama {cfg.llm_model} : 30-60 sec")
            ctx_label = CONTEXT_PROFILES[cfg.context_profile]["label"]
            st.caption(f"Contexte : {ctx_label}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Interroger le RAG", "📜 Historique"])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.header("📤 Importer des documents")
        st.info(f"Indexation via **`{cfg.llm_model}`** (local) — one-time cost")

        uploaded_files = st.file_uploader(
            "Glissez-déposez ou cliquez",
            type=['pdf','docx','xlsx','txt'],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.markdown(f"📁 **{len(uploaded_files)} fichier(s)**")
            for f in uploaded_files:
                st.caption(f"  • {f.name} ({f.size/1024:.1f} KB)")

            if st.button("🚀 Indexer", type="primary", use_container_width=True):
                progress = st.progress(0)
                status   = st.empty()
                try:
                    from document_processor import DocumentProcessor
                except ImportError:
                    DocumentProcessor = None

                for idx, uf in enumerate(uploaded_files):
                    status.text(f"⏳ {uf.name} ...")
                    try:
                        ext  = Path(uf.name).suffix
                        text = DocumentProcessor.process_uploaded_file(uf, ext) if DocumentProcessor else uf.read().decode("utf-8", errors="ignore")
                        if text and text.strip():
                            insert_document(text, cfg)
                            st.success(f"✅ {uf.name}")
                            st.session_state.uploaded_files_count += 1
                        else:
                            st.warning(f"⚠️ {uf.name} — texte vide")
                    except Exception as e:
                        st.error(f"❌ {uf.name} : {e}")
                    progress.progress((idx + 1) / len(uploaded_files))
                status.text("✨ Terminé !")
                st.balloons()

        with st.expander("ℹ️ Formats supportés"):
            st.markdown("PDF · DOCX · XLSX · TXT")

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.header("💬 Poser une question")

        # Badge query provider
        if cfg.query_provider == "mistral":
            st.success(f"⚡ Query via **Mistral API** (`{cfg.get_query_model()}`) — ~2-5 sec")
        else:
            st.warning(f"🖥️ Query via **Ollama** (`{cfg.get_query_model()}`) — 15-60 sec selon le modèle")

        question = st.text_area("Votre question :", height=100,
                                placeholder="Ex: Quel technicien certifié est disponible en Occitanie ?")

        col_mode, col_ctx, col_btn = st.columns([2, 2, 1])
        with col_mode:
            query_mode = st.selectbox("Mode", ['hybrid','naive','local','global'], index=0)
        with col_ctx:
            # Permettre override du profil à la volée
            profile_override = st.selectbox(
                "Contexte",
                options=list(CONTEXT_PROFILES.keys()),
                index=list(CONTEXT_PROFILES.keys()).index(cfg.context_profile),
                format_func=lambda k: CONTEXT_PROFILES[k]["label"]
            )
        with col_btn:
            st.write("")
            st.write("")
            search_btn = st.button("🔍 Rechercher", type="primary", use_container_width=True)

        if search_btn:
            if not question.strip():
                st.warning("⚠️ Saisissez une question.")
            else:
                # Override temporaire du profil
                original_profile = cfg.context_profile
                cfg.context_profile = profile_override

                STAGES = [
                    (0.1,  "🔢 Embedding de la requête..."),
                    (0.3,  "🕸️ Traversée du knowledge graph..."),
                    (0.5,  "📦 Assemblage du contexte..."),
                    (0.7,  "🤖 Génération LLM..."),
                    (0.9,  "✍️ Finalisation..."),
                ]

                progress_bar = st.progress(0)
                status_text  = st.empty()
                result_container = {"result": None, "error": None, "done": False}

                def _run():
                    try:
                        result_container["result"] = query_rag(question, query_mode, cfg)
                    except Exception as e:
                        result_container["error"] = str(e)
                    finally:
                        result_container["done"] = True

                thread = threading.Thread(target=_run)
                thread.start()

                stage_idx = 0
                while not result_container["done"]:
                    if stage_idx < len(STAGES):
                        pct, label = STAGES[stage_idx]
                        progress_bar.progress(pct)
                        status_text.markdown(f"**{label}**")
                        stage_idx += 1
                    time.sleep(3 if cfg.query_provider == "mistral" else 6)

                thread.join()
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()

                # Restaurer le profil original
                cfg.context_profile = original_profile

                if result_container["error"]:
                    st.error(f"❌ {result_container['error']}")
                else:
                    result = result_container["result"]
                    st.subheader("📝 Réponse")
                    st.markdown(result)

                    with st.expander("🔍 Détail du retrieval"):
                        mode_info = {
                            "naive":  "RAG classique, pas de graph",
                            "local":  "Entités proches — 1 hop",
                            "global": "Patterns transversaux du graph",
                            "hybrid": "Multi-hop — local + global",
                        }
                        ctx_used = CONTEXT_PROFILES[profile_override]
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Mode :** {mode_info.get(query_mode)}")
                            st.markdown(f"**Query model :** `{cfg.get_query_model()}`")
                        with col_b:
                            st.markdown(f"**Contexte :** {ctx_used['label']}")
                            st.markdown(f"**top_k={ctx_used['top_k']}** chunk_top_k={ctx_used['chunk_top_k']} | max_total={ctx_used['max_total_tokens']}tk")

                    st.session_state.query_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'question': question,
                        'mode': query_mode,
                        'context_profile': profile_override,
                        'query_model': cfg.get_query_model(),
                        'answer': result,
                    })

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.header("📜 Historique")

        if st.session_state.query_history:
            if st.button("🗑️ Effacer"):
                st.session_state.query_history = []
                st.rerun()

            for q in reversed(st.session_state.query_history):
                ctx_label = CONTEXT_PROFILES.get(q.get("context_profile","balanced"), {}).get("label","")
                with st.expander(f"🕐 {q['timestamp']} — [{q['mode']}] {q['question'][:60]}..."):
                    st.markdown(f"**Question :** {q['question']}")
                    st.caption(f"Query model: `{q.get('query_model','?')}` | Contexte: {ctx_label}")
                    st.markdown("---")
                    st.markdown(q['answer'])
        else:
            st.info("Aucune requête.")

    st.markdown("---")
    st.markdown('<p style="text-align:center;color:#888;">☕ CaféIA — Powered by LightRAG & Ollama</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()