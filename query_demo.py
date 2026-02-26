import asyncio
from main import initialize_rag
from lightrag import QueryParam

async def demo_queries():
    rag = await initialize_rag()
    
    question = "Quel technicien certifié est disponible en Occitanie pour installer le SolarMax 3000 ?"
    
    # Mode naive : RAG classique, pas de graph
    result_naive = await rag.aquery(
        question,
        param=QueryParam(mode="naive")
    )
    
    # Mode local : entités et relations proches (1 hop)
    result_local = await rag.aquery(
        question,
        param=QueryParam(mode="local")
    )
    
    # Mode global : patterns globaux du knowledge graph
    result_global = await rag.aquery(
        question,
        param=QueryParam(mode="global")
    )
    
    # Mode hybrid : local + global combinés (RECOMMANDÉ pour multi-hop)
    result_hybrid = await rag.aquery(
        question,
        param=QueryParam(mode="hybrid")
    )
    
    print("=== NAIVE ===")
    print(result_naive)
    print("\n=== HYBRID (multi-hop) ===")
    print(result_hybrid)

if __name__ == "__main__":
    asyncio.run(demo_queries())