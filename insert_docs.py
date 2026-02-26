import asyncio
from main import initialize_rag

async def insert_documents():
    rag = await initialize_rag()
    
    # Insertion d'un fichier texte
    with open("data/documents/greenpower_products.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    await rag.ainsert(content)
    print("Document indexé - knowledge graph construit")

    # Insertion de plusieurs fichiers
    import os
    doc_dir = "data/documents/"
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
            await rag.ainsert(content)
            print(f"Indexé : {filename}")

if __name__ == "__main__":
    asyncio.run(insert_documents())