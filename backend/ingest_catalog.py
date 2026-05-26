import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from rag_store import RagStore, STORAGE_DIR, CHUNKS_PATH, EMBEDDINGS_PATH


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "menu_catalog.json"
CONFIG_PATH = BASE_DIR / "data" / "restaurant_config.json"

load_dotenv(BASE_DIR / ".env", override=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def item_to_searchable_text(category_name: str, item: Dict[str, Any]) -> str:
    ingredients = ", ".join(item.get("ingredients", [])) or "non specificati"
    allergens = ", ".join(item.get("allergens", [])) or "nessun allergene indicato"
    diet_tags = ", ".join(item.get("diet_tags", [])) or "nessuna caratteristica specificata"

    price = item.get("price")
    price_text = f"{price:.2f} euro" if isinstance(price, (int, float)) else "prezzo non indicato"

    spicy_text = "piccante" if item.get("spicy") else "non piccante"
    recommended_text = "piatto consigliato" if item.get("recommended") else "piatto non segnato come consigliato"

    return f"""
Categoria: {category_name}
Nome piatto: {item.get("name", "")}
Prezzo: {price_text}
Descrizione: {item.get("description", "")}
Ingredienti: {ingredients}
Allergeni: {allergens}
Caratteristiche alimentari: {diet_tags}
Piccantezza: {spicy_text}
Consigliato: {recommended_text}
""".strip()


def build_chunks_from_catalog() -> List[Dict[str, Any]]:
    catalog = load_json(CATALOG_PATH)
    config = load_json(CONFIG_PATH)

    chunks: List[Dict[str, Any]] = []

    restaurant_text = f"""
Nome ristorante: {config.get("restaurant_name", "")}
Indirizzo: {config.get("address", "")}
Telefono: {config.get("phone", "")}
Google Maps: {config.get("google_maps_url", "")}
Orari: {json.dumps(config.get("opening_hours", {}), ensure_ascii=False)}
Note: {" ".join(config.get("notes", []))}
""".strip()

    chunks.append({
        "id": "restaurant-info",
        "type": "restaurant_info",
        "category": "Informazioni",
        "item_id": "",
        "name": config.get("restaurant_name", "Ristorante"),
        "price": None,
        "text": restaurant_text,
        "image": "",
        "images": [],
        "metadata": config
    })

    for category in catalog.get("categories", []):
        category_name = category.get("name", "Senza categoria")

        for item in category.get("items", []):
            text = item_to_searchable_text(category_name, item)
            image = item.get("image", "")

            chunks.append({
                "id": item.get("id", item.get("name", "")),
                "type": "dish",
                "category": category_name,
                "item_id": item.get("id", ""),
                "name": item.get("name", ""),
                "price": item.get("price"),
                "text": text,
                "image": image,
                "images": [image] if image else [],
                "metadata": item
            })

    return chunks


def main():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks_from_catalog()

    if len(chunks) <= 1:
        raise RuntimeError("Catalogo vuoto. Controlla backend/data/menu_catalog.json")

    texts = [chunk["text"] for chunk in chunks]

    print("[INFO] Creo embeddings con OpenAI...")
    print(f"[INFO] Elementi da indicizzare: {len(texts)}")

    rag = RagStore()
    embeddings = rag.embed_texts(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError("Numero embeddings diverso dal numero chunks.")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print("[OK] Catalogo indicizzato con OpenAI embeddings.")
    print(f"[OK] Chunks salvati in: {CHUNKS_PATH}")
    print(f"[OK] Embeddings salvati in: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()