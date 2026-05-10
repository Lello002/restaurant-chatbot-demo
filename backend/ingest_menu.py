import json
from pathlib import Path
from typing import List, Dict, Any

import fitz
import numpy as np

from rag_store import RagStore, STORAGE_DIR, CHUNKS_PATH, EMBEDDINGS_PATH


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "menu.pdf"
IMAGES_DIR = BASE_DIR / "static_assets" / "menu_images"


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def split_text(text: str, max_chars: int = 1200, overlap: int = 250) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]

        last_newline = chunk.rfind("\n")
        if last_newline > 300:
            chunk = chunk[:last_newline]
            end = start + last_newline

        chunks.append(chunk.strip())
        start = max(0, end - overlap)

        if start >= len(text):
            break

    return [c for c in chunks if c.strip()]


def extract_images_from_page(doc: fitz.Document, page_index: int) -> List[str]:
    page = doc[page_index]
    image_paths = []

    images = page.get_images(full=True)

    for img_number, img in enumerate(images, start=1):
        xref = img[0]

        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"page_{page_index + 1}_img_{img_number}.{image_ext}"
            image_path = IMAGES_DIR / image_name

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(f"/assets/menu_images/{image_name}")

        except Exception as e:
            print(f"[WARN] Impossibile estrarre immagine pagina {page_index + 1}: {e}")

    return image_paths


def build_chunks_from_pdf() -> List[Dict[str, Any]]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF non trovato: {PDF_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    chunks: List[Dict[str, Any]] = []

    print(f"[INFO] PDF caricato: {PDF_PATH}")
    print(f"[INFO] Numero pagine: {len(doc)}")

    for page_index in range(len(doc)):
        page = doc[page_index]
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        image_paths = extract_images_from_page(doc, page_index)

        if not text:
            print(f"[WARN] Pagina {page_index + 1} senza testo estraibile.")
            continue

        page_chunks = split_text(text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "id": f"page_{page_index + 1}_chunk_{chunk_index + 1}",
                "page": page_index + 1,
                "text": chunk_text,
                "images": image_paths
            })

    doc.close()

    print(f"[INFO] Chunks creati: {len(chunks)}")
    return chunks


def main():
    chunks = build_chunks_from_pdf()

    if not chunks:
        raise RuntimeError(
            "Nessun testo estratto dal PDF. Il PDF potrebbe essere scannerizzato come immagine."
        )

    texts = [chunk["text"] for chunk in chunks]

    rag = RagStore()
    embeddings = rag.embed_texts(texts)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(EMBEDDINGS_PATH, embeddings)

    print("[OK] Indicizzazione completata.")
    print(f"[OK] Chunks salvati in: {CHUNKS_PATH}")
    print(f"[OK] Embeddings salvati in: {EMBEDDINGS_PATH}")
    print(f"[OK] Immagini salvate in: {IMAGES_DIR}")


if __name__ == "__main__":
    main()