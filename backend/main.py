import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from rag_store import RagStore


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ASSETS_DIR = BASE_DIR / "static_assets"
CONFIG_PATH = BASE_DIR / "data" / "restaurant_config.json"

load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

if not OPENAI_API_KEY or OPENAI_API_KEY == "la_tua_chiave":
    raise RuntimeError("OPENAI_API_KEY non valida. Controlla backend/.env")

client = OpenAI(api_key=OPENAI_API_KEY)

rag = RagStore()
rag_ready = False

app = FastAPI(title="Restaurant Chatbot Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

@app.on_event("startup")
def startup_event():
    global rag_ready

    print("[STARTUP] Caricamento indice RAG...")
    rag.load()
    rag_ready = True
    print(f"[STARTUP] RAG pronto. Chunks caricati: {len(rag.chunks)}")

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class DishCard(BaseModel):
    name: str
    category: str
    price: Optional[float] = None
    description: str
    image: str = ""
    allergens: List[str] = []
    diet_tags: List[str] = []


class ChatResponse(BaseModel):
    answer: str
    images: List[str]
    dishes: List[DishCard]
    sources: List[Dict[str, Any]]


def load_restaurant_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {
            "restaurant_name": "Ristorante Demo",
            "assistant_name": "Assistente Menu",
            "logo_type": "emoji",
            "logo_emoji": "🍽️",
            "logo_image": "",
            "hero_title": "Il menu diventa una conversazione.",
            "hero_subtitle": "Chiedi consigli, ingredienti, prezzi e piatti disponibili.",
            "address": "",
            "phone": "",
            "google_maps_url": "",
            "opening_hours": {},
            "notes": []
        }

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[FONTE {i}]\n"
            f"Tipo: {chunk.get('type', '')}\n"
            f"Categoria: {chunk.get('category', '')}\n"
            f"Nome: {chunk.get('name', '')}\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(parts)


def collect_images(chunks: List[Dict[str, Any]]) -> List[str]:
    images = []

    for chunk in chunks:
        for img in chunk.get("images", []):
            if img and img not in images:
                images.append(img)

    return images[:6]


def is_message_too_long(user_message: str) -> bool:
    return len(user_message.strip()) > 500


def is_clearly_out_of_scope(user_message: str) -> bool:
    message = user_message.lower().strip()

    if not message:
        return False

    greetings = [
        "ciao",
        "buongiorno",
        "buonasera",
        "salve",
        "hey",
        "ei"
    ]

    if message in greetings:
        return False

    restaurant_keywords = [
        "menu",
        "menù",
        "piatto",
        "piatti",
        "mangiare",
        "mangio",
        "consigli",
        "consiglio",
        "consigliami",
        "prendere",
        "prendo",
        "ordinare",
        "ordino",
        "antipasto",
        "antipasti",
        "primo",
        "primi",
        "pasta",
        "secondo",
        "secondi",
        "dolce",
        "dolci",
        "dessert",
        "bevanda",
        "bevande",
        "bere",
        "bibita",
        "bibite",
        "vegetariano",
        "vegetariani",
        "vegano",
        "vegani",
        "glutine",
        "lattosio",
        "allergene",
        "allergeni",
        "ingredienti",
        "contiene",
        "senza",
        "economico",
        "spendere",
        "costa",
        "costano",
        "prezzo",
        "prezzi",
        "orario",
        "orari",
        "aperto",
        "aperti",
        "aperta",
        "aperte",
        "chiuso",
        "chiusi",
        "chiusa",
        "chiuse",
        "telefono",
        "numero",
        "indirizzo",
        "dove siete",
        "dove si trova",
        "come arrivare",
        "prenotare",
        "prenotazione",
        "prenotazioni",
        "contatto",
        "contatti",
        "chiamare",
        "bruschetta",
        "tartare",
        "salmone",
        "risotto",
        "funghi",
        "carbonara",
        "spaghetti",
        "pomodoro",
        "tagliata",
        "manzo",
        "cheesecake",
        "limonata"
    ]

    if any(keyword in message for keyword in restaurant_keywords):
        return False

    blocked_keywords = [
        "python",
        "javascript",
        "codice",
        "programma",
        "programmare",
        "compito",
        "tesi",
        "tema",
        "poesia",
        "traduci",
        "riassumi",
        "politica",
        "bitcoin",
        "crypto",
        "meteo",
        "calcolo",
        "matematica",
        "equazione",
        "film",
        "serie tv",
        "calcio",
        "guerra",
        "email",
        "scrivimi",
        "curriculum",
        "jailbreak",
        "prompt",
        "chatgpt"
    ]

    if any(keyword in message for keyword in blocked_keywords):
        return True

    # Se il messaggio è lungo e non contiene parole da ristorante, lo blocchiamo.
    if len(message) > 80:
        return True

    return False


def should_show_dish_cards(user_message: str) -> bool:
    message = user_message.lower()

    no_card_keywords = [
        "orario",
        "orari",
        "aperto",
        "aperti",
        "aperta",
        "aperte",
        "chiuso",
        "chiusi",
        "chiusa",
        "chiuse",
        "telefono",
        "numero",
        "indirizzo",
        "dove siete",
        "dove si trova",
        "come arrivare",
        "prenotare",
        "prenotazione",
        "prenotazioni",
        "contatto",
        "contatti",
        "chiamare",
        "chiamo"
    ]

    if any(keyword in message for keyword in no_card_keywords):
        return False

    card_keywords = [
        "piatto",
        "piatti",
        "mangiare",
        "mangio",
        "consigli",
        "consiglio",
        "consigliami",
        "prendere",
        "prendo",
        "ordinare",
        "ordino",
        "menu",
        "menù",
        "antipasto",
        "antipasti",
        "primo",
        "primi",
        "pasta",
        "secondo",
        "secondi",
        "dolce",
        "dolci",
        "dessert",
        "bevanda",
        "bevande",
        "bere",
        "bibita",
        "bibite",
        "vegetariano",
        "vegetariani",
        "vegano",
        "vegani",
        "glutine",
        "lattosio",
        "allergene",
        "allergeni",
        "ingredienti",
        "contiene",
        "senza",
        "economico",
        "spendere",
        "costa",
        "costano",
        "prezzo",
        "prezzi",
        "bruschetta",
        "tartare",
        "salmone",
        "risotto",
        "funghi",
        "carbonara",
        "spaghetti",
        "pomodoro",
        "tagliata",
        "manzo",
        "cheesecake",
        "limonata"
    ]

    return any(keyword in message for keyword in card_keywords)


def is_specific_dish_question(user_message: str, dish_name: str) -> bool:
    message = user_message.lower()
    name = dish_name.lower()

    if name in message:
        return True

    meaningful_words = [
        word
        for word in name.replace("'", " ").replace("-", " ").split()
        if len(word) >= 5
    ]

    matches = [word for word in meaningful_words if word in message]

    very_specific_words = [
        "bruschetta",
        "tartare",
        "salmone",
        "risotto",
        "funghi",
        "carbonara",
        "spaghetti",
        "pomodoro",
        "tagliata",
        "manzo",
        "cheesecake",
        "limonata"
    ]

    if any(word in message and word in name for word in very_specific_words):
        return True

    return len(matches) >= 2


def collect_dish_cards(
    user_message: str,
    chunks: List[Dict[str, Any]]
) -> List[DishCard]:
    if not should_show_dish_cards(user_message):
        return []

    message = user_message.lower()

    category_map = {
        "antipasto": "Antipasti",
        "antipasti": "Antipasti",
        "primo": "Primi",
        "primi": "Primi",
        "pasta": "Primi",
        "secondo": "Secondi",
        "secondi": "Secondi",
        "dolce": "Dolci",
        "dolci": "Dolci",
        "dessert": "Dolci",
        "bevanda": "Bevande",
        "bevande": "Bevande",
        "bere": "Bevande",
        "bibita": "Bevande",
        "bibite": "Bevande"
    }

    requested_category = None

    for keyword, category_name in category_map.items():
        if keyword in message:
            requested_category = category_name
            break

    exact_matches: List[DishCard] = []
    category_matches: List[DishCard] = []
    other_matches: List[DishCard] = []

    for chunk in chunks:
        if chunk.get("type") != "dish":
            continue

        metadata = chunk.get("metadata", {})

        name = metadata.get("name", chunk.get("name", ""))
        category = chunk.get("category", "")

        dish = DishCard(
            name=name,
            category=category,
            price=metadata.get("price"),
            description=metadata.get("description", ""),
            image=metadata.get("image", ""),
            allergens=metadata.get("allergens", []),
            diet_tags=metadata.get("diet_tags", [])
        )

        if not dish.name:
            continue

        already_added = any(
            existing.name == dish.name
            for existing in exact_matches + category_matches + other_matches
        )

        if already_added:
            continue

        if is_specific_dish_question(user_message, dish.name):
            exact_matches.append(dish)
            continue

        if requested_category and category == requested_category:
            category_matches.append(dish)
            continue

        other_matches.append(dish)

    # Priorità 1: se chiede un piatto preciso, mostra solo quel piatto.
    if exact_matches:
        return exact_matches[:1]

    # Priorità 2: se chiede una categoria precisa, mostra solo piatti di quella categoria.
    if requested_category:
        return category_matches[:5]

    multi_suggestion_keywords = [
        "consigli",
        "consiglio",
        "consigliami",
        "cosa",
        "piatti",
        "mangiare",
        "menu",
        "menù",
        "vegetariano",
        "vegetariani",
        "vegano",
        "vegani",
        "senza glutine",
        "senza lattosio",
        "allergeni",
        "massimo",
        "meno di",
        "sotto",
        "economico",
        "spendere",
        "completo"
    ]

    if any(keyword in message for keyword in multi_suggestion_keywords):
        return other_matches[:5]

    return other_matches[:3]


def ask_model(user_message: str, context: str, history: List[Dict[str, str]]) -> str:
    history_text = ""

    for msg in history[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in {"user", "assistant"} and content:
            history_text += f"{role.upper()}: {content}\n"

    restaurant_config = load_restaurant_config()
    restaurant_name = restaurant_config.get("restaurant_name", "il ristorante")

    system_prompt = f"""
Sei un assistente virtuale per il ristorante {restaurant_name}.

Regole fondamentali:
- Rispondi sempre in italiano.
- Usa SOLO le informazioni presenti nel menu recuperato.
- Se una cosa non è presente nel menu, dillo chiaramente.
- Non inventare ingredienti, prezzi, allergeni, disponibilità o immagini.
- Sii gentile, naturale e professionale.
- Aiuta il cliente a scegliere piatti, bevande, alternative e informazioni dal menu.
- Se il cliente chiede consigli, suggerisci solo piatti presenti nel menu recuperato.
- Se nel menu ci sono prezzi, riportali.
- Se il cliente chiede allergeni e non sono indicati, rispondi che non sono specificati e consiglia di chiedere al personale.
- Non dire "secondo il contesto" o "dalle fonti". Parla come un cameriere digitale.
- Se suggerisci più piatti, organizzali in modo leggibile.
- Non promettere prenotazioni, ordini o disponibilità in tempo reale se non sono presenti nel menu.
- Tu non prendi prenotazioni: per prenotare il cliente deve chiamare il ristorante.
- Non essere troppo discorsivo in maniera inutile. Rispondi solo alla domanda.
- Se ti chiedono gli orari, rispondi solo sugli orari.
- Se ti chiedono telefono, indirizzo o informazioni del ristorante, rispondi solo a quello.
- Se ti chiedono un piatto specifico, parla solo di quel piatto.
- Se la domanda non riguarda il ristorante, il menu, i piatti, gli orari, i contatti o le prenotazioni, rispondi brevemente che puoi aiutare solo con informazioni sul ristorante.
"""

    user_prompt = f"""
Conversazione recente:
{history_text}

Menu recuperato:
{context}

Domanda del cliente:
{user_message}

Rispondi in modo utile, elegante, chiaro e commerciale.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
    )

    return response.output_text


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()

    if not rag_ready:
        return ChatResponse(
            answer="Sto completando l'avvio del menu digitale. Riprova tra qualche secondo.",
            images=[],
            dishes=[],
            sources=[]
    )

    if not message:
        raise HTTPException(status_code=400, detail="Messaggio vuoto.")

    if is_message_too_long(message):
        return ChatResponse(
            answer="Il messaggio è un po’ troppo lungo. Posso aiutarti con domande brevi sul menu, sui piatti, sugli orari o sui contatti del ristorante.",
            images=[],
            dishes=[],
            sources=[]
        )

    if is_clearly_out_of_scope(message):
        return ChatResponse(
            answer="Posso aiutarti solo con informazioni sul menu, sui piatti, sugli allergeni, sugli orari e sui contatti del ristorante.",
            images=[],
            dishes=[],
            sources=[]
        )

    retrieved_chunks = rag.search(message, top_k=7)

    context = build_context(retrieved_chunks)
    images = collect_images(retrieved_chunks)
    dishes = collect_dish_cards(message, retrieved_chunks)

    try:
        answer = ask_model(message, context, req.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore OpenAI: {str(e)}")

    sources = [
        {
            "type": chunk.get("type", ""),
            "category": chunk.get("category", ""),
            "name": chunk.get("name", ""),
            "score": round(chunk["score"], 4),
            "preview": chunk["text"][:250] + "..."
        }
        for chunk in retrieved_chunks
    ]

    return ChatResponse(
        answer=answer,
        images=images,
        dishes=dishes,
        sources=sources
    )


@app.get("/api/config")
def get_config():
    config = load_restaurant_config()

    public_config = {
        "restaurant_name": config.get("restaurant_name", "Ristorante Demo"),
        "assistant_name": config.get("assistant_name", "Assistente Menu"),
        "logo_type": config.get("logo_type", "emoji"),
        "logo_emoji": config.get("logo_emoji", "🍽️"),
        "logo_image": config.get("logo_image", ""),
        "hero_title": config.get("hero_title", "Il menu diventa una conversazione."),
        "hero_subtitle": config.get("hero_subtitle", "Chiedi consigli, ingredienti, prezzi e piatti disponibili."),
        "address": config.get("address", ""),
        "phone": config.get("phone", ""),
        "google_maps_url": config.get("google_maps_url", ""),
        "opening_hours": config.get("opening_hours", {}),
        "notes": config.get("notes", [])
    }

    return public_config


@app.get("/api/menu")
def get_menu():
    return {
        "items": [
            {
                "type": chunk.get("type", ""),
                "category": chunk.get("category", ""),
                "name": chunk.get("name", ""),
                "price": chunk.get("price"),
                "image": chunk.get("image", ""),
                "metadata": chunk.get("metadata", {})
            }
            for chunk in rag.chunks
            if chunk.get("type") == "dish"
        ]
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": OPENAI_MODEL,
        "rag_ready": rag_ready,
        "rag_chunks": len(rag.chunks)
    }


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")