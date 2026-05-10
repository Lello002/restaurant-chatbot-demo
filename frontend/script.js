const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const chatMessages = document.getElementById("chatMessages");
const clearChatBtn = document.getElementById("clearChatBtn");
const quickButtons = document.querySelectorAll(".quick-btn");

const restaurantLogo = document.getElementById("restaurantLogo");
const brandPill = document.getElementById("brandPill");
const restaurantMeta = document.getElementById("restaurantMeta");
const heroTitle = document.getElementById("heroTitle");
const heroSubtitle = document.getElementById("heroSubtitle");
const callButton = document.getElementById("callButton");
const mapsButton = document.getElementById("mapsButton");
const chatAssistantName = document.getElementById("chatAssistantName");

const imageModal = document.getElementById("imageModal");
const modalImage = document.getElementById("modalImage");
const closeImageModal = document.getElementById("closeImageModal");

let history = [];

let restaurantConfig = {
  restaurant_name: "Ristorante",
  assistant_name: "Assistente AI",
  logo_type: "emoji",
  logo_emoji: "🍽️",
  logo_image: "",
  hero_title: "Benvenuto",
  hero_subtitle: "Scopri il menu, chiedi consigli sui piatti e trova subito ciò che fa per te.",
  phone: "",
  address: "",
  google_maps_url: ""
};

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function openImageModal(src) {
  if (!src) return;

  modalImage.src = src;
  imageModal.classList.remove("hidden");
}

function closeModal() {
  imageModal.classList.add("hidden");
  modalImage.src = "";
}

closeImageModal.addEventListener("click", closeModal);

imageModal.addEventListener("click", (event) => {
  if (event.target === imageModal) {
    closeModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
  }
});

async function loadRestaurantConfig() {
  try {
    const response = await fetch("/api/config", {
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error("Impossibile caricare configurazione ristorante");
    }

    restaurantConfig = await response.json();
    applyRestaurantConfig();

  } catch (error) {
    console.warn("Configurazione non caricata:", error);
    applyRestaurantConfig();
  }
}

function applyRestaurantConfig() {
  const name = restaurantConfig.restaurant_name || "Ristorante";
  const assistantName = restaurantConfig.assistant_name || `Assistente AI di ${name}`;

  document.title = `${name} - Assistente AI Menu`;

  if (restaurantConfig.logo_type === "image" && restaurantConfig.logo_image) {
    restaurantLogo.innerHTML = `<img src="${restaurantConfig.logo_image}" alt="Logo ${name}">`;
  } else {
    restaurantLogo.textContent = restaurantConfig.logo_emoji || "🍽️";
  }

  brandPill.textContent = `Assistente AI di ${name}`;
  restaurantMeta.textContent = restaurantConfig.address || "Menu digitale intelligente";

  heroTitle.textContent = restaurantConfig.hero_title || `Benvenuto da ${name}`;
  heroSubtitle.textContent = restaurantConfig.hero_subtitle || "Chiedi consigli sui piatti e scopri il menu.";

  chatAssistantName.textContent = assistantName;

  const cleanPhone = (restaurantConfig.phone || "").replace(/[^\d+]/g, "");

  if (cleanPhone) {
    callButton.href = `tel:${cleanPhone}`;
    callButton.classList.remove("disabled");
    callButton.setAttribute("aria-label", `Chiama ${name}`);
  } else {
    callButton.href = "#";
    callButton.classList.add("disabled");
  }

  const mapsUrl = restaurantConfig.google_maps_url || "";

  if (mapsUrl.startsWith("http")) {
    mapsButton.href = mapsUrl;
    mapsButton.target = "_blank";
    mapsButton.rel = "noopener noreferrer";
    mapsButton.classList.remove("disabled");
    mapsButton.setAttribute("aria-label", `Apri mappa di ${name}`);
  } else {
    mapsButton.href = "#";
    mapsButton.classList.add("disabled");
  }

  resetWelcomeMessage();
}

function resetWelcomeMessage() {
  const name = restaurantConfig.restaurant_name || "il ristorante";

  chatMessages.innerHTML = `
    <div class="message assistant">
      <div class="avatar">🍽️</div>
      <div class="bubble">
        Ciao! Sono l’assistente di ${name}. Dimmi cosa ti va e ti aiuto a scegliere dal menu.
      </div>
    </div>
  `;
}

function addMessage(role, content, isTyping = false) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "👤" : "🍽️";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (isTyping) {
    bubble.classList.add("typing");
  }

  bubble.textContent = content;

  message.appendChild(avatar);
  message.appendChild(bubble);

  chatMessages.appendChild(message);
  scrollToBottom();

  return message;
}

function formatPrice(price) {
  if (price === null || price === undefined || price === "") {
    return "";
  }

  return `${Number(price).toFixed(2).replace(".", ",")} €`;
}

function addDishCardsMessage(dishes) {
  if (!dishes || dishes.length === 0) {
    return;
  }

  const message = document.createElement("div");
  message.className = "message assistant dish-message";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🍽️";

  const cardsWrapper = document.createElement("div");
  cardsWrapper.className = "dish-cards-wrapper";

  dishes.forEach((dish) => {
    const card = document.createElement("div");
    card.className = "dish-card";

    const hasImage = dish.image && dish.image.trim() !== "";

    const media = document.createElement("div");
    media.className = "dish-media";

    if (hasImage) {
      const img = document.createElement("img");
      img.src = dish.image;
      img.alt = dish.name;
      img.addEventListener("click", () => openImageModal(dish.image));
      media.appendChild(img);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "dish-placeholder";
      placeholder.textContent = "🍽️";
      media.appendChild(placeholder);
    }

    const body = document.createElement("div");
    body.className = "dish-card-body";

    const allergens = dish.allergens && dish.allergens.length > 0
      ? dish.allergens.join(", ")
      : "non indicati";

    const tags = dish.diet_tags && dish.diet_tags.length > 0
      ? dish.diet_tags.map(tag => `<span>${tag}</span>`).join("")
      : "";

    body.innerHTML = `
      <div class="dish-card-top">
        <small>${dish.category || "Menu"}</small>
        <strong>${formatPrice(dish.price)}</strong>
      </div>

      <h3>${dish.name}</h3>

      <p>${dish.description || ""}</p>

      <div class="dish-tags">
        ${tags}
      </div>

      <div class="dish-allergens">
        Allergeni: ${allergens}
      </div>
    `;

    card.appendChild(media);
    card.appendChild(body);
    cardsWrapper.appendChild(card);
  });

  message.appendChild(avatar);
  message.appendChild(cardsWrapper);

  chatMessages.appendChild(message);
  scrollToBottom();
}

async function sendMessage(text) {
  const message = text.trim();

  if (!message) return;

  if (message.length > 500) {
    addMessage("assistant", "Il messaggio è troppo lungo. Scrivi una domanda più breve sul menu o sul ristorante.");
    return;
  }

  addMessage("user", message);
  history.push({ role: "user", content: message });

  messageInput.value = "";
  messageInput.disabled = true;

  const submitButton = chatForm.querySelector("button");
  submitButton.disabled = true;

  const typingMessage = addMessage("assistant", "Sto consultando il menu...", true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message,
        history
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Errore sconosciuto");
    }

    const data = await response.json();

    typingMessage.remove();

    addMessage("assistant", data.answer);
    history.push({ role: "assistant", content: data.answer });

    addDishCardsMessage(data.dishes);

  } catch (error) {
    typingMessage.remove();

    addMessage(
      "assistant",
      "Mi dispiace, c'è stato un problema tecnico: " + error.message
    );

  } finally {
    messageInput.disabled = false;
    submitButton.disabled = false;
    messageInput.focus();
    scrollToBottom();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

quickButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const message = button.dataset.message;
    sendMessage(message);
  });
});

clearChatBtn.addEventListener("click", () => {
  history = [];
  resetWelcomeMessage();
  closeModal();
  messageInput.focus();
});

loadRestaurantConfig();