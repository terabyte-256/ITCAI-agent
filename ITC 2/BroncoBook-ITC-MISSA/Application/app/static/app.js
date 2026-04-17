let conversationId = null;
let activeProvider = "ollama";
let loading = false;

const topicQuestions = {
  Admissions: "What are the freshman admission requirements?",
  "Financial Aid": "What financial aid resources are available?",
  "Health Center": "Where is Student Health Services located?",
  Academics: "How do I change my major?",
  Dining: "What dining options are available on campus?"
};

const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const searchButton = chatForm.querySelector(".search-button");
const topicButtons = Array.from(document.querySelectorAll(".topic-pill"));
const responsePanel = document.getElementById("responsePanel");
const responseTitle = document.getElementById("responseTitle");
const responseBody = document.getElementById("responseBody");
const sourcesEl = document.getElementById("sources");

async function loadHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }
    const data = await response.json();
    activeProvider = data.provider_default || "ollama";
  } catch {
    activeProvider = "ollama";
  }
}

function setLoading(nextLoading) {
  loading = nextLoading;
  searchButton.disabled = nextLoading;
  messageInput.disabled = nextLoading;
  topicButtons.forEach((button) => {
    button.disabled = nextLoading;
  });
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  if (!Array.isArray(sources) || sources.length === 0) {
    sourcesEl.hidden = true;
    return;
  }

  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "Sources";
  sourcesEl.appendChild(title);

  sources.forEach((source) => {
    const card = document.createElement("a");
    card.className = "source-card";
    card.href = source.source_url || "#";
    card.target = "_blank";
    card.rel = "noreferrer";
    card.innerHTML = `
      <strong>${source.title || "Campus Source"}</strong>
      <span>${source.section || source.markdown_file || ""}</span>
      <p>${source.snippet || ""}</p>
    `;
    sourcesEl.appendChild(card);
  });

  sourcesEl.hidden = false;
}

function renderResponse(title, body, sources = []) {
  responsePanel.hidden = false;
  responseTitle.textContent = title;
  responseBody.textContent = body;
  renderSources(sources);
}

async function sendMessage(messageText) {
  const message = (messageText || "").trim();
  if (!message || loading) {
    return;
  }

  setLoading(true);
  renderResponse("BroncoBook Answer", "Working on your answer...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId,
        message,
        provider: activeProvider
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || "Request failed.");
    }

    conversationId = data.conversation_id || conversationId;
    renderResponse(
      "BroncoBook Answer",
      data.answer || "I could not find that information in the indexed corpus.",
      data.sources || []
    );
  } catch (error) {
    const messageTextOut = error instanceof Error ? error.message : "Unknown error";
    renderResponse("BroncoBook Error", `Error: ${messageTextOut}`);
  } finally {
    setLoading(false);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage(messageInput.value);
});

topicButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const label = (button.textContent || "").trim();
    const question = topicQuestions[label] || label;
    messageInput.value = question;
    messageInput.focus();
  });
});

void loadHealth();
