let conversationId = null;

const starterQuestions = [
  'What are the admission requirements?',
  'When is the application deadline?',
  'What campus housing options are available?',
  'How can I contact financial aid?'
];

const state = {
  totalQueries: 0,
  unanswered: 0,
  toolCalls: 0,
  totalSources: 0
};

const messagesEl = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const starterQuestionsEl = document.getElementById('starterQuestions');
const messageTemplate = document.getElementById('messageTemplate');
const providerSelect = document.getElementById('providerSelect');
const modelInput = document.getElementById('modelInput');

const metricQueries = document.getElementById('metricQueries');
const metricUnanswered = document.getElementById('metricUnanswered');
const metricTools = document.getElementById('metricTools');
const metricSources = document.getElementById('metricSources');
const healthStatus = document.getElementById('healthStatus');

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderMetrics() {
  if (!metricQueries || !metricUnanswered || !metricTools || !metricSources) return;
  metricQueries.textContent = state.totalQueries;
  metricUnanswered.textContent = state.unanswered;
  metricTools.textContent = state.toolCalls;
  metricSources.textContent = state.totalQueries
    ? (state.totalSources / state.totalQueries).toFixed(1)
    : '0';
}

function addMessage(role, body, sources = []) {
  const node = messageTemplate.content.cloneNode(true);
  node.querySelector('.message-role').textContent = role;
  node.querySelector('.message-body').textContent = body;

  const sourcesEl = node.querySelector('.sources');
  if (sources.length) {
    const title = document.createElement('div');
    title.className = 'sources-title';
    title.textContent = 'Sources';
    sourcesEl.appendChild(title);

    sources.forEach((source) => {
      const card = document.createElement('a');
      card.className = 'source-card';
      card.href = source.source_url;
      card.target = '_blank';
      card.rel = 'noreferrer';
      card.innerHTML = `
        <strong>${escapeHtml(source.title)}</strong>
        <span>${escapeHtml(source.section || source.markdown_file)}</span>
        <p>${escapeHtml(source.snippet)}</p>
      `;
      sourcesEl.appendChild(card);
    });
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  const button = chatForm.querySelector('button');
  button.disabled = isLoading;
  button.textContent = isLoading ? 'Sending...' : 'Send';
  messageInput.disabled = isLoading;
  if (providerSelect) providerSelect.disabled = isLoading;
  if (modelInput) modelInput.disabled = isLoading;
}

function renderStarterQuestions() {
  if (!starterQuestionsEl) return;
  starterQuestionsEl.innerHTML = '';
  starterQuestions.forEach((question) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = question;
    chip.addEventListener('click', () => {
      messageInput.value = question;
      messageInput.focus();
    });
    starterQuestionsEl.appendChild(chip);
  });
}

async function loadStarters() {
  try {
    const response = await fetch('/api/starter-questions');
    if (!response.ok) return;
    const data = await response.json();
    if (!Array.isArray(data.questions) || !starterQuestionsEl) return;
    starterQuestionsEl.innerHTML = '';
    data.questions.forEach((question) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.textContent = question;
      chip.addEventListener('click', () => {
        messageInput.value = question;
        messageInput.focus();
      });
      starterQuestionsEl.appendChild(chip);
    });
  } catch {
    // fallback to local starter list
  }
}

async function loadAnalytics() {
  try {
    const response = await fetch('/api/analytics');
    if (!response.ok) return;
    const data = await response.json();
    if (!metricQueries) return;
    metricQueries.textContent = data.total_queries ?? state.totalQueries;
    metricUnanswered.textContent = data.unanswered_queries ?? state.unanswered;
    metricTools.textContent = data.tool_calls ?? state.toolCalls;
    metricSources.textContent = data.avg_sources_per_answer ?? (state.totalQueries ? (state.totalSources / state.totalQueries).toFixed(1) : '0');
  } catch {
    renderMetrics();
  }
}

async function loadHealth() {
  if (!healthStatus) return;
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error('Health check failed');
    const data = await response.json();
    const provider = data.provider_default || 'unknown';
    const embeddingState = data.embedding_status || 'n/a';
    healthStatus.textContent = `${data.ok ? 'Online' : 'Offline'} (${provider})`;
    healthStatus.title = `Embeddings: ${embeddingState}`;
  } catch {
    healthStatus.textContent = 'Unavailable';
  }
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage('You', message);
  messageInput.value = '';
  addMessage('Assistant', 'Thinking...');
  const thinkingCard = messagesEl.lastElementChild;
  setLoading(true);

  const provider = providerSelect ? providerSelect.value : 'openai';
  const model = modelInput ? modelInput.value.trim() : '';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        provider,
        model: model || undefined
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Something went wrong.');
    }

    conversationId = data.conversation_id || conversationId;
    const answer = data.answer || 'I could not find that information in the indexed corpus.';
    thinkingCard.querySelector('.message-body').textContent = answer;
    const sourcesContainer = thinkingCard.querySelector('.sources');
    sourcesContainer.innerHTML = '';
    if (Array.isArray(data.sources) && data.sources.length) {
      const title = document.createElement('div');
      title.className = 'sources-title';
      title.textContent = 'Sources';
      sourcesContainer.appendChild(title);
      data.sources.forEach((source) => {
        const card = document.createElement('a');
        card.className = 'source-card';
        card.href = source.source_url;
        card.target = '_blank';
        card.rel = 'noreferrer';
        card.innerHTML = `
          <strong>${escapeHtml(source.title)}</strong>
          <span>${escapeHtml(source.section || source.markdown_file)}</span>
          <p>${escapeHtml(source.snippet)}</p>
        `;
        sourcesContainer.appendChild(card);
      });
    }

    state.totalQueries += 1;
    state.toolCalls += Array.isArray(data.tool_trace) ? data.tool_trace.length : 0;
    state.totalSources += Array.isArray(data.sources) ? data.sources.length : 0;
    if (!Array.isArray(data.sources) || data.sources.length === 0) state.unanswered += 1;
    await loadAnalytics();
  } catch (error) {
    thinkingCard.querySelector('.message-body').textContent = `Error: ${error.message}`;
    state.totalQueries += 1;
    state.unanswered += 1;
    renderMetrics();
  } finally {
    setLoading(false);
  }
});

renderStarterQuestions();
renderMetrics();
void loadStarters();
void loadHealth();
void loadAnalytics();
addMessage(
  'Assistant',
  'Hello! Ask me about admissions, deadlines, campus services, or other university information.'
);

