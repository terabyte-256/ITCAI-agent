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
  totalSources: 0,
};

const messagesEl = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const starterQuestionsEl = document.getElementById('starterQuestions');
const messageTemplate = document.getElementById('messageTemplate');

const metricQueries = document.getElementById('metricQueries');
const metricUnanswered = document.getElementById('metricUnanswered');
const metricTools = document.getElementById('metricTools');
const metricSources = document.getElementById('metricSources');
const healthStatus = document.getElementById('healthStatus');

function renderMetrics() {
  metricQueries.textContent = state.totalQueries;
  metricUnanswered.textContent = state.unanswered;
  metricTools.textContent = state.toolCalls;
  metricSources.textContent = state.totalQueries
    ? (state.totalSources / state.totalQueries).toFixed(1)
    : '0';
}

function addMessage(role, body) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector('.message-role').textContent = role;
  node.querySelector('.message-body').textContent = body;
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  const button = chatForm.querySelector('button');
  button.disabled = isLoading;
  button.textContent = isLoading ? 'Sending...' : 'Send';
  messageInput.disabled = isLoading;
}

function renderStarterQuestions() {
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

async function loadHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error('Health check failed');

    const data = await response.json();
    const provider = data.provider || 'unknown';
    const model = provider === 'openai'
      ? (data.openai_model || 'unknown')
      : (data.ollama_host ? `ollama @ ${data.ollama_host}` : 'ollama');

    healthStatus.textContent = `${data.ok ? 'Online' : 'Offline'} - ${provider} (${model})`;
  } catch (error) {
    healthStatus.textContent = 'Unavailable';
  }
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage('You', message);
  messageInput.value = '';
  setLoading(true);

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Something went wrong.');
    }

    const answer = data.response || 'No response returned.';
    const provider = data.provider || 'unknown';
    const model = data.model || 'unknown';

    addMessage('Assistant', `${answer}\n\n[Provider: ${provider} | Model: ${model}]`);

    state.totalQueries += 1;
    state.toolCalls += 1;
    if (!data.response) state.unanswered += 1;
    renderMetrics();
  } catch (error) {
    addMessage('System', `Error: ${error.message}`);
    state.totalQueries += 1;
    state.unanswered += 1;
    renderMetrics();
  } finally {
    setLoading(false);
  }
});

renderStarterQuestions();
renderMetrics();
loadHealth();
addMessage(
  'Assistant',
  'Hello! Ask me about admissions, deadlines, campus services, or other university information.'
);