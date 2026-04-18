let conversationId = null;
let isSubmitting = false;

const starterQuestions = [
  'What are the admission requirements?',
  'When is the application deadline?',
  'What campus housing options are available?',
  'How can I contact financial aid?'
];

const messagesEl = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const starterQuestionsEl = document.getElementById('starterQuestions');
const messageTemplate = document.getElementById('messageTemplate');
const healthStatus = document.getElementById('healthStatus');
const activeModelEl = document.getElementById('activeModel');

function escapeHtml(str) {
  return String(str ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderInlineMarkdown(text) {
  let rendered = escapeHtml(text);
  rendered = rendered.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, href) => {
    const safeHref = escapeHtml(href);
    return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`;
  });
  rendered = rendered.replace(/`([^`]+)`/g, '<code>$1</code>');
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  rendered = rendered.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
  return rendered;
}

function polishResponseText(text) {
  const lines = String(text ?? '').replace(/\r\n/g, '\n').split('\n');
  const cleaned = [];
  let inCodeBlock = false;
  let lastNormalized = '';

  for (const line of lines) {
    const fenceMatch = line.match(/^```/);
    if (fenceMatch) {
      inCodeBlock = !inCodeBlock;
      cleaned.push(line.trimEnd());
      continue;
    }

    if (inCodeBlock) {
      cleaned.push(line);
      continue;
    }

    const normalized = line
      .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
      .replace(/([^\n])\s{0,3}#{2,6}\s+/g, '$1 ')
      .replace(/\s+([,.;:!?])/g, '$1')
      .replace(/([.!?])([A-Za-z])/g, '$1 $2')
      .replace(/[ \t]{2,}/g, ' ')
      .replace(/\b([A-Za-z]{2,})\s+\1\b/gi, '$1')
      .trimEnd();

    if (normalized && normalized.toLowerCase() === lastNormalized) {
      continue;
    }
    lastNormalized = normalized.toLowerCase();
    cleaned.push(normalized);
  }

  return cleaned.join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

function renderMarkdownWithMarked(markdown) {
  if (typeof window === 'undefined' || !window.marked || typeof window.marked.parse !== 'function') {
    return null;
  }
  const safeMarkdown = String(markdown ?? '')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  window.marked.setOptions({
    gfm: true,
    breaks: true,
    headerIds: false,
    mangle: false
  });
  return window.marked.parse(safeMarkdown);
}

function renderMarkdown(markdown) {
  const prepared = String(markdown ?? '').replace(/\r\n/g, '\n');
  const markedHtml = renderMarkdownWithMarked(prepared);
  if (markedHtml) {
    return markedHtml;
  }

  const lines = prepared.split('\n');
  const html = [];
  let paragraphLines = [];
  let listType = null;
  let listItems = [];
  let inCodeBlock = false;
  let codeLanguage = '';
  let codeLines = [];

  function flushParagraph() {
    if (!paragraphLines.length) return;
    const text = paragraphLines.map((line) => renderInlineMarkdown(line)).join('<br />');
    html.push(`<p>${text}</p>`);
    paragraphLines = [];
  }

  function closeList() {
    if (!listType) return;
    html.push(`<${listType}>`);
    listItems.forEach((item) => {
      html.push(`<li>${item}</li>`);
    });
    html.push(`</${listType}>`);
    listType = null;
    listItems = [];
  }

  function closeCodeBlock() {
    const languageClass = codeLanguage ? ` class="language-${escapeHtml(codeLanguage)}"` : '';
    html.push(`<pre><code${languageClass}>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    inCodeBlock = false;
    codeLanguage = '';
    codeLines = [];
  }

  for (const rawLine of lines) {
    const fenceMatch = rawLine.match(/^```([\w-]+)?\s*$/);
    if (inCodeBlock) {
      if (fenceMatch) {
        closeCodeBlock();
      } else {
        codeLines.push(rawLine);
      }
      continue;
    }

    if (fenceMatch) {
      flushParagraph();
      closeList();
      inCodeBlock = true;
      codeLanguage = fenceMatch[1] || '';
      codeLines = [];
      continue;
    }

    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushParagraph();
      closeList();
      continue;
    }

    const headingMatch = rawLine.match(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (headingMatch) {
      flushParagraph();
      closeList();
      const level = headingMatch[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    const blockquoteMatch = rawLine.match(/^>\s?(.*)$/);
    if (blockquoteMatch) {
      flushParagraph();
      closeList();
      html.push(`<blockquote>${renderInlineMarkdown(blockquoteMatch[1])}</blockquote>`);
      continue;
    }

    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      flushParagraph();
      closeList();
      html.push('<hr />');
      continue;
    }

    const unorderedMatch = rawLine.match(/^[-*+]\s+(.+)$/);
    if (unorderedMatch) {
      flushParagraph();
      if (listType !== 'ul') {
        closeList();
        listType = 'ul';
      }
      listItems.push(renderInlineMarkdown(unorderedMatch[1]));
      continue;
    }

    const orderedMatch = rawLine.match(/^\d+\.\s+(.+)$/);
    if (orderedMatch) {
      flushParagraph();
      if (listType !== 'ol') {
        closeList();
        listType = 'ol';
      }
      listItems.push(renderInlineMarkdown(orderedMatch[1]));
      continue;
    }

    if (listType && /^[ \t]{2,}\S/.test(rawLine) && listItems.length > 0) {
      const continuation = renderInlineMarkdown(trimmed);
      listItems[listItems.length - 1] = `${listItems[listItems.length - 1]}<br />${continuation}`;
      continue;
    }

    closeList();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  closeList();
  if (inCodeBlock) {
    closeCodeBlock();
  }

  return html.join('\n');
}

function sanitizeAssistantResponse(text) {
  const stripped = String(text ?? '')
    .replace(/^\s*based on the indexed corpus:\s*/i, '')
    .replace(/\n\s*based on the indexed corpus:\s*/gi, '\n')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\n\|[^\n]*\|/g, '\n')
    .replace(/([^\n])\s{0,3}#{2,6}\s+/g, '$1 ')
    .trim();
  return polishResponseText(stripped);
}

function scrollToLatest(smooth = true) {
  if (!messagesEl) return;
  const latestMessage = messagesEl.lastElementChild;
  if (!latestMessage) return;
  latestMessage.scrollIntoView({
    behavior: smooth ? 'smooth' : 'auto',
    block: 'end'
  });
}

function setActiveModel(provider, model) {
  if (!activeModelEl) return;
  const providerLabel = provider ? provider.toUpperCase() : 'UNKNOWN';
  activeModelEl.textContent = model ? `${providerLabel}: ${model}` : providerLabel;
}

function renderSources(container, sources) {
  container.innerHTML = '';
  if (!Array.isArray(sources) || !sources.length) return;

  const block = document.createElement('section');
  block.className = 'sources-block';

  const title = document.createElement('h4');
  title.className = 'sources-title';
  title.textContent = 'Sources';
  block.appendChild(title);

  const list = document.createElement('ul');
  list.className = 'sources-list';

  sources.forEach((source) => {
    const item = document.createElement('li');
    item.className = 'sources-item';
    const label = source.section
      ? `${source.title} > ${source.section}`
      : source.title;

    if (source.source_url) {
      const link = document.createElement('a');
      link.href = source.source_url;
      link.target = '_blank';
      link.rel = 'noreferrer';
      link.textContent = label;
      item.appendChild(link);
    } else {
      const text = document.createElement('span');
      text.textContent = label;
      item.appendChild(text);
    }

    if (source.snippet) {
      const snippet = document.createElement('p');
      snippet.className = 'sources-snippet';
      const cleanedSnippet = String(source.snippet)
        .replace(/\r\n/g, '\n')
        .replace(/(^|\n)\s{0,3}#{1,6}\s+/g, '$1')
        .replace(/([^\n])\s{0,3}#{2,6}\s+/g, '$1 ')
        .replace(/`{1,3}/g, '')
        .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/\|[^\n]*\|/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      snippet.textContent = cleanedSnippet;
      item.appendChild(snippet);
    }

    list.appendChild(item);
  });

  block.appendChild(list);
  container.appendChild(block);
}

function addMessage(role, body, sources = [], options = {}) {
  if (!messagesEl || !messageTemplate) return null;
  const node = messageTemplate.content.cloneNode(true);
  node.querySelector('.message-role').textContent = role;
  const bodyEl = node.querySelector('.message-body');
  if (options.markdown) {
    bodyEl.innerHTML = renderMarkdown(body);
  } else {
    bodyEl.textContent = body;
  }

  const sourcesEl = node.querySelector('.sources');
  renderSources(sourcesEl, sources);

  messagesEl.appendChild(node);
  const created = messagesEl.lastElementChild;
  scrollToLatest(options.smoothScroll !== false);
  return created;
}

function setLoading(isLoading) {
  if (!chatForm || !messageInput) return;
  const button = chatForm.querySelector('button');
  if (!button) return;
  isSubmitting = isLoading;
  button.disabled = isLoading;
  button.textContent = isLoading ? 'Sending...' : 'Send';
  messageInput.disabled = isLoading;
}

async function submitMessage(messageText) {
  if (!chatForm || !messageInput || isSubmitting) return;
  const message = String(messageText ?? '').trim();
  if (!message) return;

  addMessage('You', message, [], { smoothScroll: true });
  messageInput.value = '';
  const thinkingCard = addMessage('Assistant', 'Thinking...', [], { markdown: true, smoothScroll: true });
  setLoading(true);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: conversationId
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Something went wrong.');
    }

    conversationId = data.conversation_id || conversationId;
    const rawAnswer = data.answer || data.response || 'I could not find a reliable answer in the available sources.';
    const answer = sanitizeAssistantResponse(rawAnswer) || 'I could not find a reliable answer in the available sources.';
    if (thinkingCard) {
      thinkingCard.querySelector('.message-body').innerHTML = renderMarkdown(answer);
      renderSources(thinkingCard.querySelector('.sources'), data.sources);
      scrollToLatest(true);
    }
    setActiveModel(data.provider, data.model);
  } catch (error) {
    if (thinkingCard) {
      thinkingCard.querySelector('.message-body').textContent = `Error: ${error.message}`;
      scrollToLatest(true);
    }
  } finally {
    setLoading(false);
  }
}

function renderStarterQuestions(questions) {
  if (!starterQuestionsEl || !messageInput) return;
  starterQuestionsEl.innerHTML = '';
  questions.forEach((question) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'topic-pill';
    chip.textContent = question;
    chip.addEventListener('click', () => {
      messageInput.value = question;
      void submitMessage(question);
    });
    starterQuestionsEl.appendChild(chip);
  });
}

async function loadStarters() {
  renderStarterQuestions(starterQuestions);
  try {
    const response = await fetch('/api/starter-questions');
    if (!response.ok) return;
    const data = await response.json();
    if (!Array.isArray(data.questions)) return;
    renderStarterQuestions(data.questions);
  } catch {
    // Keep fallback starter questions.
  }
}

async function loadHealth() {
  if (!healthStatus) return;
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error('Health check failed');
    const data = await response.json();
    const provider = data.active_provider || data.provider_default || 'unknown';
    const model = data.active_model || data.model_default || null;
    healthStatus.textContent = data.ok ? 'Online' : 'Offline';
    setActiveModel(provider, model);
  } catch {
    healthStatus.textContent = 'Unavailable';
    setActiveModel(null, null);
  }
}

if (chatForm && messageInput) {
  messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void submitMessage(messageInput.value);
    }
  });

  chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    await submitMessage(messageInput.value);
  });
}

void loadStarters();
void loadHealth();
addMessage(
  'Assistant',
  'Hello! Ask me about admissions, deadlines, campus services, or other university information.',
  [],
  { markdown: true, smoothScroll: false }
);
