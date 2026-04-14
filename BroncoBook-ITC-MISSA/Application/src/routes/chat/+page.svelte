<script lang="ts">
	import { onMount } from 'svelte';

	import ChatMessage from '$lib/components/ChatMessage.svelte';
	import type { CitationRecord, ProviderName } from '$lib/types';

	interface UiMessage {
		role: 'user' | 'assistant' | 'system';
		content: string;
		citations?: CitationRecord[];
	}

	const starterQuestions = [
		'What are the freshman admission requirements?',
		'How do I change my major?',
		'Where is Student Health Services located?',
		'What financial aid resources are available?'
	];

	let provider = $state<ProviderName>('openai');
	let model = $state('');
	let draft = $state('');
	let loading = $state(false);
	let error = $state('');
	let conversationId = $state<string | null>(null);
	let messages = $state<UiMessage[]>([
		{
			role: 'assistant',
			content:
				'Hi! I answer campus questions using only the indexed markdown corpus and I always attach sources.'
		}
	]);

	const storageKey = 'campus-agent:conversation-id';
	const messagesStorageKey = 'campus-agent:messages';

	onMount(() => {
		const stored = localStorage.getItem(storageKey);
		if (stored) {
			conversationId = stored;
		}
		const storedMessages = localStorage.getItem(messagesStorageKey);
		if (storedMessages) {
			try {
				messages = JSON.parse(storedMessages) as UiMessage[];
			} catch {
				// ignore corrupted local state
			}
		}
	});

	async function send(messageText: string) {
		const message = messageText.trim();
		if (!message || loading) {
			return;
		}
		error = '';
		messages = [...messages, { role: 'user', content: message }];
		draft = '';
		loading = true;

		try {
			const response = await fetch('/api/chat', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					conversation_id: conversationId,
					message,
					provider,
					model: model.trim() || undefined
				})
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(data.error ?? 'Request failed.');
			}
			conversationId = data.conversation_id ?? conversationId;
			if (conversationId) {
				localStorage.setItem(storageKey, conversationId);
			}
			messages = [
				...messages,
				{
					role: 'assistant',
					content: data.answer,
					citations: (data.sources ?? []).map((source: Record<string, string>) => ({
						chunkId: source.markdown_file + ':' + (source.section ?? ''),
						documentId: source.markdown_file,
						title: source.title,
						headingPath: source.section ?? null,
						originalUrl: source.source_url,
						excerpt: source.snippet
					}))
				}
			];
			localStorage.setItem(messagesStorageKey, JSON.stringify(messages));
		} catch (err) {
			error = err instanceof Error ? err.message : 'Unknown error';
			messages = [...messages, { role: 'system', content: `Error: ${error}` }];
		} finally {
			loading = false;
		}
	}

	function useStarter(question: string) {
		draft = question;
	}

	function resetConversation() {
		conversationId = null;
		localStorage.removeItem(storageKey);
		localStorage.removeItem(messagesStorageKey);
		messages = [
			{
				role: 'assistant',
				content:
					'Started a new conversation. I will still retrieve fresh corpus evidence for every campus question.'
			}
		];
	}
</script>

<main class="shell chat-page">
	<section class="panel sidebar">
		<h1>Campus Knowledge Agent</h1>
		<p class="muted">Grounded answers from the indexed markdown corpus only.</p>

		<label>
			<div class="muted">Provider</div>
			<select class="select" bind:value={provider}>
				<option value="openai">OpenAI</option>
				<option value="ollama">Ollama</option>
			</select>
		</label>

		<label>
			<div class="muted">Model (optional override)</div>
			<input class="input" placeholder="Leave blank for provider default" bind:value={model} />
		</label>

		<div class="chip-row">
			{#each starterQuestions as question}
				<button class="button" type="button" onclick={() => useStarter(question)}>{question}</button>
			{/each}
		</div>

		<button class="button" type="button" onclick={resetConversation}>New conversation</button>
		{#if conversationId}
			<div class="muted conversation-id">Conversation: {conversationId}</div>
		{/if}
		{#if error}
			<div class="error">{error}</div>
		{/if}
	</section>

	<section class="panel chat-main">
		<div class="messages">
			{#each messages as message}
				<ChatMessage {message} />
			{/each}
		</div>

		<form
			class="composer"
			onsubmit={(event) => {
				event.preventDefault();
				void send(draft);
			}}
		>
			<textarea
				class="input"
				rows="4"
				placeholder="Ask about admissions, aid, deadlines, services, and more..."
				bind:value={draft}
			></textarea>
			<button class="button is-primary" disabled={loading || !draft.trim()} type="submit">
				{loading ? 'Sending…' : 'Send'}
			</button>
		</form>
	</section>
</main>

<style>
	.chat-page {
		display: grid;
		grid-template-columns: 320px 1fr;
		gap: 1rem;
	}

	.sidebar,
	.chat-main {
		display: grid;
		gap: 0.9rem;
		align-content: start;
	}

	.conversation-id {
		word-break: break-all;
		font-size: 0.85rem;
	}

	.messages {
		display: grid;
		gap: 0.8rem;
		max-height: 62vh;
		overflow: auto;
		padding-right: 0.25rem;
	}

	.composer {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 0.6rem;
		align-items: end;
	}

	.error {
		color: var(--danger);
		font-size: 0.9rem;
	}

	@media (max-width: 980px) {
		.chat-page {
			grid-template-columns: 1fr;
		}
	}
</style>
