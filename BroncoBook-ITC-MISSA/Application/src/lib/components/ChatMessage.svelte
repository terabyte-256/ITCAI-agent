<script lang="ts">
	import type { CitationRecord } from '$lib/types';

	interface UiMessage {
		role: 'user' | 'assistant' | 'system';
		content: string;
		citations?: CitationRecord[];
	}

	let { message } = $props<{ message: UiMessage }>();
</script>

<article class={`message ${message.role}`}>
	<div class="role">{message.role === 'user' ? 'You' : message.role === 'assistant' ? 'Assistant' : 'System'}</div>
	<div class="content">{message.content}</div>

	{#if message.citations?.length}
		<div class="citations">
			<div class="citations-label">Sources</div>
			{#each message.citations as citation}
				<a href={citation.originalUrl} target="_blank" rel="noreferrer" class="citation-card">
					<strong>{citation.title}</strong>
					<div class="muted">{citation.headingPath ?? 'Page section'}</div>
					<div class="muted">{citation.originalUrl}</div>
					<p>{citation.excerpt}</p>
				</a>
			{/each}
		</div>
	{/if}
</article>

<style>
	.message {
		border: 1px solid var(--border);
		border-radius: 1rem;
		padding: 0.9rem;
		background: var(--panel);
		display: grid;
		gap: 0.6rem;
	}

	.message.user {
		background: #223b7a;
	}

	.role {
		font-size: 0.78rem;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--accent);
	}

	.content {
		white-space: pre-wrap;
		line-height: 1.5;
	}

	.citations {
		display: grid;
		gap: 0.55rem;
	}

	.citations-label {
		font-size: 0.8rem;
		color: var(--muted);
	}

	.citation-card {
		display: grid;
		gap: 0.2rem;
		text-decoration: none;
		color: inherit;
		border: 1px solid var(--border);
		background: rgba(0, 0, 0, 0.2);
		border-radius: 0.75rem;
		padding: 0.7rem;
	}

	.citation-card p {
		margin: 0.2rem 0 0;
		font-size: 0.92rem;
	}
</style>

