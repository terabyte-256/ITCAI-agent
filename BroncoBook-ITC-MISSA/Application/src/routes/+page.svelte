<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	onMount(() => {
		void goto('/chat');
	});
</script>

<main class="shell">
	<section class="panel">
		<h1>Campus Knowledge Agent</h1>
		<p>Redirecting to chat…</p>
	</section>
</main>
