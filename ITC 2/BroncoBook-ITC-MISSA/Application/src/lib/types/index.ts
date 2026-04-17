export interface CitationRecord {
	chunkId: string;
	documentId: string;
	title: string;
	headingPath: string | null;
	originalUrl: string;
	excerpt: string;
}
export type ProviderName = 'openai' | 'ollama';
