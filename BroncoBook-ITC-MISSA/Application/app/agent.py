from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI

from .db import SQLiteStore
from .models import ChatResponse, RetrievalDebugChunk, RetrievalDebugInfo, SearchResult, SourceItem
from .prompts import FINANCIAL_AID_FALLBACK_MESSAGE, NOT_FOUND_MESSAGE, SYSTEM_PROMPT, TOOLS
from .retriever import CorpusRetriever

# Factoid triggers: when any of these appears in the user's question, we
# require that same token (or a clear synonym) to appear in the retrieved
# chunks before we produce a grounded answer. This is the main defense
# against false-grounding hallucinations of the form "Who is the CEO of
# Cal Poly Pomona?" where brand-name lexical overlap otherwise passes
# coverage checks.
FACTOID_TRIGGERS: Dict[str, tuple[str, ...]] = {
    "ceo": ("ceo", "chief executive"),
    "cfo": ("cfo", "chief financial"),
    "cto": ("cto", "chief technology"),
    "coo": ("coo", "chief operating"),
    "founder": ("founder", "founded by"),
    "owner": ("owner", "owned by"),
    "provost": ("provost",),
    "dean": ("dean",),
    "principal": ("principal",),
    "chancellor": ("chancellor",),
    "mascot": ("mascot",),
    "address": ("address", "located at", "street", "building", "room"),
    "phone": ("phone", "call", "telephone", "tel:"),
    "fax": ("fax",),
    "email": ("email", "@", "e-mail"),
    "hours": ("hours", "open", "close", "a.m.", "p.m.", "am-", "pm-"),
    "price": ("price", "cost", "$", "fee"),
    "tuition": ("tuition", "$", "fee"),
    "cost": ("cost", "$", "price", "fee"),
    "deadline": ("deadline", "by ", "due ", "no later than"),
    "gpa": ("gpa", "grade point"),
    "ranking": ("ranking", "ranked", "rank"),
    "enrollment": ("enrollment", "enrolled", "students"),
    "population": ("population", "students", "enrollment"),
    "founded": ("founded", "established", "since "),
    "established": ("established", "founded", "since "),
}

# Minimum sentence quality gate used by the deterministic fallback so we
# don't promote dangling fragments like "11th-17th" to an answer line.
MIN_SENTENCE_CHARS = 24
MIN_SENTENCE_WORDS = 4

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "available",
}


class ToolDispatcher:
    def __init__(
        self,
        *,
        retriever: CorpusRetriever,
        provider: str,
        model: Optional[str],
        max_results: int,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.model = model
        self.max_results = max_results

    def search_corpus(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        limit = max(1, min(int(top_k), self.max_results))
        results = self.retriever.search_corpus(
            query=query,
            top_k=limit,
            provider=self.provider,
            embedding_model=self.model,
        )
        return {
            "tool": "search_corpus",
            "query": query,
            "top_k": limit,
            "mode": results[0].retrieval_method if results else "none",
            "results": [result.model_dump() for result in results],
        }

    def get_chunk_context(self, chunk_ids: List[str]) -> Dict[str, Any]:
        unique_ids = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
        chunks = self.retriever.get_chunk_context(unique_ids)
        return {
            "tool": "get_chunk_context",
            "chunk_ids": unique_ids,
            "chunks": [item.model_dump() for item in chunks],
        }

    def list_sources_for_answer(self, chunk_ids: List[str]) -> Dict[str, Any]:
        unique_ids = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
        sources = self.retriever.list_sources_for_answer(unique_ids)
        return {
            "tool": "list_sources_for_answer",
            "chunk_ids": unique_ids,
            "sources": [item.model_dump() for item in sources],
        }

    def dispatch(self, tool_name: str, arguments: str | Dict[str, Any]) -> Dict[str, Any]:
        args: Dict[str, Any]
        if isinstance(arguments, str):
            args = json.loads(arguments or "{}")
        else:
            args = arguments
        if tool_name == "search_corpus":
            return self.search_corpus(str(args["query"]), int(args.get("top_k", 5)))
        if tool_name == "get_chunk_context":
            return self.get_chunk_context([str(item) for item in args.get("chunk_ids", [])])
        if tool_name == "list_sources_for_answer":
            return self.list_sources_for_answer([str(item) for item in args.get("chunk_ids", [])])
        raise ValueError(f"Unknown tool: {tool_name}")


class AgentService:
    def __init__(self, retriever: CorpusRetriever, store: SQLiteStore, ttl_minutes: int = 90) -> None:
        self.retriever = retriever
        self.store = store
        self.ttl_minutes = ttl_minutes
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.max_tool_results = int(os.getenv("MAX_TOOL_RESULTS", "6"))
        self.min_retrieval_score = float(os.getenv("MIN_RETRIEVAL_FINAL_SCORE", "0.25"))
        self.ollama_host = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.3")

    def _is_retrieval_confident(self, results: List[SearchResult]) -> bool:
        if not results:
            return False
        top = results[0].final_score if results[0].final_score is not None else results[0].score
        return float(top) >= self.min_retrieval_score

    def _extract_query_terms(self, query: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 2 and token not in QUERY_STOPWORDS
        ]

    def _query_relevance_score(self, query: str, result: SearchResult) -> float:
        terms = self._extract_query_terms(query)
        if not terms:
            return 0.0
        haystack = f"{result.title}\n{result.section or ''}\n{result.source_url}\n{result.markdown_file}\n{result.content}".lower()
        matched = sum(1 for term in terms if term in haystack)
        coverage = matched / len(terms)
        query_phrase = query.strip().lower()
        phrase_bonus = 0.35 if query_phrase and query_phrase in haystack else 0.0
        score = coverage + phrase_bonus

        source_text = f"{result.source_url}\n{result.markdown_file}".lower()
        if "financial aid" in query.lower():
            if any(marker in source_text for marker in ("/~financial-aid", "/financial-aid/", "financial-aid__")):
                score += 0.35
            if "/aboutcpp/" in source_text and "financial aid office" not in haystack:
                score -= 0.45
        if self._is_finals_schedule_intent(query):
            if any(term in haystack for term in ("final exam", "final exams", "finals week", "exam schedule")):
                score += 0.25
            if "/admissions/" in source_text:
                score -= 0.55
        return score

    def _filter_relevant_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return []
        scored: List[tuple[float, SearchResult]] = [
            (self._query_relevance_score(query, result), result)
            for result in results
        ]
        scored.sort(
            key=lambda pair: (
                -pair[0],
                -(pair[1].final_score if pair[1].final_score is not None else pair[1].score),
                pair[1].chunk_id,
            )
        )
        top = scored[0][0]
        threshold = max(0.34, top * 0.55)
        filtered = [item for score, item in scored if score >= threshold]
        return filtered

    def _has_query_coverage(self, query: str, results: List[SearchResult]) -> bool:
        terms = self._extract_query_terms(query)
        if not terms or not results:
            return False
        combined = " ".join(f"{item.title} {item.section or ''} {item.content}" for item in results[:4]).lower()
        matched = sum(1 for term in terms if term in combined)
        if len(terms) <= 2:
            has_coverage = matched >= 1
        else:
            has_coverage = matched >= 2 and (matched / len(terms)) >= 0.5
        if not has_coverage:
            return False
        # Additional gate: if the query contains a factoid trigger (a role,
        # contact-detail word, or other entity-specific word), at least one
        # of that trigger's synonyms must be present in the retrieved
        # content. Brand-name overlap alone is not sufficient.
        missing_triggers = self._missing_factoid_triggers(query, combined)
        if missing_triggers:
            return False
        return True

    def _missing_factoid_triggers(self, query: str, combined_text_lower: str) -> List[str]:
        """Return factoid trigger words from the query whose synonyms are
        completely absent from the retrieved text. If any are missing, the
        retrieval does not actually answer the question and we should
        decline instead of paraphrasing around the gap.

        Synonyms are matched with word-boundary semantics where possible so
        that, for example, "ceo" does not spuriously match inside the URL
        slug "officeofequity".
        """
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        missing: List[str] = []
        for trigger, synonyms in FACTOID_TRIGGERS.items():
            if trigger not in query_tokens:
                continue
            if not any(self._text_contains_token(combined_text_lower, syn) for syn in synonyms):
                missing.append(trigger)
        return missing

    @staticmethod
    def _text_contains_token(haystack_lower: str, needle: str) -> bool:
        needle_lower = needle.lower()
        if not needle_lower:
            return False
        # Punctuation or non-alphanumeric-heavy synonyms ("$", "@", "tel:",
        # "e-mail") are matched literally because adding word boundaries
        # around them would never match.
        if not re.search(r"[A-Za-z0-9]", needle_lower) or any(
            ch in needle_lower for ch in ("@", "$", ":", "-")
        ):
            return needle_lower in haystack_lower
        pattern = r"(?<![A-Za-z0-9])" + re.escape(needle_lower) + r"(?![A-Za-z0-9])"
        return re.search(pattern, haystack_lower) is not None

    def _is_valid_answer_sentence(self, sentence: str) -> bool:
        """Reject fragments, headings-as-sentences, and list-dangle remnants
        so the deterministic fallback doesn't stitch together nonsense."""
        stripped = sentence.strip()
        if len(stripped) < MIN_SENTENCE_CHARS:
            return False
        alpha_words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", stripped)]
        if len(alpha_words) < MIN_SENTENCE_WORDS:
            return False
        # Reject sentences that are mostly digits / dates / ranges without
        # any real verb-like content.
        letters = sum(1 for ch in stripped if ch.isalpha())
        if letters < len(stripped) * 0.5:
            return False
        return True

    @staticmethod
    def _strip_markdown_markers(text: str) -> str:
        """Remove markdown bold/italic markers, heading prefixes, list
        bullets, and link syntax, then collapse whitespace so source cards
        don't display raw ``**foo**`` or ``### foo`` fragments."""
        if not text:
            return ""
        cleaned = text
        # Heading markers: strip leading ``#{1,6} `` at line starts and any
        # mid-string inline ``#{2,6} `` that survived chunking.
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"(?<!\w)#{2,6}\s+", " ", cleaned)
        cleaned = re.sub(r"\*{1,3}", "", cleaned)
        cleaned = re.sub(r"(?<!\w)_{1,3}(?!\w)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _sanitize_source(self, source: SourceItem) -> SourceItem:
        """Clean a SourceItem for display: strip markdown markers from text
        fields and promote a better title when the scraped HTML H1 was just a
        generic chrome token like 'Menu' or '|'."""
        section = self._strip_markdown_markers(source.section or "")
        snippet = self._strip_markdown_markers(source.snippet or "")
        title = self._strip_markdown_markers(source.title or "")

        generic_titles = {"", "menu", "|", "home", "home page", "cal poly pomona", "cpp"}
        if title.lower() in generic_titles:
            # Prefer the last meaningful segment of the section/heading path.
            if section:
                segments = [seg.strip() for seg in section.split(">") if seg.strip()]
                if segments:
                    title = segments[-1]
            if title.lower() in generic_titles and source.markdown_file:
                # Derive a human-ish title from the markdown filename slug.
                slug = source.markdown_file
                slug = re.sub(r"\.md$", "", slug, flags=re.IGNORECASE)
                slug = re.sub(r"\.shtml$", "", slug, flags=re.IGNORECASE)
                slug = slug.replace("__", " · ").replace("_", " ").replace("-", " ")
                title = slug.strip().title() or title

        title = title or "Cal Poly Pomona"

        return source.model_copy(
            update={
                "title": title,
                "section": section or None,
                "snippet": snippet,
            }
        )

    def _filter_sources_for_query(
        self,
        *,
        query: str,
        sources: List[SourceItem],
        selected_results: List[SearchResult],
    ) -> List[SourceItem]:
        if not sources or not selected_results:
            return []
        results_by_chunk = {item.chunk_id: item for item in selected_results}
        deduped: List[SourceItem] = []
        seen: set[tuple[str, str]] = set()
        for source in sources:
            if not source.chunk_id or source.chunk_id not in results_by_chunk:
                continue
            result = results_by_chunk[source.chunk_id]
            if self._query_relevance_score(query, result) < 0.34:
                continue
            cleaned = self._sanitize_source(source)
            key = (cleaned.source_url, (cleaned.section or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _is_answer_supported_by_results(self, user_message: str, answer: str, results: List[SearchResult]) -> bool:
        if not answer or answer.strip().lower() == NOT_FOUND_MESSAGE.lower():
            return True
        if not results:
            return False
        answer_plain = re.sub(r"```[\s\S]*?```", " ", answer)
        answer_plain = re.sub(r"^\s{0,3}#{1,6}\s*", "", answer_plain, flags=re.MULTILINE)
        answer_plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", answer_plain)
        answer_lower = answer_plain.lower()
        answer_terms = self._extract_query_terms(answer_plain)
        if not answer_terms:
            return False
        context_text = " ".join(item.content for item in results).lower()

        # Reject answers that assert facts the context never mentions.
        missing_triggers = self._missing_factoid_triggers(user_message, context_text)
        if missing_triggers:
            return False

        # Claim-level guard: if the answer contains a numeric/date-looking
        # claim, that claim must appear in the retrieved context. Hallucinated
        # dates (e.g., "December 11th-17th") are a common failure mode of
        # the deterministic fallback and LLMs alike, so we enforce presence
        # of any distinctive numeric tokens.
        numeric_claims = set(
            re.findall(r"\b(?:\d{1,4}(?:st|nd|rd|th)?|\$\d[\d,\.]*|\d+%)\b", answer_plain)
        )
        # Allow common bullet / ordinal numbers to pass without strict
        # lookup, but any multi-digit or currency/percent value must trace
        # back to retrieval.
        unsupported_numbers = [
            token for token in numeric_claims
            if (len(token) > 1 or token.startswith("$"))
            and token.lower() not in context_text
        ]
        if unsupported_numbers:
            return False

        supported_terms = sum(1 for term in set(answer_terms[:32]) if term in context_text)
        support_ratio = supported_terms / max(1, len(set(answer_terms[:32])))
        query_terms = self._extract_query_terms(user_message)
        query_terms_in_answer = sum(1 for term in query_terms if term in answer_lower)
        min_query_terms = max(1, (len(query_terms) + 1) // 2) if query_terms else 0
        if query_terms and query_terms_in_answer < min_query_terms:
            return False
        return support_ratio >= 0.3

    def _is_low_quality_result(self, result: SearchResult) -> bool:
        title = (result.title or "").strip().lower()
        section = (result.section or "").strip().lower()
        content = (result.content or "").lower()
        source_url = (result.source_url or "").lower()
        if not section and title in {"", "|", "menu"}:
            return True
        if "skip to content" in content and (not section or section == "menu"):
            return True
        if "close menu" in content and not section:
            return True
        if any(marker in content for marker in ("home page menu", "search this site", "main menu")) and not section:
            return True
        if content.count("![](") >= 2 and content.count("](") >= 6:
            return True
        if "follow us" in section:
            return True
        if source_url.endswith((".jpg", ".jpeg", ".png", ".gif", ".svg")):
            return True
        return False

    def _rerank_for_query_intent(self, user_message: str, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results
        query = user_message.lower()

        def apply_rank(predicate, scorer):
            candidates = []
            for item in results:
                haystack = f"{item.title}\n{item.section or ''}\n{item.content}".lower()
                if not predicate(haystack):
                    continue
                candidates.append((scorer(haystack), item))
            if not candidates:
                return []
            candidates.sort(
                key=lambda pair: (
                    -pair[0],
                    -(pair[1].final_score if pair[1].final_score is not None else pair[1].score),
                    pair[1].chunk_id,
                )
            )
            return [item for _, item in candidates]

        freshman_intent = ("freshman" in query or "freshmen" in query or "first-year" in query or "first year" in query) and (
            "admission" in query or "admissions" in query
        )
        if freshman_intent:
            candidates = []
            for item in results:
                if not self._is_freshman_admissions_candidate(item, include_content=True):
                    continue
                source_text = f"{item.source_url or ''}\n{item.markdown_file or ''}".lower()
                title_section = f"{item.title}\n{item.section or ''}".lower()
                full_text = f"{title_section}\n{source_text}\n{item.content}".lower()
                score = (
                    (3 if any(token in full_text for token in ("requirement", "requirements")) else 0)
                    + (2 if "checklist" in full_text else 0)
                    + (2 if any(token in full_text for token in ("application", "apply")) else 0)
                    + (2 if "office of admissions" in title_section else 0)
                    + (3 if any(marker in source_text for marker in ("/admissions/freshmen/", "admissions__freshmen")) else 0)
                )
                candidates.append((score, item))
            if candidates:
                candidates.sort(
                    key=lambda pair: (
                        -pair[0],
                        -(pair[1].final_score if pair[1].final_score is not None else pair[1].score),
                        pair[1].chunk_id,
                    )
                )
                return [item for _, item in candidates]

        financial_aid_intent = self._is_financial_aid_intent(query)
        if financial_aid_intent:
            candidates = []
            for item in results:
                if not self._is_financial_aid_candidate(item, include_content=True):
                    continue
                source_text = f"{item.source_url or ''}\n{item.markdown_file or ''}".lower()
                full_text = f"{item.title}\n{item.section or ''}\n{item.content}".lower()
                score = (
                    (4 if "financial aid" in full_text else 0)
                    + (3 if any(term in full_text for term in ("resource", "resources", "scholarship", "grant", "loan")) else 0)
                    + (2 if any(term in full_text for term in ("fafsa", "dream act", "application", "deadline")) else 0)
                    + (
                        4
                        if any(
                            marker in source_text
                            for marker in ("/~financial-aid", "/financial-aid/", "financial-aid__")
                        )
                        else 0
                    )
                    - (7 if "/aboutcpp/" in source_text else 0)
                    - (5 if "follow us" in full_text else 0)
                )
                candidates.append((score, item))
            if candidates:
                candidates.sort(
                    key=lambda pair: (
                        -pair[0],
                        -(pair[1].final_score if pair[1].final_score is not None else pair[1].score),
                        pair[1].chunk_id,
                    )
                )
                return [item for _, item in candidates]

        finals_intent = self._is_finals_schedule_intent(query)
        if finals_intent:
            candidates = []
            for item in results:
                if not self._is_finals_schedule_candidate(item, include_content=True):
                    continue
                source_text = f"{item.source_url or ''}\n{item.markdown_file or ''}".lower()
                full_text = f"{item.title}\n{item.section or ''}\n{item.content}".lower()
                score = (
                    (4 if any(term in full_text for term in ("final exam", "final exams", "finals week", "exam schedule")) else 0)
                    + (3 if any(term in full_text for term in ("academic calendar", "calendar")) else 0)
                    + (2 if any(term in full_text for term in ("date", "dates", "week")) else 0)
                    - (6 if "/admissions/" in source_text else 0)
                )
                candidates.append((score, item))
            if candidates:
                candidates.sort(
                    key=lambda pair: (
                        -pair[0],
                        -(pair[1].final_score if pair[1].final_score is not None else pair[1].score),
                        pair[1].chunk_id,
                    )
                )
                return [item for _, item in candidates]

        health_location_intent = ("student health" in query or "health services" in query) and (
            "where" in query or "location" in query or "located" in query or "address" in query
        )
        if health_location_intent:
            def health_predicate(text: str) -> bool:
                return ("health" in text and ("service" in text or "services" in text or "center" in text))

            def health_score(text: str) -> int:
                return (3 if ("location" in text or "address" in text or "located" in text) else 0)

            reranked = apply_rank(health_predicate, health_score)
            if reranked:
                return reranked

        return results

    def _is_freshman_admissions_intent(self, user_message: str) -> bool:
        query = user_message.lower()
        return ("freshman" in query or "freshmen" in query or "first-year" in query or "first year" in query) and (
            "admission" in query or "admissions" in query
        )

    def _is_financial_aid_intent(self, user_message: str) -> bool:
        query = user_message.lower()
        return ("financial aid" in query) or (
            "aid" in query and any(term in query for term in ("resource", "resources", "scholarship", "grant", "loan", "fafsa"))
        )

    def _is_financial_aid_candidate(self, item: SearchResult, include_content: bool = True) -> bool:
        source_text = f"{item.source_url or ''}\n{item.markdown_file or ''}".lower()
        title_section = f"{item.title}\n{item.section or ''}".lower()
        full_text = f"{title_section}\n{source_text}\n{item.content or ''}".lower() if include_content else f"{title_section}\n{source_text}"
        has_financial_scope = (
            "office of financial aid" in title_section
            or any(marker in source_text for marker in ("/~financial-aid", "/financial-aid/", "financial-aid__"))
        )
        has_resource_signal = any(
            term in full_text
            for term in ("resource", "resources", "scholarship", "grant", "loan", "fafsa", "dream act", "application", "deadline")
        )
        if "/aboutcpp/" in source_text and "financial aid office" not in full_text:
            return False
        if "follow us" in title_section or "follow us" in full_text:
            return False
        if any(
            marker in full_text
            for marker in (
                "credential students",
                "course eligibility",
                "ineligible courses",
                "after you apply",
                "award notification",
                "getting your aid",
                "keeping your aid",
                "changes in income or circumstances",
                "verification process",
                "document verification",
                "verify your",
                "status update",
                "status of your",
                "check your status",
                "notification email",
                "notification letter",
                "notifications are sent",
                "broncodirect",
                "sap appeal",
                "satisfactory academic progress appeal",
            )
        ):
            return False
        # Exclude procedural/FAQ URL slugs explicitly -- these describe the
        # administrative workflow (verification, notifications, status) rather
        # than the available aid resources themselves.
        procedural_url_markers = (
            "/faqs",
            "/faq.",
            "/credentials/",
            "/after-you-apply",
            "/award-notification",
            "/course-eligibility",
            "/getting-your-aid",
            "/keeping-your-aid",
            "/verification",
            "/on-campus-employers-information",
            "/student-information",
        )
        if any(marker in source_text for marker in procedural_url_markers):
            return False
        return has_financial_scope and has_resource_signal

    def _has_strong_financial_aid_evidence(self, results: List[SearchResult]) -> bool:
        if not results:
            return False
        strong_markers = (
            "grant",
            "loan",
            "fafsa",
            "dream act",
            "work-study",
            "scholarships and financial aid",
            "types of aid",
            "how to apply",
            "application deadline",
        )
        weak_markers = ("credential students", "course eligibility", "ineligible courses", "what are my options")
        strong_hits = 0
        weak_hits = 0
        for item in results[:4]:
            text = f"{item.title}\n{item.section or ''}\n{item.content}".lower()
            if any(marker in text for marker in strong_markers):
                strong_hits += 1
            if any(marker in text for marker in weak_markers):
                weak_hits += 1
        return strong_hits > 0 and strong_hits >= weak_hits

    def _is_finals_schedule_intent(self, user_message: str) -> bool:
        query = user_message.lower()
        finals_signal = any(term in query for term in ("when are finals", "finals", "final exam", "final exams"))
        schedule_signal = any(term in query for term in ("when", "date", "dates", "schedule", "calendar", "week"))
        return finals_signal and schedule_signal

    def _is_finals_schedule_candidate(self, item: SearchResult, include_content: bool = True) -> bool:
        source_text = f"{item.source_url or ''}\n{item.markdown_file or ''}".lower()
        title_section = f"{item.title}\n{item.section or ''}".lower()
        full_text = f"{title_section}\n{source_text}\n{item.content or ''}".lower() if include_content else f"{title_section}\n{source_text}"
        has_finals = any(term in full_text for term in ("final exam", "final exams", "finals week", "exam schedule", "finals"))
        has_schedule = any(term in full_text for term in ("date", "dates", "calendar", "schedule", "week", "term"))
        if "/admissions/" in source_text and "final exam" not in full_text and "finals week" not in full_text:
            return False
        return has_finals and has_schedule

    def _is_freshman_admissions_candidate(self, item: SearchResult, include_content: bool = True) -> bool:
        source_text = f"{item.source_url or ''}\n{item.markdown_file or ''}".lower()
        title_section = f"{item.title}\n{item.section or ''}".lower()
        scope_text = f"{title_section}\n{source_text}"
        full_text = f"{scope_text}\n{item.content or ''}".lower() if include_content else scope_text

        has_admissions_scope = (
            any(marker in source_text for marker in ("/admissions/", "admissions__", "/office-of-admissions/"))
            or "office of admissions" in title_section
            or ("admission" in title_section and "registrar" not in source_text)
        )
        has_freshman_signal = any(term in full_text for term in ("freshman", "freshmen", "first-year", "first year"))
        return has_admissions_scope and has_freshman_signal

    def _result_dedupe_key(self, result: SearchResult) -> tuple[str, str, str]:
        normalized_file = (result.markdown_file or "").lower().lstrip("_")
        normalized_url = (result.source_url or "").lower().replace("%7e", "~")
        normalized_section = (result.section or "").strip().lower()
        content_signature = " ".join((result.content or "").split())[:180].lower()
        return (normalized_file or normalized_url, normalized_section, content_signature)

    def _lookup_freshman_admissions_results(self, limit: int) -> List[SearchResult]:
        rows = self.retriever.store.fetchall(
            """
            SELECT
                dc.id AS chunk_id,
                dc.document_id AS document_id,
                dc.chunk_index AS chunk_index,
                dc.heading_path AS heading_path,
                dc.content AS content,
                d.title AS title,
                d.original_url AS source_url,
                d.file_path AS markdown_file
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE (
                lower(d.original_url) LIKE '%/admissions/freshmen/%'
                OR lower(d.file_path) LIKE '%admissions__freshmen%'
            )
            """
        )
        if not rows:
            return []

        ranked: List[tuple[int, SearchResult]] = []
        for row in rows:
            title = str(row["title"] or "")
            section = str(row["heading_path"] or "")
            content = str(row["content"] or "")
            source_url = str(row["source_url"] or "")
            markdown_file = str(row["markdown_file"] or "")
            source_text = f"{source_url}\n{markdown_file}".lower()
            full_text = f"{title}\n{section}\n{content}\n{source_text}".lower()
            score = 0
            if "requirements" in source_text or "/requirements" in source_text:
                score += 14
            if "app-checklist" in source_text or "checklist" in source_text:
                score += 11
            if "requirement" in section.lower():
                score += 7
            if "checklist" in section.lower():
                score += 6
            if any(term in full_text for term in ("admission requirement", "admissions requirements")):
                score += 6
            if any(term in full_text for term in ("freshman applicants", "freshman students", "begin your application")):
                score += 3
            if "spring application cycle is not open to freshmen applicants" in full_text:
                score += 2
            if score <= 0:
                continue

            condensed = " ".join(content.split())
            snippet = condensed[:280].strip()
            if len(condensed) > 280:
                snippet += "..."
            result = SearchResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                score=min(1.0, score / 30.0),
                title=title,
                source_url=source_url,
                markdown_file=markdown_file,
                section=section or None,
                snippet=snippet,
                content=content[:2600],
                retrieval_method="targeted_admissions_lookup",
                fts_score=None,
                vector_score=None,
                final_score=min(1.0, score / 30.0),
                lexical_score=None,
                semantic_score=None,
            )
            if self._is_low_quality_result(result):
                continue
            ranked.append((score, result))

        if not ranked:
            return []

        ranked.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].markdown_file,
                pair[1].chunk_id,
            )
        )
        deduped: List[SearchResult] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for _, result in ranked:
            key = self._result_dedupe_key(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

    def _lookup_finals_schedule_results(self, limit: int) -> List[SearchResult]:
        rows = self.retriever.store.fetchall(
            """
            SELECT
                dc.id AS chunk_id,
                dc.document_id AS document_id,
                dc.heading_path AS heading_path,
                dc.content AS content,
                d.title AS title,
                d.original_url AS source_url,
                d.file_path AS markdown_file
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE (
                lower(dc.content) LIKE '%final exam%'
                OR lower(dc.content) LIKE '%final exams%'
                OR lower(dc.content) LIKE '%finals week%'
                OR lower(dc.content) LIKE '%exam schedule%'
                OR lower(dc.heading_path) LIKE '%final%'
            )
            AND lower(d.original_url) NOT LIKE '%/admissions/%'
            """
        )
        if not rows:
            return []

        ranked: List[tuple[int, SearchResult]] = []
        for row in rows:
            title = str(row["title"] or "")
            section = str(row["heading_path"] or "")
            content = str(row["content"] or "")
            source_url = str(row["source_url"] or "")
            markdown_file = str(row["markdown_file"] or "")
            full_text = f"{title}\n{section}\n{content}\n{source_url}\n{markdown_file}".lower()
            score = 0
            if any(term in full_text for term in ("final exam", "final exams", "finals week", "exam schedule")):
                score += 12
            if "academic calendar" in full_text:
                score += 10
            if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b", full_text):
                score += 6
            if any(term in full_text for term in ("date", "dates", "deadline", "week")):
                score += 4
            if "scholarship" in full_text or "club and organization" in full_text:
                score -= 6
            if score <= 0:
                continue

            condensed = " ".join(content.split())
            snippet = condensed[:280].strip()
            if len(condensed) > 280:
                snippet += "..."
            result = SearchResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                score=min(1.0, score / 28.0),
                title=title,
                source_url=source_url,
                markdown_file=markdown_file,
                section=section or None,
                snippet=snippet,
                content=content[:2600],
                retrieval_method="targeted_finals_lookup",
                fts_score=None,
                vector_score=None,
                final_score=min(1.0, score / 28.0),
                lexical_score=None,
                semantic_score=None,
            )
            if self._is_low_quality_result(result):
                continue
            if not self._is_finals_schedule_candidate(result, include_content=True):
                continue
            ranked.append((score, result))

        if not ranked:
            return []

        ranked.sort(key=lambda pair: (-pair[0], pair[1].markdown_file, pair[1].chunk_id))
        deduped: List[SearchResult] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for _, result in ranked:
            key = self._result_dedupe_key(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

    def _lookup_financial_aid_results(self, limit: int) -> List[SearchResult]:
        rows = self.retriever.store.fetchall(
            """
            SELECT
                dc.id AS chunk_id,
                dc.document_id AS document_id,
                dc.heading_path AS heading_path,
                dc.content AS content,
                d.title AS title,
                d.original_url AS source_url,
                d.file_path AS markdown_file
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE (
                lower(d.original_url) LIKE '%/financial-aid/resources/%'
                OR lower(d.original_url) LIKE '%/financial-aid/federal-work-study%'
                OR lower(d.original_url) LIKE '%/financial-aid/%/aid-programs%'
                OR lower(d.file_path) LIKE '%financial-aid__resources__%'
                OR lower(d.file_path) LIKE '%financial-aid__federal-work-study%'
                OR lower(d.file_path) LIKE '%aid-programs%'
            )
            AND lower(d.file_path) NOT LIKE '%after-you-apply%'
            AND lower(d.file_path) NOT LIKE '%award-notification%'
            AND lower(d.file_path) NOT LIKE '%course-eligibility%'
            AND lower(d.file_path) NOT LIKE '%getting-your-aid%'
            AND lower(d.file_path) NOT LIKE '%keeping-your-aid%'
            AND lower(d.file_path) NOT LIKE '%costs%'
            AND lower(d.file_path) NOT LIKE '%verification%'
            AND lower(d.file_path) NOT LIKE '%faqs%'
            AND lower(d.file_path) NOT LIKE '%credentials%'
            AND lower(d.file_path) NOT LIKE '%on-campus-employers%'
            AND lower(d.file_path) NOT LIKE '%student-information%'
            AND lower(d.original_url) NOT LIKE '%/aboutcpp/%'
            AND lower(d.original_url) NOT LIKE '%/faqs%'
            AND lower(d.original_url) NOT LIKE '%/verification%'
            AND lower(d.original_url) NOT LIKE '%/credentials/%'
            """
        )
        if not rows:
            return []

        ranked: List[tuple[int, SearchResult]] = []
        for row in rows:
            title = str(row["title"] or "")
            section = str(row["heading_path"] or "")
            content = str(row["content"] or "")
            source_url = str(row["source_url"] or "")
            markdown_file = str(row["markdown_file"] or "")
            full_text = f"{title}\n{section}\n{content}\n{source_url}\n{markdown_file}".lower()
            score = 0
            if "financial aid" in full_text:
                score += 12
            if any(marker in full_text for marker in ("/~financial-aid", "/financial-aid/", "financial-aid__")):
                score += 12
            if any(term in full_text for term in ("resource", "resources", "scholarship", "grant", "loan", "fafsa", "dream act")):
                score += 10
            if any(term in full_text for term in ("application", "deadline", "apply", "eligibility")):
                score += 5
            if "office of the president" in full_text or "/aboutcpp/" in full_text:
                score -= 12
            if "follow us" in full_text:
                score -= 10
            if score <= 0:
                continue

            condensed = " ".join(content.split())
            snippet = condensed[:280].strip()
            if len(condensed) > 280:
                snippet += "..."
            result = SearchResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                score=min(1.0, score / 30.0),
                title=title,
                source_url=source_url,
                markdown_file=markdown_file,
                section=section or None,
                snippet=snippet,
                content=content[:2600],
                retrieval_method="targeted_financial_aid_lookup",
                fts_score=None,
                vector_score=None,
                final_score=min(1.0, score / 30.0),
                lexical_score=None,
                semantic_score=None,
            )
            if self._is_low_quality_result(result):
                continue
            if not self._is_financial_aid_candidate(result, include_content=True):
                continue
            ranked.append((score, result))

        if not ranked:
            return []

        ranked.sort(key=lambda pair: (-pair[0], pair[1].markdown_file, pair[1].chunk_id))
        deduped: List[SearchResult] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for _, result in ranked:
            key = self._result_dedupe_key(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

    def _rewrite_query_for_retrieval(self, user_message: str) -> str:
        query = user_message.strip()
        lowered = query.lower()
        if self._is_freshman_admissions_intent(lowered):
            return f"{query} office of admissions freshmen students requirements checklist"
        if self._is_financial_aid_intent(lowered):
            return f"{query} financial aid resources scholarships grants loans FAFSA deadlines"
        if self._is_finals_schedule_intent(lowered):
            return f"{query} final exam schedule finals week academic calendar dates"
        if ("student health" in lowered or "health services" in lowered) and (
            "where" in lowered or "location" in lowered or "located" in lowered or "address" in lowered
        ):
            return f"{query} student health center location address"
        return query

    def _build_retrieval_debug(
        self,
        *,
        enabled: bool,
        mode: str,
        query_used: str,
        tooling_mode: str,
        used_tool_calls: bool,
        results: List[SearchResult],
    ) -> Optional[RetrievalDebugInfo]:
        if not enabled:
            return None
        return RetrievalDebugInfo(
            enabled=True,
            mode=mode,
            query_used=query_used,
            tooling_mode=tooling_mode,
            used_tool_calls=used_tool_calls,
            top_chunks=[
                RetrievalDebugChunk(
                    chunk_id=item.chunk_id,
                    title=item.title,
                    heading_path=item.section,
                    original_url=item.source_url,
                    fts_score=item.fts_score,
                    vector_score=item.vector_score,
                    final_score=item.final_score if item.final_score is not None else item.score,
                )
                for item in results[:8]
            ],
        )

    def _deterministic_grounded_fallback(self, results: List[SearchResult]) -> str:
        if not self._is_retrieval_confident(results):
            return NOT_FOUND_MESSAGE
        sections: List[str] = []
        seen_signatures: set[str] = set()
        top_score = results[0].final_score if results and results[0].final_score is not None else (results[0].score if results else 0.0)
        relevance_threshold = max(self.min_retrieval_score, float(top_score) * 0.5)

        for item in results[:5]:
            item_score = item.final_score if item.final_score is not None else item.score
            if float(item_score) < relevance_threshold:
                continue
            heading_source = (item.section or item.title or "Relevant Information").strip()
            heading = heading_source.split(" > ")[-1].strip() or heading_source
            raw_text = (item.content or item.snippet or "").strip()
            if not raw_text:
                continue

            plain_text = re.sub(r"```[\s\S]*?```", " ", raw_text)
            plain_text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", plain_text)
            plain_text = re.sub(r"\|[^\n]*\|", " ", plain_text)
            plain_text = re.sub(r"^\s{0,3}#{1,6}\s*", "", plain_text, flags=re.MULTILINE)
            plain_text = re.sub(r"(?<!\n)\s{0,3}#{2,6}\s+", " ", plain_text)
            plain_text = re.sub(r"^\s*[-*+]\s+", "", plain_text, flags=re.MULTILINE)
            plain_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain_text)
            plain_text = re.sub(r"(home page menu|close menu|skip to content)", " ", plain_text, flags=re.IGNORECASE)
            plain_text = " ".join(plain_text.split())
            sentence_candidates = [part.strip() for part in re.split(r"(?<=[.!?])\s+", plain_text) if part.strip()]
            cleaned_sentences: List[str] = []
            for sentence in sentence_candidates:
                normalized_sentence = re.sub(rf"^{re.escape(heading)}\s+", "", sentence, flags=re.IGNORECASE).strip()
                if not normalized_sentence:
                    continue
                if normalized_sentence.lower() == heading.lower():
                    continue
                # Skip dangling fragments so the deterministic fallback
                # doesn't promote lines like "11th-17th" to an answer.
                if not self._is_valid_answer_sentence(normalized_sentence):
                    continue
                cleaned_sentences.append(normalized_sentence)

            if not cleaned_sentences:
                # The chunk had no substantive sentences (e.g., it was just a
                # heading with a boilerplate list). Skip it rather than
                # fabricating around an empty body.
                continue

            summary = " ".join(cleaned_sentences[:2]).strip()
            summary = re.sub(rf"^{re.escape(heading)}\s+", "", summary, flags=re.IGNORECASE).strip()
            summary = re.sub(r"\b([A-Za-z]{2,})\s+\1\b", r"\1", summary, flags=re.IGNORECASE)
            if not summary or summary.lower() == heading.lower():
                continue
            if not self._is_valid_answer_sentence(summary):
                continue
            if len(summary) > 360:
                summary = summary[:360].rsplit(" ", 1)[0].rstrip() + "..."
            if not summary:
                continue

            signature = re.sub(r"\W+", " ", summary.lower()).strip()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            sections.append(f"## {heading}\n{summary}")
            if len(sections) >= 2:
                break

        if not sections:
            return NOT_FOUND_MESSAGE
        return "\n\n".join(sections)

    def _synthesize_financial_aid_answer(self, results: List[SearchResult]) -> Optional[str]:
        """Build a structured answer for 'what financial aid resources are
        available' questions by bucketing retrieved content into Types of Aid
        vs. Support Resources. Returns None when the retrieved chunks do not
        contain any on-topic aid content so the caller can fall back to the
        financial-aid-specific not-found message.
        """
        if not results:
            return None

        type_buckets: Dict[str, List[str]] = {
            "Scholarships": [],
            "Grants": [],
            "Loans": [],
            "Work-Study": [],
            "Other Aid Programs": [],
        }
        support_bucket: List[str] = []
        seen_signatures: set[str] = set()

        def clean(text: str) -> str:
            plain = re.sub(r"```[\s\S]*?```", " ", text)
            plain = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", plain)
            plain = re.sub(r"\|[^\n]*\|", " ", plain)
            plain = re.sub(r"^\s{0,3}#{1,6}\s*", "", plain, flags=re.MULTILINE)
            plain = re.sub(r"(?<!\n)\s{0,3}#{2,6}\s+", " ", plain)
            plain = re.sub(r"^\s*[-*+]\s+", "", plain, flags=re.MULTILINE)
            plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
            plain = re.sub(r"(home page menu|close menu|skip to content|follow us)", " ", plain, flags=re.IGNORECASE)
            # Strip markdown bold/italic marker noise like **foo** or __foo__ so
            # the synthesized answer reads as prose rather than a raw dump.
            plain = re.sub(r"\*{1,3}", "", plain)
            plain = re.sub(r"(?<!\w)_{1,3}(?!\w)", "", plain)
            # Collapse multiple spaces/newlines/soft breaks.
            plain = re.sub(r"\s+", " ", plain)
            return plain.strip()

        for item in results[:6]:
            full = f"{item.title} {item.section or ''} {item.content or ''}".lower()
            # Skip procedural / FAQ / notification content regardless of how
            # it got here.
            if any(
                marker in full
                for marker in (
                    "verification",
                    "notification",
                    "status update",
                    "check your status",
                    "award notification",
                    "after you apply",
                    "course eligibility",
                    "credential students",
                    "satisfactory academic progress appeal",
                )
            ):
                continue

            plain_text = clean(item.content or item.snippet or "")
            if not plain_text:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain_text) if s.strip()]
            substantive = [s for s in sentences if self._is_valid_answer_sentence(s)]
            if not substantive:
                continue
            summary = " ".join(substantive[:2]).strip()
            if len(summary) > 320:
                summary = summary[:320].rsplit(" ", 1)[0].rstrip() + "..."

            signature = re.sub(r"\W+", " ", summary.lower()).strip()
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            bucket_key: Optional[str] = None
            if "scholarship" in full:
                bucket_key = "Scholarships"
            elif "grant" in full and "granted" not in full[:80]:
                bucket_key = "Grants"
            elif "loan" in full:
                bucket_key = "Loans"
            elif "work-study" in full or "work study" in full or "federal work" in full:
                bucket_key = "Work-Study"
            elif any(term in full for term in ("fafsa", "dream act", "aid program", "types of aid", "aid programs")):
                bucket_key = "Other Aid Programs"

            if bucket_key:
                type_buckets[bucket_key].append(summary)
                continue

            if any(
                term in full
                for term in (
                    "office of financial aid",
                    "financial aid office",
                    "financial aid counselor",
                    "counseling",
                    "advising",
                    "appointment",
                    "workshop",
                    "contact",
                    "help",
                    "support",
                )
            ):
                support_bucket.append(summary)

        has_type_content = any(type_buckets[key] for key in type_buckets)
        if not has_type_content and not support_bucket:
            return None

        parts: List[str] = []
        if has_type_content:
            parts.append("## Types of Financial Aid")
            for label, items in type_buckets.items():
                if not items:
                    continue
                parts.append(f"**{label}.** {items[0]}")
        if support_bucket:
            parts.append("## Support Resources")
            parts.append(support_bucket[0])

        if not parts:
            return None
        return "\n\n".join(parts)

    def _record_analytics(
        self,
        *,
        provider: str,
        model: Optional[str],
        event_type: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store.add_analytics_event(
            event_type=event_type,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def _save_turn(
        self,
        *,
        conversation_id: str,
        user_message: str,
        answer: str,
        provider: str,
        model: Optional[str],
        sources: List[SourceItem],
    ) -> None:
        self.store.add_message(conversation_id, "user", user_message, provider=provider, model=model)
        assistant_message_id = self.store.add_message(conversation_id, "assistant", answer, provider=provider, model=model)
        self.store.add_citations(assistant_message_id, [source.model_dump() for source in sources])

    def _build_openai_input(self, conversation_id: str, user_message: str) -> List[Dict[str, Any]]:
        history = self.store.get_recent_messages(conversation_id, limit=12)
        input_items: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]}
        ]
        for item in history:
            input_items.append(
                {
                    "role": item["role"],
                    "content": [{"type": "input_text", "text": item["content"]}],
                }
            )
        input_items.append({"role": "user", "content": [{"type": "input_text", "text": user_message}]})
        return input_items

    def _generate_openai_grounded_answer(
        self,
        *,
        user_message: str,
        context_chunks: List[SearchResult],
        model: str,
    ) -> str:
        if self.openai_client is None:
            return NOT_FOUND_MESSAGE
        context_text = "\n\n".join(
            (
                f"[{idx + 1}] chunk_id={item.chunk_id}\n"
                f"title={item.title}\n"
                f"section={item.section or '(none)'}\n"
                f"url={item.source_url}\n"
                f"content={item.content}"
            )
            for idx, item in enumerate(context_chunks)
        )
        response = self.openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Use only the retrieved corpus context below.\n\n"
                                f"Context:\n{context_text}\n\n"
                                f"Question: {user_message}"
                            ),
                        }
                    ],
                },
            ],
        )
        return (response.output_text or "").strip() or NOT_FOUND_MESSAGE

    def _chat_openai(self, user_message: str, conversation_id: str, model: Optional[str], debug: bool) -> ChatResponse:
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        started = time.time()
        active_model = model or self.openai_model
        dispatcher = ToolDispatcher(
            retriever=self.retriever,
            provider="openai",
            model=None,
            max_results=self.max_tool_results,
        )
        input_items = self._build_openai_input(conversation_id, user_message)
        tool_trace: List[Dict[str, Any]] = []
        last_results: List[SearchResult] = []
        usage_input = None
        usage_output = None
        used_tool_calls = False
        tooling_mode = "openai_tool_calls"

        for _ in range(6):
            response = self.openai_client.responses.create(
                model=active_model,
                input=input_items,
                tools=TOOLS,
            )
            response_items = response.output or []
            usage_input = getattr(response.usage, "input_tokens", None) if getattr(response, "usage", None) else None
            usage_output = getattr(response.usage, "output_tokens", None) if getattr(response, "usage", None) else None
            input_items.extend([item.model_dump() for item in response_items])

            function_calls = [item for item in response_items if item.type == "function_call"]
            if not function_calls:
                answer_text = (response.output_text or "").strip() or NOT_FOUND_MESSAGE
                if not used_tool_calls:
                    tooling_mode = "openai_forced_retrieval_fallback"
                    retrieval_query = self._rewrite_query_for_retrieval(user_message)
                    forced = dispatcher.search_corpus(retrieval_query, self.max_tool_results)
                    last_results = [SearchResult(**item) for item in forced.get("results", [])]
                    last_results = self._rerank_for_query_intent(user_message, last_results)
                    last_results = [item for item in last_results if not self._is_low_quality_result(item)]
                    last_results = self._filter_relevant_results(user_message, last_results)
                    is_fa_intent = self._is_financial_aid_intent(user_message)
                    if is_fa_intent and not self._has_strong_financial_aid_evidence(last_results):
                        last_results = []
                    if last_results and not self._has_query_coverage(user_message, last_results):
                        last_results = []
                    if self._is_retrieval_confident(last_results):
                        if is_fa_intent:
                            synthesized = self._synthesize_financial_aid_answer(last_results)
                            answer_text = synthesized or FINANCIAL_AID_FALLBACK_MESSAGE
                        else:
                            answer_text = self._generate_openai_grounded_answer(
                                user_message=user_message,
                                context_chunks=last_results,
                                model=active_model,
                            )
                    else:
                        answer_text = FINANCIAL_AID_FALLBACK_MESSAGE if is_fa_intent else NOT_FOUND_MESSAGE

                is_fa_intent = self._is_financial_aid_intent(user_message)
                fallback_message = FINANCIAL_AID_FALLBACK_MESSAGE if is_fa_intent else NOT_FOUND_MESSAGE
                sources = self._filter_sources_for_query(
                    query=user_message,
                    sources=self.retriever.to_sources(last_results),
                    selected_results=last_results,
                )
                if not self._is_retrieval_confident(last_results):
                    answer_text = fallback_message
                    sources = []
                if not sources:
                    answer_text = fallback_message
                if not self._is_answer_supported_by_results(user_message, answer_text, last_results):
                    answer_text = fallback_message
                    sources = []

                self._save_turn(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    answer=answer_text,
                    provider="openai",
                    model=active_model,
                    sources=sources,
                )
                elapsed_ms = int((time.time() - started) * 1000)
                total_tokens = (usage_input or 0) + (usage_output or 0) if usage_input is not None else None
                self._record_analytics(
                    provider="openai",
                    model=active_model,
                    event_type="chat_completion",
                    prompt_tokens=usage_input,
                    completion_tokens=usage_output,
                    total_tokens=total_tokens,
                    latency_ms=elapsed_ms,
                    metadata={
                        "conversation_id": conversation_id,
                        "tool_calls": len(tool_trace),
                        "source_count": len(sources),
                        "tooling_mode": tooling_mode,
                    },
                )
                return ChatResponse(
                    answer=answer_text,
                    conversation_id=conversation_id,
                    sources=sources,
                    tool_trace=tool_trace,
                    provider="openai",
                    model=active_model,
                    retrieval_debug=self._build_retrieval_debug(
                        enabled=debug,
                        mode=last_results[0].retrieval_method if last_results else "none",
                        query_used=user_message,
                        tooling_mode=tooling_mode,
                        used_tool_calls=used_tool_calls,
                        results=last_results,
                    ),
                )

            for call in function_calls:
                used_tool_calls = True
                tool_output = dispatcher.dispatch(call.name, call.arguments)
                tool_trace.append({"tool": call.name, "arguments": json.loads(call.arguments or "{}")})
                if call.name == "search_corpus":
                    last_results = [SearchResult(**item) for item in tool_output.get("results", [])]
                    last_results = self._rerank_for_query_intent(user_message, last_results)
                    last_results = [item for item in last_results if not self._is_low_quality_result(item)]
                    last_results = self._filter_relevant_results(user_message, last_results)
                    if self._is_financial_aid_intent(user_message) and not self._has_strong_financial_aid_evidence(last_results):
                        last_results = []
                    if last_results and not self._has_query_coverage(user_message, last_results):
                        last_results = []
                elif call.name == "get_chunk_context":
                    last_results = [SearchResult(**item) for item in tool_output.get("chunks", [])]
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(tool_output),
                    }
                )

        self._save_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            answer=NOT_FOUND_MESSAGE,
            provider="openai",
            model=active_model,
            sources=[],
        )
        elapsed_ms = int((time.time() - started) * 1000)
        self._record_analytics(
            provider="openai",
            model=active_model,
            event_type="chat_fallback",
            prompt_tokens=usage_input,
            completion_tokens=usage_output,
            total_tokens=(usage_input or 0) + (usage_output or 0) if usage_input is not None else None,
            latency_ms=elapsed_ms,
            metadata={"conversation_id": conversation_id, "tool_calls": len(tool_trace)},
        )
        return ChatResponse(
            answer=NOT_FOUND_MESSAGE,
            conversation_id=conversation_id,
            sources=[],
            tool_trace=tool_trace,
            provider="openai",
            model=active_model,
            retrieval_debug=self._build_retrieval_debug(
                enabled=debug,
                mode=last_results[0].retrieval_method if last_results else "none",
                query_used=user_message,
                tooling_mode=tooling_mode,
                used_tool_calls=used_tool_calls,
                results=last_results,
            ),
        )

    def _chat_ollama(self, user_message: str, conversation_id: str, model: Optional[str], debug: bool) -> ChatResponse:
        started = time.time()
        active_model = model or self.ollama_model
        dispatcher = ToolDispatcher(
            retriever=self.retriever,
            provider="ollama",
            model=None,
            max_results=self.max_tool_results,
        )
        tool_trace: List[Dict[str, Any]] = []

        retrieval_query = self._rewrite_query_for_retrieval(user_message)
        search_output = dispatcher.search_corpus(retrieval_query, self.max_tool_results)
        tool_trace.append({"tool": "search_corpus", "arguments": {"query": retrieval_query, "top_k": self.max_tool_results}})
        search_results = [SearchResult(**item) for item in search_output.get("results", [])]
        search_results = self._rerank_for_query_intent(user_message, search_results)
        search_results = [item for item in search_results if not self._is_low_quality_result(item)]
        search_results = self._filter_relevant_results(user_message, search_results)
        if self._is_freshman_admissions_intent(user_message):
            admissions_freshman_results = [
                item
                for item in search_results
                if self._is_freshman_admissions_candidate(item, include_content=True)
            ]
            if admissions_freshman_results:
                search_results = admissions_freshman_results

            has_admissions_candidate = any(
                self._is_freshman_admissions_candidate(item, include_content=False)
                for item in search_results
            )
            if not has_admissions_candidate:
                targeted_query = "office of admissions freshmen students requirements application checklist"
                targeted_results = self.retriever.search_corpus(
                    targeted_query,
                    top_k=max(self.max_tool_results * 4, 24),
                    provider="ollama",
                    embedding_model=None,
                )
                tool_trace.append(
                    {
                        "tool": "search_corpus",
                        "arguments": {"query": targeted_query, "top_k": max(self.max_tool_results * 4, 24)},
                    }
                )
                reranked_targeted = self._rerank_for_query_intent(user_message, targeted_results)
                reranked_targeted = [item for item in reranked_targeted if not self._is_low_quality_result(item)]
                admissions_targeted = [
                    item
                    for item in reranked_targeted
                    if self._is_freshman_admissions_candidate(item, include_content=True)
                ]
                if admissions_targeted:
                    search_results = admissions_targeted[: self.max_tool_results]
                else:
                    fallback_results = self._lookup_freshman_admissions_results(self.max_tool_results)
                    if fallback_results:
                        tool_trace.append(
                            {
                                "tool": "targeted_admissions_lookup",
                                "arguments": {"scope": "admissions/freshmen", "top_k": self.max_tool_results},
                            }
                        )
                        search_results = fallback_results
                    else:
                        search_results = []

        if self._is_financial_aid_intent(user_message):
            financial_aid_results = [
                item
                for item in search_results
                if self._is_financial_aid_candidate(item, include_content=True)
            ]
            if financial_aid_results:
                search_results = financial_aid_results
            else:
                aid_query = "financial aid resources scholarships grants loans FAFSA Dream Act office"
                targeted_aid = self.retriever.search_corpus(
                    aid_query,
                    top_k=max(self.max_tool_results * 4, 24),
                    provider="ollama",
                    embedding_model=None,
                )
                tool_trace.append(
                    {
                        "tool": "search_corpus",
                        "arguments": {"query": aid_query, "top_k": max(self.max_tool_results * 4, 24)},
                    }
                )
                reranked_aid = self._rerank_for_query_intent(user_message, targeted_aid)
                reranked_aid = [item for item in reranked_aid if not self._is_low_quality_result(item)]
                aid_targeted = [
                    item
                    for item in reranked_aid
                    if self._is_financial_aid_candidate(item, include_content=True)
                ]
                if aid_targeted:
                    search_results = aid_targeted[: self.max_tool_results]
                else:
                    fallback_aid = self._lookup_financial_aid_results(self.max_tool_results)
                    if fallback_aid:
                        tool_trace.append(
                            {
                                "tool": "targeted_financial_aid_lookup",
                                "arguments": {"scope": "financial aid resources", "top_k": self.max_tool_results},
                            }
                        )
                        search_results = fallback_aid
                    else:
                        search_results = []

        if self._is_finals_schedule_intent(user_message):
            finals_results = [
                item
                for item in search_results
                if self._is_finals_schedule_candidate(item, include_content=True)
            ]
            if finals_results:
                search_results = finals_results
            else:
                finals_query = "final exam schedule finals week academic calendar date"
                targeted_finals = self.retriever.search_corpus(
                    finals_query,
                    top_k=max(self.max_tool_results * 4, 24),
                    provider="ollama",
                    embedding_model=None,
                )
                tool_trace.append(
                    {
                        "tool": "search_corpus",
                        "arguments": {"query": finals_query, "top_k": max(self.max_tool_results * 4, 24)},
                    }
                )
                reranked_finals = self._rerank_for_query_intent(user_message, targeted_finals)
                reranked_finals = [item for item in reranked_finals if not self._is_low_quality_result(item)]
                finals_targeted = [
                    item
                    for item in reranked_finals
                    if self._is_finals_schedule_candidate(item, include_content=True)
                ]
                if finals_targeted:
                    search_results = finals_targeted[: self.max_tool_results]
                else:
                    fallback_finals = self._lookup_finals_schedule_results(self.max_tool_results)
                    if fallback_finals:
                        tool_trace.append(
                            {
                                "tool": "targeted_finals_lookup",
                                "arguments": {"scope": "finals schedule", "top_k": self.max_tool_results},
                            }
                        )
                        search_results = fallback_finals
                    else:
                        search_results = []
        search_results = self._filter_relevant_results(user_message, search_results)
        is_fa_intent = self._is_financial_aid_intent(user_message)
        fallback_message = FINANCIAL_AID_FALLBACK_MESSAGE if is_fa_intent else NOT_FOUND_MESSAGE
        if is_fa_intent and not self._has_strong_financial_aid_evidence(search_results):
            search_results = []
        if search_results and not self._has_query_coverage(user_message, search_results):
            search_results = []
        if not self._is_retrieval_confident(search_results):
            answer = fallback_message
            self._save_turn(
                conversation_id=conversation_id,
                user_message=user_message,
                answer=answer,
                provider="ollama",
                model=active_model,
                sources=[],
            )
            elapsed_ms = int((time.time() - started) * 1000)
            self._record_analytics(
                provider="ollama",
                model=active_model,
                event_type="chat_completion",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=elapsed_ms,
                metadata={
                    "conversation_id": conversation_id,
                    "tool_calls": len(tool_trace),
                    "source_count": 0,
                    "tooling_mode": "ollama_deterministic_fallback",
                },
            )
            return ChatResponse(
                answer=answer,
                conversation_id=conversation_id,
                sources=[],
                tool_trace=tool_trace,
                provider="ollama",
                model=active_model,
                retrieval_debug=self._build_retrieval_debug(
                    enabled=debug,
                    mode=search_results[0].retrieval_method if search_results else "none",
                    query_used=user_message,
                    tooling_mode="ollama_deterministic_fallback",
                    used_tool_calls=False,
                    results=search_results,
                ),
            )

        top_score = search_results[0].final_score if search_results[0].final_score is not None else search_results[0].score
        relevance_threshold = max(self.min_retrieval_score, float(top_score) * 0.5)
        selected_results = [
            result
            for result in search_results
            if float(result.final_score if result.final_score is not None else result.score) >= relevance_threshold
        ][:3]
        if not selected_results:
            selected_results = search_results[:2]

        deduped_selected: List[SearchResult] = []
        seen_result_keys: set[tuple[str, str, str]] = set()
        for result in selected_results:
            key = self._result_dedupe_key(result)
            if key in seen_result_keys:
                continue
            seen_result_keys.add(key)
            deduped_selected.append(result)
        if deduped_selected:
            selected_results = deduped_selected

        chunk_ids = [result.chunk_id for result in selected_results]
        context_output = dispatcher.get_chunk_context(chunk_ids)
        tool_trace.append({"tool": "get_chunk_context", "arguments": {"chunk_ids": chunk_ids}})
        sources_output = dispatcher.list_sources_for_answer(chunk_ids)
        tool_trace.append({"tool": "list_sources_for_answer", "arguments": {"chunk_ids": chunk_ids}})
        sources = [SourceItem(**item) for item in sources_output.get("sources", [])]
        sources = self._filter_sources_for_query(query=user_message, sources=sources, selected_results=selected_results)

        history = self.store.get_recent_messages(conversation_id, limit=8)
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)
        context_items = context_output.get("chunks", [])
        context_text = "\n\n".join(
            (
                f"[{idx + 1}] chunk_id={item.get('chunk_id')}\n"
                f"title={item.get('title')}\n"
                f"section={item.get('section') or '(none)'}\n"
                f"url={item.get('source_url')}\n"
                f"content={item.get('content')}"
            )
            for idx, item in enumerate(context_items)
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Conversation history:\n"
            f"{history_text or '(none)'}\n\n"
            "Retrieved corpus context:\n"
            f"{context_text}\n\n"
            f"User question:\n{user_message}"
        )

        payload = json.dumps(
            {
                "model": active_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(f"{self.ollama_host}/api/chat", data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        tooling_mode = "ollama_deterministic_fallback"
        try:
            with urlopen(request, timeout=90) as raw:
                response_payload = json.loads(raw.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            response_payload = {}
            tooling_mode = "forced_retrieval_fallback"

        if is_fa_intent:
            synthesized_fa = self._synthesize_financial_aid_answer(selected_results or search_results)
            if synthesized_fa:
                answer = synthesized_fa
            else:
                answer = (
                    response_payload.get("message", {}).get("content", "").strip()
                    or self._deterministic_grounded_fallback(search_results)
                )
                # If the model/fallback didn't produce a structured aid answer,
                # prefer the specific FA fallback over dumping procedural text.
                if answer.strip().lower() == NOT_FOUND_MESSAGE.lower():
                    answer = FINANCIAL_AID_FALLBACK_MESSAGE
        else:
            answer = (
                response_payload.get("message", {}).get("content", "").strip()
                or self._deterministic_grounded_fallback(search_results)
            )
        if not sources:
            answer = fallback_message
        if not self._is_answer_supported_by_results(user_message, answer, selected_results):
            # Allow the synthesized FA answer to pass the support check even
            # when its structured section headings don't share enough tokens
            # with a single chunk -- the synthesizer already constrains content
            # to retrieved sentences.
            if not (is_fa_intent and answer and answer != fallback_message and answer.startswith("## ")):
                answer = fallback_message
                sources = []

        self._save_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            answer=answer,
            provider="ollama",
            model=active_model,
            sources=sources,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        prompt_tokens = response_payload.get("prompt_eval_count")
        completion_tokens = response_payload.get("eval_count")
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens is not None else None
        self._record_analytics(
            provider="ollama",
            model=active_model,
            event_type="chat_completion",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=elapsed_ms,
            metadata={
                "conversation_id": conversation_id,
                "tool_calls": len(tool_trace),
                "source_count": len(sources),
                "tooling_mode": tooling_mode,
            },
        )
        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            tool_trace=tool_trace,
            provider="ollama",
            model=active_model,
            retrieval_debug=self._build_retrieval_debug(
                enabled=debug,
                mode=search_results[0].retrieval_method if search_results else "none",
                query_used=user_message,
                tooling_mode=tooling_mode,
                used_tool_calls=False,
                results=search_results,
            ),
        )

    def chat(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        debug: bool = False,
    ) -> ChatResponse:
        selected_provider = provider.strip().lower() if provider else "openai"
        if selected_provider not in {"openai", "ollama"}:
            selected_provider = "openai"
        if selected_provider == "openai" and self.openai_client is None:
            selected_provider = "ollama"

        conversation_id = self.store.ensure_conversation(
            conversation_id=conversation_id,
            title=user_message[:80],
        )
        if selected_provider == "openai":
            return self._chat_openai(user_message, conversation_id, model, debug)
        return self._chat_ollama(user_message, conversation_id, model, debug)
