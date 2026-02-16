"""Subject Intelligence Layer (SIL) - Factual grounding through query-based MAV.

This module implements the query-based MAV (Multi-Agent Verification) architecture:
1. Each model independently researches the subject and generates neutral factual queries
2. All queries are pooled and deduplicated
3. Each model independently answers ALL pooled queries
4. Each model judges agreement on others' answers (LLM-based consensus with point voting)
5. Verified answers are classified into categories (characteristics, positives, negatives, use_cases)
"""

from dataclasses import dataclass, field
from math import ceil
from typing import Optional
import asyncio
import json
import logging
import re

from cera.prompts import load_and_format

logger = logging.getLogger(__name__)


@dataclass
class MAVConfig:
    """Multi-Agent Verification configuration."""

    enabled: bool = True
    models: list[str] = None  # N models for cross-validation (minimum 2)
    similarity_threshold: float = 0.75  # Query deduplication threshold
    answer_threshold: float = 0.80  # Legacy: embedding similarity (used in fallback dedup)
    max_queries: int = 30  # Soft cap on pooled queries after dedup
    min_verification_rate: float = 0.30  # Minimum consensus rate before fallback

    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass
class SubjectContext:
    """Context document containing subject intelligence."""

    subject: str
    features: list[str]  # Characteristics/specs/properties
    pros: list[str]  # Advantages/praised aspects
    cons: list[str]  # Disadvantages/complaints
    use_cases: list[str]
    availability: Optional[str] = None
    mav_verified: bool = False  # Whether MAV verification was applied
    search_sources: list[str] = field(default_factory=list)  # URLs of sources used


@dataclass
class SubjectUnderstanding:
    """Result from a model's understanding of the subject."""

    model: str
    subject_type: str
    relevant_aspects: list[str]
    search_queries: list[str]
    raw_response: str = ""


@dataclass
class FactualQuery:
    """A single neutral factual query about the subject."""

    id: str  # Unique ID (e.g., "q1", "q2", ...)
    query: str  # The question text
    source_model: str  # Which model generated this query
    deduplicated_from: list[str] = field(default_factory=list)  # IDs of queries merged into this
    overlap_count: int = 1  # How many models generated a similar query


@dataclass
class QueryAnswer:
    """A single model's answer to a pooled query."""

    query_id: str  # References FactualQuery.id
    model: str  # Which model answered
    response: str  # The answer text
    confidence: str  # "high", "medium", "low"


@dataclass
class QueryConsensusResult:
    """Consensus result for a single query."""

    query_id: str
    query: str
    answers: list[QueryAnswer]  # All model answers
    consensus_reached: bool  # Whether sufficient mutual agreement exists
    consensus_answer: Optional[str]  # The agreed-upon answer (most complete)
    agreeing_models: list[str]  # Models whose answers passed
    pairwise_similarities: dict[str, float]  # Legacy: kept for backward compat (empty for new runs)
    agreement_count: int  # How many models' answers passed
    # LLM-judged consensus fields (new)
    agreement_votes: dict[str, list[str]] = field(default_factory=dict)  # judge -> [models they agreed with]
    total_points: int = 0  # Aggregated agreement edges across all judges
    points_by_source: dict[str, int] = field(default_factory=dict)  # model -> incoming vote count


@dataclass
class MAVQueryPoolReport:
    """Report of the full query-based MAV process."""

    total_queries_generated: int = 0  # Across all models before dedup
    queries_after_dedup: int = 0  # After semantic deduplication
    queries_with_consensus: int = 0  # How many reached agreement
    queries_without_consensus: int = 0  # How many did not
    per_query_results: list[QueryConsensusResult] = field(default_factory=list)
    threshold_used: float = 0.80  # Legacy: embedding threshold
    used_fallback: bool = False


@dataclass
class MAVModelData:
    """Raw data from a single model's MAV research."""

    model: str
    sanitized_model: str  # For folder naming (e.g., "anthropic-claude-sonnet-4")
    understanding: Optional[SubjectUnderstanding] = None
    queries_generated: list[str] = field(default_factory=list)  # Queries this model generated
    answers: list[QueryAnswer] = field(default_factory=list)  # This model's answers to pooled queries
    search_content: str = ""  # Raw search results used
    error: Optional[str] = None


@dataclass
class MAVResult:
    """Complete MAV result including raw data from all models."""

    context: SubjectContext
    model_data: list[MAVModelData] = field(default_factory=list)
    total_facts_extracted: int = 0
    facts_verified: int = 0
    facts_rejected: int = 0
    query_pool_report: Optional[MAVQueryPoolReport] = None
    verified_facts: Optional[dict] = None  # Entity-clustered facts (verified-facts.json content)


class SubjectIntelligenceLayer:
    """
    Subject Intelligence Layer (SIL) with Query-Based MAV.

    Provides factual grounding through a 4-round protocol:
    1. Each MAV model independently researches and generates neutral factual queries
    2. All queries are pooled, deduplicated, and capped
    3. Each model independently answers ALL pooled queries (with web search)
    4. Per-query 2/3 majority voting achieves consensus
    5. Verified answers are classified into categories

    Prompts loaded from cli/cera/prompts/sil/:
    - understand.md: Subject understanding (for initial research)
    - search.md: Web search extraction (for research context)
    - extract.md: Extraction from Tavily results (fallback)
    - generate_queries.md: Neutral factual query generation
    - answer_queries.md: Query answering with web search
    - classify_facts.md: Post-consensus categorization
    """

    def __init__(
        self,
        api_key: str,
        mav_config: Optional[MAVConfig] = None,
        tavily_api_key: Optional[str] = None,
        usage_tracker=None,
        log_callback=None,
    ):
        self.api_key = api_key
        self.mav_config = mav_config or MAVConfig(enabled=False)
        self.tavily_api_key = tavily_api_key
        self.usage_tracker = usage_tracker
        self._similarity_model = None  # Lazy-loaded SentenceTransformer
        self._log = log_callback  # Optional async callback: async (level, phase, message, progress) -> None

    async def _emit_log(self, level: str, phase: str, message: str, progress: int = None):
        """Emit a log message via the async callback if configured.

        Args:
            level: Log level (INFO, WARN, ERROR)
            phase: Current phase (SIL, MAV, etc.)
            message: Log message
            progress: Optional progress percentage (0-100) for composition phase
        """
        # Always print to stdout for Docker visibility
        prog_str = f" ({progress}%)" if progress is not None else ""
        print(f"[{phase}] {message}{prog_str}", flush=True)

        if self._log:
            try:
                await self._log(level, phase, message, progress)
            except TypeError:
                # Fallback for callbacks that don't accept progress parameter
                try:
                    await self._log(level, phase, message)
                except Exception:
                    pass
            except Exception:
                pass  # Silently ignore logging errors

    def _sanitize_model_name(self, model: str) -> str:
        """Sanitize model name for use as folder name."""
        sanitized = model.lower()
        sanitized = re.sub(r'[^a-z0-9]+', '-', sanitized)
        sanitized = re.sub(r'^-|-$', '', sanitized)
        return sanitized

    def _supports_online(self, model: str) -> bool:
        """Check if a model supports web search via :online suffix or native search."""
        from cera.llm.openrouter import supports_online_search
        return supports_online_search(model)

    def _get_search_model(self, model: str) -> Optional[str]:
        """Get the model ID configured for web search."""
        from cera.llm.openrouter import get_search_model_id
        return get_search_model_id(model)

    def _get_similarity_model(self):
        """Lazy-load the SentenceTransformer model (shared across all comparisons)."""
        if self._similarity_model is None:
            import os
            import sys
            import logging
            import warnings
            from io import StringIO

            # Suppress all progress bars and verbose logging
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("safetensors").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

            try:
                from sentence_transformers import SentenceTransformer
                import torch

                # Auto-detect GPU availability
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Temporarily redirect stderr to suppress progress bar output
                old_stderr = sys.stderr
                sys.stderr = StringIO()
                try:
                    # Try to load from cache without network check
                    try:
                        self._similarity_model = SentenceTransformer(
                            "all-MiniLM-L6-v2", device=device, local_files_only=True
                        )
                    except Exception:
                        # Fallback: download if not cached
                        self._similarity_model = SentenceTransformer(
                            "all-MiniLM-L6-v2", device=device
                        )
                finally:
                    sys.stderr = old_stderr
            except ImportError:
                self._similarity_model = None
        return self._similarity_model

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        model = self._get_similarity_model()
        if model is not None:
            from sentence_transformers import util
            embeddings = model.encode([text1, text2], show_progress_bar=False)
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity[0][0])
        else:
            # Fallback: Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0

    # ─── Round 1: Research + Query Generation ───────────────────────────

    async def _research_subject(
        self,
        model: str,
        subject: str,
        additional_context: Optional[str] = None,
    ) -> tuple[SubjectUnderstanding, str]:
        """
        Research the subject using web search and return understanding + search content.

        Args:
            model: Model ID to use
            subject: Subject being researched
            additional_context: Optional extra context about the subject

        Returns:
            Tuple of (SubjectUnderstanding, raw_search_content)
        """
        from cera.llm.openrouter import OpenRouterClient

        # Step 1: Understand the subject type
        prompt = load_and_format("sil", "understand", subject=subject)

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.research") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
            )

        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")

            understanding = SubjectUnderstanding(
                model=model,
                subject_type=data.get("subject_type", "general"),
                relevant_aspects=data.get("relevant_aspects", []),
                search_queries=data.get("search_queries", [f"{subject} reviews"]),
                raw_response=response,
            )
        except Exception:
            understanding = SubjectUnderstanding(
                model=model,
                subject_type="general",
                relevant_aspects=[],
                search_queries=[
                    f"{subject} features",
                    f"{subject} reviews pros cons",
                    f"{subject} specifications",
                ],
                raw_response=response,
            )

        # Step 2: Perform web search to gather research context
        search_model = self._get_search_model(model)
        raw_search_content = ""

        if search_model:
            queries_str = ", ".join(understanding.search_queries)
            search_prompt = load_and_format("sil", "search", subject=subject, queries=queries_str)

            async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.search") as client:
                search_response = await client.chat(
                    messages=[{"role": "user", "content": search_prompt}],
                    model=search_model,
                    temperature=0.0,
                )
            raw_search_content = f"[Model used native web search]\n\nQueries: {queries_str}\n\nResponse:\n{search_response}"
        else:
            search_content, sources = await self._search_with_tavily(
                understanding.search_queries
            )
            raw_search_content = search_content

        return understanding, raw_search_content

    async def _generate_factual_queries(
        self,
        model: str,
        subject: str,
        search_content: str,
        additional_context: Optional[str] = None,
    ) -> list[str]:
        """
        Generate neutral factual queries based on research.

        Args:
            model: Model ID to use
            subject: Subject being researched
            search_content: Research context from web search
            additional_context: Optional extra context about the subject

        Returns:
            List of query strings
        """
        from cera.llm.openrouter import OpenRouterClient

        additional_context_block = ""
        if additional_context:
            additional_context_block = f"\n\nAdditional context provided by the user:\n{additional_context}\n"

        research_context_block = ""
        if search_content:
            research_context_block = f"\n\nResearch context from web search:\n{search_content[:10000]}\n"

        prompt = load_and_format(
            "sil", "generate_queries",
            subject=subject,
            additional_context_block=additional_context_block,
            research_context_block=research_context_block,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.queries") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
            )

        # Parse JSON response
        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")

            queries = data.get("queries", [])
            # Ensure all queries are strings
            return [str(q) for q in queries if q]
        except Exception:
            # Try to extract queries from plain text (one per line)
            lines = [line.strip().strip("-•*").strip() for line in response.split("\n")]
            return [line for line in lines if line and "?" in line]

    # ─── Round 2: Query Pooling + Deduplication ─────────────────────────

    async def _deduplicate_queries(
        self,
        all_queries: list[tuple[str, str]],  # (query_text, source_model)
        threshold: float,
        max_queries: int,
    ) -> list[FactualQuery]:
        """
        Deduplicate and cap queries from all models.

        Uses batch encoding for efficiency: encodes all queries in one forward pass,
        then uses vectorized cosine similarity for O(n) neural inferences instead of O(n²).

        Args:
            all_queries: List of (query_text, source_model) tuples
            threshold: Semantic similarity threshold for deduplication
            max_queries: Maximum number of queries to keep

        Returns:
            List of deduplicated FactualQuery objects with unique IDs
        """
        if not all_queries:
            return []

        total_queries = len(all_queries)
        await self._emit_log("INFO", "SIL", f"Deduplicating {total_queries} queries...")

        # Get the model for batch encoding
        model = self._get_similarity_model()

        if model is not None:
            # OPTIMIZED: Batch encode all queries at once
            await self._emit_log("INFO", "SIL", "Encoding queries (batch)...")
            query_texts = [q[0] for q in all_queries]
            embeddings = model.encode(query_texts, show_progress_bar=False, convert_to_numpy=True)

            # Import numpy for vectorized operations
            import numpy as np
            from numpy.linalg import norm

            # Normalize embeddings for cosine similarity
            norms = norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized = embeddings / norms

            await self._emit_log("INFO", "SIL", "Computing similarities...")

            # Greedy deduplication using precomputed embeddings
            pooled: list[FactualQuery] = []
            pooled_indices: list[int] = []  # Track which original indices are in pooled
            query_id_counter = 1

            for idx, (query_text, source_model) in enumerate(all_queries):
                is_duplicate = False

                if pooled_indices:
                    # Compute cosine similarity with all pooled queries at once (vectorized)
                    current_emb = normalized[idx]
                    pooled_embs = normalized[pooled_indices]
                    similarities = np.dot(pooled_embs, current_emb)

                    # Find if any similarity exceeds threshold
                    max_sim_idx = np.argmax(similarities)
                    if similarities[max_sim_idx] >= threshold:
                        # Merge with the most similar existing query
                        pooled[max_sim_idx].overlap_count += 1
                        pooled[max_sim_idx].deduplicated_from.append(f"q{query_id_counter}")
                        is_duplicate = True

                if not is_duplicate:
                    pooled.append(FactualQuery(
                        id=f"q{query_id_counter}",
                        query=query_text,
                        source_model=source_model,
                        overlap_count=1,
                    ))
                    pooled_indices.append(idx)

                query_id_counter += 1

            await self._emit_log("INFO", "SIL", f"Deduplication complete: {len(pooled)} unique queries")

        else:
            # Fallback: no model available, use simple exact matching
            await self._emit_log("WARN", "SIL", "No similarity model available, using exact matching")
            pooled: list[FactualQuery] = []
            seen_queries: set[str] = set()
            query_id_counter = 1

            for query_text, source_model in all_queries:
                normalized_text = query_text.lower().strip()
                if normalized_text not in seen_queries:
                    seen_queries.add(normalized_text)
                    pooled.append(FactualQuery(
                        id=f"q{query_id_counter}",
                        query=query_text,
                        source_model=source_model,
                        overlap_count=1,
                    ))
                else:
                    # Find and update existing
                    for existing in pooled:
                        if existing.query.lower().strip() == normalized_text:
                            existing.overlap_count += 1
                            existing.deduplicated_from.append(f"q{query_id_counter}")
                            break
                query_id_counter += 1

        # Apply soft cap: prioritize queries with higher overlap_count
        if len(pooled) > max_queries:
            # Sort by overlap count (descending) then by original order
            pooled.sort(key=lambda q: q.overlap_count, reverse=True)
            pooled = pooled[:max_queries]
            # Re-assign IDs after sorting
            for i, q in enumerate(pooled):
                q.id = f"q{i + 1}"

        return pooled

    # ─── Round 3: Query Answering ───────────────────────────────────────

    async def _answer_queries(
        self,
        model: str,
        subject: str,
        queries: list[FactualQuery],
        research_context: str,
        additional_context: Optional[str] = None,
    ) -> list[QueryAnswer]:
        """
        Have a model answer all pooled queries independently.

        Uses web search if available, otherwise uses research context.

        Args:
            model: Model ID to use
            subject: Subject being researched
            queries: List of pooled queries to answer
            research_context: Research context from Round 1
            additional_context: Optional extra context about the subject

        Returns:
            List of QueryAnswer objects
        """
        from cera.llm.openrouter import OpenRouterClient

        additional_context_block = ""
        if additional_context:
            additional_context_block = f"\nAdditional context provided by the user:\n{additional_context}\n"

        research_context_block = ""
        if research_context:
            research_context_block = f"\nResearch context from web search:\n{research_context[:10000]}\n"

        # Format queries as JSON for the prompt
        queries_json = json.dumps(
            [{"query_id": q.id, "query": q.query} for q in queries],
            indent=2,
        )

        prompt = load_and_format(
            "sil", "answer_queries",
            subject=subject,
            additional_context_block=additional_context_block,
            research_context_block=research_context_block,
            queries_json=queries_json,
        )

        # Round 3 does NOT need web search — models already have research_context
        # from Round 1 injected in the prompt. Using :online here is redundant
        # and can cause JSON parse failures (e.g., Opus with :online suffix).
        use_model = model

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.answers") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=use_model,
                temperature=0.0,
            )

        # Parse JSON response using robust extractor
        answers = []
        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")

            raw_answers = data.get("answers", [])
            for ans in raw_answers:
                answers.append(QueryAnswer(
                    query_id=ans.get("query_id", ""),
                    model=model,
                    response=ans.get("response", "No information available"),
                    confidence=ans.get("confidence", "low"),
                ))
        except (json.JSONDecodeError, Exception) as e:
            # Log the raw response for debugging
            logger.warning(
                f"MAV Round 3: Failed to parse JSON from {model} "
                f"(error: {e}). Response preview: {response[:500]}"
            )
            # If parsing fails, create descriptive fallback answers
            for q in queries:
                answers.append(QueryAnswer(
                    query_id=q.id,
                    model=model,
                    response=f"Unable to answer: {q.query}",
                    confidence="low",
                ))

        return answers

    # ─── Round 4: LLM-Judged Consensus ──────────────────────────────────

    async def _judge_agreement(
        self,
        judge_model: str,
        subject: str,
        topics: list[dict],
    ) -> dict[str, dict[str, int]]:
        """
        Ask a single model to judge agreement between all models' answers.

        Args:
            judge_model: The model acting as judge
            subject: The subject being researched
            topics: List of topic dicts with all answers

        Returns:
            Dict mapping query_id -> {model_name: 0 or 1}
        """
        from cera.llm.openrouter import OpenRouterClient

        prompt = load_and_format(
            "sil", "judge_agreement",
            subject=subject,
            judge_model=judge_model,
            topics_json=json.dumps(topics, indent=2, ensure_ascii=False),
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.judge") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=judge_model,
                temperature=1.0,
                max_tokens=16384,
                reasoning={"effort": "high"},
            )

        # Parse response
        judgments: dict[str, dict[str, int]] = {}
        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")

            for item in data.get("judgments", []):
                query_id = item.get("query_id", "")
                scores = item.get("scores", {})
                # Validate: remove self-scores and normalize to 0/1
                clean_scores = {}
                for model_name, score in scores.items():
                    if model_name != judge_model:
                        clean_scores[model_name] = 1 if score == 1 else 0
                judgments[query_id] = clean_scores
        except Exception:
            logger.warning(f"MAV Round 4: Failed to parse judgments from {judge_model}. Response preview: {response[:300]}")
            pass  # Return empty judgments → zero votes from this model

        return judgments

    def _compute_llm_consensus(
        self,
        query: FactualQuery,
        answers: list[QueryAnswer],
        all_judgments: dict[str, dict[str, dict[str, int]]],
        n_models: int,
    ) -> QueryConsensusResult:
        """
        Determine consensus for a single query using LLM judgment votes.

        Uses the >=2 points from >=2 different sources rule:
        A model's answer passes if it received agreement votes (score=1)
        from at least ceil(2/3 * (N-1)) different source models.

        Args:
            query: The query being evaluated
            answers: All model answers for this query
            all_judgments: judge_model -> {query_id -> {model: 0/1}}
            n_models: Total number of responding models

        Returns:
            QueryConsensusResult with consensus details
        """
        valid_answers = answers

        if len(valid_answers) < 2:
            return QueryConsensusResult(
                query_id=query.id,
                query=query.query,
                answers=answers,
                consensus_reached=False,
                consensus_answer=None,
                agreeing_models=[],
                pairwise_similarities={},
                agreement_count=0,
                agreement_votes={},
                total_points=0,
                points_by_source={},
            )

        answer_models = {a.model for a in valid_answers}

        # Collect agreement votes per judge for this query
        agreement_votes: dict[str, list[str]] = {}  # judge -> [models they gave 1 to]
        incoming_votes: dict[str, set[str]] = {m: set() for m in answer_models}

        for judge_model, query_judgments in all_judgments.items():
            if judge_model not in answer_models:
                continue
            scores = query_judgments.get(query.id, {})
            agreed_with = []
            for target_model, score in scores.items():
                if target_model in answer_models and target_model != judge_model and score == 1:
                    agreed_with.append(target_model)
                    incoming_votes[target_model].add(judge_model)
            agreement_votes[judge_model] = agreed_with

        # Points per model = number of distinct incoming agreement votes
        points_by_source = {m: len(incoming_votes[m]) for m in answer_models}
        total_points = sum(points_by_source.values())

        # Mutual agreement rule: answer passes if it received votes from
        # >= ceil(2/3 * (N-1)) different models (with N=3, needs >=2 sources)
        min_sources = max(1, ceil(2 / 3 * (len(answer_models) - 1)))
        passing_models = [m for m in answer_models if points_by_source[m] >= min_sources]

        consensus_reached = len(passing_models) >= 1

        # Select canonical answer: passing model with most incoming votes (tiebreak: longest)
        consensus_answer = None
        agreeing_models = []
        if consensus_reached:
            agreeing_models = passing_models
            best = max(
                [a for a in valid_answers if a.model in passing_models],
                key=lambda a: (points_by_source[a.model], len(a.response.strip()))
            )
            consensus_answer = best.response

        return QueryConsensusResult(
            query_id=query.id,
            query=query.query,
            answers=answers,
            consensus_reached=consensus_reached,
            consensus_answer=consensus_answer,
            agreeing_models=agreeing_models,
            pairwise_similarities={},
            agreement_count=len(passing_models),
            agreement_votes=agreement_votes,
            total_points=total_points,
            points_by_source=points_by_source,
        )

    # ─── Round 4 Legacy: Embedding-Based Consensus (fallback) ─────────

    def _compute_query_consensus(
        self,
        query: FactualQuery,
        answers: list[QueryAnswer],
        threshold: float,
        n_models: int,
    ) -> QueryConsensusResult:
        """
        Determine consensus for a single query across all model answers.

        Args:
            query: The query being evaluated
            answers: All model answers for this query
            threshold: Semantic similarity threshold for agreement
            n_models: Total number of models (for min_agreement calculation)

        Returns:
            QueryConsensusResult with consensus details
        """
        valid_answers = answers
        effective_n = len(valid_answers)

        min_agreement = ceil(2 / 3 * effective_n) if effective_n >= 2 else effective_n

        # Compute pairwise similarities
        pairwise_similarities: dict[str, float] = {}
        for i, a1 in enumerate(valid_answers):
            for j, a2 in enumerate(valid_answers):
                if i >= j:
                    continue
                sim = self._compute_similarity(a1.response, a2.response)
                key = f"{self._sanitize_model_name(a1.model)}-{self._sanitize_model_name(a2.model)}"
                pairwise_similarities[key] = round(sim, 4)

        # Find the largest agreeing group
        if effective_n < 2:
            # Can't have consensus with fewer than 2 valid answers
            return QueryConsensusResult(
                query_id=query.id,
                query=query.query,
                answers=answers,
                consensus_reached=False,
                consensus_answer=None,
                agreeing_models=[],
                pairwise_similarities=pairwise_similarities,
                agreement_count=effective_n,
            )

        # Build agreement groups: find pairs that agree, then find largest clique
        agreeing_pairs: list[tuple[int, int]] = []
        for i, a1 in enumerate(valid_answers):
            for j, a2 in enumerate(valid_answers):
                if i >= j:
                    continue
                sim = self._compute_similarity(a1.response, a2.response)
                if sim >= threshold:
                    agreeing_pairs.append((i, j))

        # Find largest group where all members agree with each other
        best_group: list[int] = []
        for i in range(effective_n):
            group = [i]
            for j in range(effective_n):
                if j == i:
                    continue
                # Check if j agrees with all current group members
                agrees_with_all = True
                for member in group:
                    pair = tuple(sorted([j, member]))
                    if pair not in [(a, b) for a, b in agreeing_pairs]:
                        agrees_with_all = False
                        break
                if agrees_with_all:
                    group.append(j)
            if len(group) > len(best_group):
                best_group = group

        consensus_reached = len(best_group) >= min_agreement

        # Select canonical answer: most complete (longest substantive) from agreeing group
        consensus_answer = None
        agreeing_models = []
        if consensus_reached and best_group:
            agreeing_answers = [valid_answers[i] for i in best_group]
            agreeing_models = [a.model for a in agreeing_answers]
            # Pick the longest response (most information units)
            canonical = max(agreeing_answers, key=lambda a: len(a.response.strip()))
            consensus_answer = canonical.response

        return QueryConsensusResult(
            query_id=query.id,
            query=query.query,
            answers=answers,
            consensus_reached=consensus_reached,
            consensus_answer=consensus_answer,
            agreeing_models=agreeing_models,
            pairwise_similarities=pairwise_similarities,
            agreement_count=len(best_group),
        )

    # ─── Post-Consensus: Classification ─────────────────────────────────

    async def _classify_verified_answers(
        self,
        verified_results: list[QueryConsensusResult],
        subject: str,
    ) -> dict[str, list[str]]:
        """
        Classify verified consensus answers into categories.

        Args:
            verified_results: List of consensus results that reached agreement
            subject: The subject being researched

        Returns:
            Dict with keys: characteristics, positives, negatives, use_cases
        """
        from cera.llm.openrouter import OpenRouterClient

        if not verified_results:
            return {
                "characteristics": [],
                "positives": [],
                "negatives": [],
                "use_cases": [],
            }

        # Build facts list from verified answers
        facts_list = []
        for result in verified_results:
            if result.consensus_answer:
                facts_list.append({
                    "query": result.query,
                    "answer": result.consensus_answer,
                })

        facts_json = json.dumps(facts_list, indent=2)

        prompt = load_and_format(
            "sil", "classify_facts",
            subject=subject,
            facts_json=facts_json,
        )

        # Use first available model for classification
        classify_model = self.mav_config.models[0] if self.mav_config.models else "anthropic/claude-sonnet-4"

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.classify") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=classify_model,
                temperature=0.0,
            )

        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")

            return {
                "characteristics": data.get("characteristics", []),
                "positives": data.get("positives", []),
                "negatives": data.get("negatives", []),
                "use_cases": data.get("use_cases", []),
            }
        except Exception:
            # Fallback: put all facts in characteristics
            return {
                "characteristics": [r.consensus_answer for r in verified_results if r.consensus_answer],
                "positives": [],
                "negatives": [],
                "use_cases": [],
            }

    # ─── Round 5: Entity Clustering (Multi-Model Consensus) ────────────

    async def _cluster_facts_by_entity(
        self,
        model: str,
        classified: dict[str, list[str]],
        aspect_categories: list[str],
        subject: str,
    ) -> dict:
        """
        Round 5a: Single model independently clusters verified facts into entities.

        Args:
            model: The LLM model to use for clustering
            classified: Dict with keys: characteristics, positives, negatives, use_cases
            aspect_categories: List of aspect category strings (e.g., ["DISPLAY#GENERAL", ...])
            subject: The subject being researched

        Returns:
            Dict with "entities" array following the verified-facts schema
        """
        from cera.llm.openrouter import OpenRouterClient

        # Build facts list from classified data
        all_facts = []
        for category, facts in classified.items():
            for fact in facts:
                all_facts.append({"category": category, "text": fact})

        if not all_facts:
            return {"entities": []}

        # Build output schema example
        output_schema = json.dumps({
            "entities": [
                {
                    "id": "entity-01",
                    "name": "Entity Name",
                    "type": "specific",
                    "description": "Short description of this entity",
                    "characteristics": ["fact about this entity"],
                    "positives": ["positive fact"],
                    "negatives": ["negative fact"],
                    "applicable_aspects": ["CATEGORY#ATTRIBUTE"]
                }
            ]
        }, indent=2)

        prompt = load_and_format(
            "sil", "cluster_facts",
            subject=subject,
            facts_json=json.dumps(all_facts, indent=2),
            aspect_categories=json.dumps(aspect_categories, indent=2),
            output_schema=output_schema,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.cluster") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
            )

        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")
            entities = data.get("entities", [])

            # Validate and assign IDs
            for i, entity in enumerate(entities):
                entity["id"] = f"entity-{i+1:02d}" if entity.get("type") != "generic" else "entity-generic"
                if "type" not in entity:
                    entity["type"] = "specific"
                if "characteristics" not in entity:
                    entity["characteristics"] = []
                if "positives" not in entity:
                    entity["positives"] = []
                if "negatives" not in entity:
                    entity["negatives"] = []
                if "applicable_aspects" not in entity:
                    entity["applicable_aspects"] = []

            return {"entities": entities}
        except Exception as e:
            logger.warning(f"Failed to parse clustering from {model}: {e}")
            # Fallback: single generic entity with all facts
            return {"entities": [{
                "id": "entity-generic",
                "name": subject,
                "type": "generic",
                "description": f"All facts about {subject}",
                "characteristics": classified.get("characteristics", []),
                "positives": classified.get("positives", []),
                "negatives": classified.get("negatives", []),
                "applicable_aspects": aspect_categories,
            }]}

    async def _judge_clustering(
        self,
        judge_model: str,
        source_model: str,
        classified: dict[str, list[str]],
        clustering: dict,
        subject: str,
    ) -> dict:
        """
        Round 5b: Judge another model's clustering output.

        Args:
            judge_model: Model doing the judging
            source_model: Model that produced the clustering
            classified: Original classified facts
            clustering: The clustering output to judge
            subject: The subject being researched

        Returns:
            Dict with "judgments" array of {entity_id, score, reason}
        """
        from cera.llm.openrouter import OpenRouterClient

        # Build original facts for comparison
        all_facts = []
        for category, facts in classified.items():
            for fact in facts:
                all_facts.append({"category": category, "text": fact})

        prompt = load_and_format(
            "sil", "judge_clustering",
            subject=subject,
            facts_json=json.dumps(all_facts, indent=2),
            source_model=source_model,
            clustering_json=json.dumps(clustering, indent=2),
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker, component="sil.cluster_judge") as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=judge_model,
                temperature=0.0,
            )

        try:
            from cera.api import _extract_json_from_llm
            data = _extract_json_from_llm(response, expected_type="object")
            return {"judgments": data.get("judgments", [])}
        except Exception as e:
            logger.warning(f"Failed to parse clustering judgment from {judge_model}: {e}")
            # Fallback: approve all entities
            entities = clustering.get("entities", [])
            return {"judgments": [
                {"entity_id": e.get("id", f"entity-{i+1:02d}"), "score": 1, "reason": "fallback approval"}
                for i, e in enumerate(entities)
            ]}

    def _compute_clustering_consensus(
        self,
        all_clusterings: dict[str, dict],
        all_judgments: dict[str, dict[str, dict]],
        n_models: int,
        aspect_categories: list[str],
        classified: dict[str, list[str]],
    ) -> dict:
        """
        Round 5c: Merge clusterings using point-vote consensus.

        Args:
            all_clusterings: model_name -> clustering output
            all_judgments: judge_model -> {source_model -> judgments}
            n_models: Number of MAV models
            aspect_categories: Full aspect categories list
            classified: Original classified facts (for orphan reassignment)

        Returns:
            Final verified-facts.json content
        """
        min_votes = ceil(2 / 3 * (n_models - 1)) if n_models > 2 else 1

        # ─── Score each entity across all models ───────────────
        # For each source model's entities, count approval votes from judges
        passing_entities: list[dict] = []  # Entities that pass judging
        rejected_entities: list[dict] = []

        for source_model, clustering in all_clusterings.items():
            for entity in clustering.get("entities", []):
                entity_id = entity.get("id", "")
                votes = 0

                # Count votes from all judges (excluding self)
                for judge_model, source_judgments in all_judgments.items():
                    if judge_model == source_model:
                        continue
                    judgments_for_source = source_judgments.get(source_model, {})
                    judgment_list = judgments_for_source.get("judgments", [])
                    for j in judgment_list:
                        if j.get("entity_id") == entity_id and j.get("score", 0) == 1:
                            votes += 1

                if votes >= min_votes:
                    passing_entities.append({**entity, "_source": source_model, "_votes": votes})
                else:
                    rejected_entities.append({**entity, "_source": source_model})

        # ─── Deduplicate passing entities across models ────────
        # Match entities by name similarity across models
        final_entities: list[dict] = []
        used_indices = set()

        for i, entity_a in enumerate(passing_entities):
            if i in used_indices:
                continue

            # Find duplicates of this entity across other models
            group = [entity_a]
            used_indices.add(i)

            for j, entity_b in enumerate(passing_entities):
                if j in used_indices:
                    continue
                if entity_a.get("_source") == entity_b.get("_source"):
                    continue  # Same model, skip

                # Compare entity names by semantic similarity
                sim = self._compute_similarity(
                    entity_a.get("name", ""),
                    entity_b.get("name", ""),
                )
                if sim >= 0.80:
                    group.append(entity_b)
                    used_indices.add(j)

            # If entity appears in >= 2 models' outputs, include it
            source_models = set(e.get("_source") for e in group)
            if len(source_models) >= 2 or n_models == 1:
                # Pick the most complete version (most facts + aspects)
                best = max(group, key=lambda e: (
                    len(e.get("characteristics", [])) +
                    len(e.get("positives", [])) +
                    len(e.get("negatives", [])) +
                    len(e.get("applicable_aspects", []))
                ))
                # Clean internal tracking fields
                clean = {k: v for k, v in best.items() if not k.startswith("_")}
                final_entities.append(clean)
            else:
                # Single-model entity that passed judging — still include if solo
                # but only if no better alternative exists
                rejected_entities.extend(group)

        # ─── Handle edge case: no entities passed ─────────────
        if not final_entities:
            # Fallback: use the clustering with the most approvals
            best_source = max(
                all_clusterings.keys(),
                key=lambda m: sum(
                    1 for e in all_clusterings[m].get("entities", [])
                    for judge, sj in all_judgments.items()
                    if judge != m
                    for j in sj.get(m, {}).get("judgments", [])
                    if j.get("entity_id") == e.get("id") and j.get("score", 0) == 1
                ),
            )
            final_entities = [
                {k: v for k, v in e.items() if not k.startswith("_")}
                for e in all_clusterings[best_source].get("entities", [])
            ]

        # ─── Reassign orphaned facts ──────────────────────────
        # Collect all facts that are covered by final entities
        covered_facts = set()
        for entity in final_entities:
            for fact in entity.get("characteristics", []):
                covered_facts.add(fact)
            for fact in entity.get("positives", []):
                covered_facts.add(fact)
            for fact in entity.get("negatives", []):
                covered_facts.add(fact)

        # Find orphaned facts from the original classified data
        all_original_facts = (
            classified.get("characteristics", []) +
            classified.get("positives", []) +
            classified.get("negatives", [])
        )
        orphans = [f for f in all_original_facts if f not in covered_facts]

        if orphans:
            # Find or create a generic entity for orphans
            generic = next((e for e in final_entities if e.get("type") == "generic"), None)
            if generic is None:
                generic = {
                    "id": "entity-generic",
                    "name": "General observations",
                    "type": "generic",
                    "description": "Domain-level observations not tied to a specific entity",
                    "characteristics": [],
                    "positives": [],
                    "negatives": [],
                    "applicable_aspects": [],
                }
                final_entities.append(generic)

            # Assign orphans to generic entity by their original category
            for orphan in orphans:
                if orphan in classified.get("positives", []):
                    generic["positives"].append(orphan)
                elif orphan in classified.get("negatives", []):
                    generic["negatives"].append(orphan)
                else:
                    generic["characteristics"].append(orphan)

        # ─── Aspect consensus: keep aspects agreed by ≥2 models ──
        if n_models >= 2:
            for entity in final_entities:
                entity_name = entity.get("name", "")
                # Collect aspects from matching entities across models
                aspect_votes: dict[str, int] = {}
                for source_model, clustering in all_clusterings.items():
                    for src_entity in clustering.get("entities", []):
                        sim = self._compute_similarity(entity_name, src_entity.get("name", ""))
                        if sim >= 0.80 or entity_name == src_entity.get("name"):
                            for aspect in src_entity.get("applicable_aspects", []):
                                aspect_votes[aspect] = aspect_votes.get(aspect, 0) + 1

                # Keep aspects with ≥2 votes (or ≥1 if only 1 model)
                min_aspect_votes = min(2, n_models)
                consensus_aspects = [
                    asp for asp, count in aspect_votes.items()
                    if count >= min_aspect_votes
                ]
                if consensus_aspects:
                    entity["applicable_aspects"] = consensus_aspects
                # If no aspects pass consensus, keep existing (from best entity)

        # ─── Reassign entity IDs ──────────────────────────────
        specific_count = 0
        generic_count = 0
        for entity in final_entities:
            if entity.get("type") == "generic":
                entity["id"] = f"entity-generic" if generic_count == 0 else f"entity-generic-{generic_count + 1}"
                generic_count += 1
            else:
                specific_count += 1
                entity["id"] = f"entity-{specific_count:02d}"

        return {
            "entities": final_entities,
            "total_entities": len(final_entities),
            "specific_count": specific_count,
            "generic_count": generic_count,
        }

    # ─── Utility Methods ────────────────────────────────────────────────

    async def _search_with_tavily(
        self,
        queries: list[str],
    ) -> tuple[str, list[str]]:
        """Search the web using Tavily with model-generated queries."""
        from cera.llm.web_search import WebSearchClient

        async with WebSearchClient(
            openrouter_api_key=self.api_key,
            tavily_api_key=self.tavily_api_key,
        ) as search_client:
            all_content = []
            all_sources = []

            for query in queries[:3]:  # Limit to first 3 queries
                response = await search_client.search(query, max_results=5)

                for result in response.results:
                    all_content.append(f"## {result.title}\n{result.content}")
                    if result.url and result.url not in all_sources:
                        all_sources.append(result.url)

                if response.answer:
                    all_content.insert(0, f"## Summary for: {query}\n{response.answer}")

            return "\n\n".join(all_content), all_sources

    def _deduplicate_items(self, items: list[str], threshold: float = 0.80) -> list[str]:
        """Deduplicate similar items using semantic similarity."""
        if len(items) <= 1:
            return items

        unique_items = []
        for item in items:
            is_duplicate = False
            for existing in unique_items:
                if self._compute_similarity(item, existing) >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_items.append(item)

        return unique_items

    # ─── Main Orchestrator ──────────────────────────────────────────────

    async def gather_intelligence(
        self,
        query: str,
        region: str = "united states",
        domain: str = "general",
        sentiment_depth: str = "praise and complain",
        additional_context: Optional[str] = None,
        aspect_categories: Optional[list[str]] = None,
    ) -> MAVResult:
        """
        Gather intelligence using the 5-round query-based MAV protocol.

        Round 1: Each model researches subject and generates neutral factual queries
        Round 2: All queries pooled, deduplicated, and capped
        Round 3: Each model answers ALL pooled queries (with web search)
        Round 4: Per-query consensus (ceil(2/3*N) majority voting)
        Post: Verified answers classified into categories

        Args:
            query: The subject to research
            region: Geographic region for context
            domain: Product/service domain hint
            sentiment_depth: Level of sentiment analysis (hint)
            additional_context: Optional extra context about the subject

        Returns:
            MAVResult with verified context and full reporting data
        """
        model_data_list: list[MAVModelData] = []

        # Check if MAV is enabled with at least 2 models
        if self.mav_config.enabled and len(self.mav_config.models) >= 2:
            n_models = len(self.mav_config.models)
            await self._emit_log("INFO", "SIL", f"Starting MAV with {n_models} models...", progress=5)

            # ─── ROUND 1: Research + Query Generation (parallel) ────
            await self._emit_log("INFO", "SIL", "Round 1: Researching subject and generating queries...", progress=6)

            # Per-model timeout for Round 1 (5 minutes)
            MODEL_TIMEOUT = 300

            async def round1_for_model(model: str) -> tuple[str, SubjectUnderstanding, str, list[str]]:
                """Run Round 1 for a single model."""
                understanding, search_content = await self._research_subject(
                    model, query, additional_context
                )
                queries = await self._generate_factual_queries(
                    model, query, search_content, additional_context
                )
                return model, understanding, search_content, queries

            async def round1_with_timeout(model: str) -> tuple[str, SubjectUnderstanding, str, list[str]]:
                """Run Round 1 with timeout protection."""
                return await asyncio.wait_for(round1_for_model(model), timeout=MODEL_TIMEOUT)

            # Create tasks with model names for tracking
            round1_tasks = {
                asyncio.create_task(round1_with_timeout(m)): m
                for m in self.mav_config.models
            }

            # Use as_completed to log progress as each model finishes
            round1_results = []
            completed_count = 0
            total_models = len(self.mav_config.models)

            for coro in asyncio.as_completed(round1_tasks.keys()):
                try:
                    result = await coro
                    round1_results.append(result)
                    completed_count += 1
                    model_name = result[0].split("/")[-1] if "/" in result[0] else result[0]
                    await self._emit_log(
                        "INFO", "SIL",
                        f"Round 1: {model_name} finished ({completed_count}/{total_models} models)",
                        progress=6 + int((completed_count / total_models) * 9)  # Progress 6-15
                    )
                except asyncio.TimeoutError:
                    completed_count += 1
                    # Find which model timed out
                    for task, model in round1_tasks.items():
                        if task.done() and task.exception() is not None:
                            model_name = model.split("/")[-1] if "/" in model else model
                            await self._emit_log(
                                "WARN", "SIL",
                                f"Round 1: {model_name} timed out after {MODEL_TIMEOUT}s ({completed_count}/{total_models} models)"
                            )
                            break
                    else:
                        await self._emit_log(
                            "WARN", "SIL",
                            f"Round 1: Model timed out ({completed_count}/{total_models} models)"
                        )
                    round1_results.append(asyncio.TimeoutError())
                except Exception as e:
                    completed_count += 1
                    await self._emit_log(
                        "WARN", "SIL",
                        f"Round 1: Model failed ({completed_count}/{total_models} models)"
                    )
                    round1_results.append(e)

            # Collect Round 1 results
            all_raw_queries: list[tuple[str, str]] = []  # (query_text, source_model)
            model_research: dict[str, tuple[SubjectUnderstanding, str]] = {}  # model -> (understanding, search_content)

            for result in round1_results:
                if isinstance(result, Exception):
                    continue
                model, understanding, search_content, queries = result
                model_research[model] = (understanding, search_content)
                for q in queries:
                    all_raw_queries.append((q, model))

            total_queries_generated = len(all_raw_queries)
            await self._emit_log("INFO", "SIL", f"Round 1 complete: {total_queries_generated} queries from {len(model_research)} models", progress=15)

            if total_queries_generated == 0:
                # No queries generated - fall back to single model
                await self._emit_log("WARN", "SIL", "No queries generated, falling back to single model")
                return await self._single_model_fallback(query, additional_context)

            # ─── ROUND 2: Query Pooling + Deduplication ─────────────
            await self._emit_log("INFO", "SIL", "Round 2: Pooling and deduplicating queries...", progress=16)
            pooled_queries = await self._deduplicate_queries(
                all_raw_queries,
                self.mav_config.similarity_threshold,
                self.mav_config.max_queries,
            )

            if not pooled_queries:
                await self._emit_log("WARN", "SIL", "No queries after deduplication, falling back")
                return await self._single_model_fallback(query, additional_context)

            await self._emit_log("INFO", "SIL", f"Round 2 complete: {len(pooled_queries)} unique queries (from {total_queries_generated})", progress=24)

            # ─── ROUND 3: Query Answering (parallel) ────────────────
            # Progress 26+ enters MAV phase in UI (MAV range is 25-35)
            await self._emit_log("INFO", "MAV", f"Round 3: Verifying {len(pooled_queries)} queries across models...", progress=26)

            async def round3_for_model(model: str) -> tuple[str, list[QueryAnswer]]:
                """Run Round 3 for a single model."""
                research_context = ""
                if model in model_research:
                    _, search_content = model_research[model]
                    research_context = search_content

                answers = await self._answer_queries(
                    model, query, pooled_queries, research_context, additional_context
                )
                return model, answers

            round3_tasks = [round3_for_model(m) for m in self.mav_config.models if m in model_research]
            round3_results = await asyncio.gather(*round3_tasks, return_exceptions=True)

            # Collect Round 3 results
            all_answers: dict[str, list[QueryAnswer]] = {}  # query_id -> list of answers
            model_answers: dict[str, list[QueryAnswer]] = {}  # model -> its answers

            for result in round3_results:
                if isinstance(result, Exception):
                    continue
                model, answers = result
                model_answers[model] = answers
                for ans in answers:
                    if ans.query_id not in all_answers:
                        all_answers[ans.query_id] = []
                    all_answers[ans.query_id].append(ans)

            # Build model_data_list for reporting
            for model in self.mav_config.models:
                sanitized = self._sanitize_model_name(model)
                understanding = model_research.get(model, (None, ""))[0]
                search_content = model_research.get(model, (None, ""))[1]
                queries_from_model = [q for q, m in all_raw_queries if m == model]

                model_data_list.append(MAVModelData(
                    model=model,
                    sanitized_model=sanitized,
                    understanding=understanding,
                    queries_generated=queries_from_model,
                    answers=model_answers.get(model, []),
                    search_content=search_content if isinstance(search_content, str) else "",
                ))

            await self._emit_log("INFO", "MAV", f"Round 3 complete: {len(model_answers)} models responded", progress=30)

            # ─── ROUND 4: LLM-Judged Consensus ───────────────────────
            await self._emit_log("INFO", "MAV", "Round 4: Sending answers to models for agreement judgment...", progress=31)

            responding_models = [m for m in self.mav_config.models if m in model_answers]
            effective_n_models = len(responding_models)

            # Build topics structure for judgment prompt
            topics_for_judgment = []
            for pq in pooled_queries:
                qa_list = all_answers.get(pq.id, [])
                topics_for_judgment.append({
                    "query_id": pq.id,
                    "query": pq.query,
                    "answers": [
                        {"model": a.model, "response": a.response}
                        for a in qa_list
                    ],
                })

            JUDGMENT_TIMEOUT = 120  # 2 minutes per model

            async def judge_with_timeout(model: str):
                try:
                    result = await asyncio.wait_for(
                        self._judge_agreement(model, query, topics_for_judgment),
                        timeout=JUDGMENT_TIMEOUT,
                    )
                    return model, result
                except asyncio.TimeoutError:
                    model_name = model.split("/")[-1] if "/" in model else model
                    await self._emit_log("WARN", "MAV", f"Round 4: {model_name} judgment timed out")
                    return model, {}
                except Exception as e:
                    model_name = model.split("/")[-1] if "/" in model else model
                    await self._emit_log("WARN", "MAV", f"Round 4: {model_name} judgment failed: {e}")
                    return model, {}

            judgment_tasks = [judge_with_timeout(m) for m in responding_models]
            judgment_results = await asyncio.gather(*judgment_tasks)

            # Collect all judgments: model -> {query_id -> {model: 0/1}}
            all_judgments: dict[str, dict[str, dict[str, int]]] = {}
            for model, judgments in judgment_results:
                all_judgments[model] = judgments
                model_name = model.split("/")[-1] if "/" in model else model
                n_agreements = sum(
                    sum(1 for s in scores.values() if s == 1)
                    for scores in judgments.values()
                )
                await self._emit_log("INFO", "MAV", f"Round 4: {model_name} judged {len(judgments)} topics ({n_agreements} agreements)")

            await self._emit_log("INFO", "MAV", f"Round 4: All {len(all_judgments)} model judgments collected", progress=34)

            # Compute consensus per query using LLM votes
            consensus_results: list[QueryConsensusResult] = []
            for pq in pooled_queries:
                query_answers = all_answers.get(pq.id, [])
                result = self._compute_llm_consensus(
                    pq, query_answers, all_judgments, effective_n_models
                )
                consensus_results.append(result)

            # Separate verified vs unverified
            verified_results = [r for r in consensus_results if r.consensus_reached]
            unverified_results = [r for r in consensus_results if not r.consensus_reached]

            await self._emit_log("INFO", "MAV", f"Round 4 complete: {len(verified_results)}/{len(pooled_queries)} queries reached consensus (LLM-judged)", progress=35)

            # Build report
            report = MAVQueryPoolReport(
                total_queries_generated=total_queries_generated,
                queries_after_dedup=len(pooled_queries),
                queries_with_consensus=len(verified_results),
                queries_without_consensus=len(unverified_results),
                per_query_results=consensus_results,
                threshold_used=self.mav_config.answer_threshold,
            )

            # Check consensus rate
            consensus_rate = len(verified_results) / len(pooled_queries) if pooled_queries else 0

            if consensus_rate < self.mav_config.min_verification_rate and len(pooled_queries) > 0:
                # Fallback: use union of all answers (deduplicated)
                report.used_fallback = True
                all_answer_texts = []
                for pq in pooled_queries:
                    for ans in all_answers.get(pq.id, []):
                        all_answer_texts.append(ans.response)

                deduped = self._deduplicate_items(all_answer_texts, 0.80)

                # Classify the union fallback answers
                # Create pseudo-results for classification
                pseudo_results = []
                for i, text in enumerate(deduped):
                    pseudo_results.append(QueryConsensusResult(
                        query_id=f"fallback_{i}",
                        query="",
                        answers=[],
                        consensus_reached=True,
                        consensus_answer=text,
                        agreeing_models=[],
                        pairwise_similarities={},
                        agreement_count=0,
                    ))

                classified = await self._classify_verified_answers(pseudo_results, query)
                context = SubjectContext(
                    subject=query,
                    features=classified["characteristics"],
                    pros=classified["positives"],
                    cons=classified["negatives"],
                    use_cases=classified["use_cases"],
                    mav_verified=False,
                )

                return MAVResult(
                    context=context,
                    model_data=model_data_list,
                    total_facts_extracted=total_queries_generated,
                    facts_verified=len(deduped),
                    facts_rejected=total_queries_generated - len(deduped),
                    query_pool_report=report,
                )

            # ─── POST-CONSENSUS: Classification ─────────────────────
            await self._emit_log("INFO", "SIL", "Classifying verified facts into categories...", progress=40)
            classified = await self._classify_verified_answers(verified_results, query)

            n_features = len(classified.get("characteristics", []))
            n_pros = len(classified.get("positives", []))
            n_cons = len(classified.get("negatives", []))
            await self._emit_log("INFO", "SIL", f"Classification complete: {n_features} features, {n_pros} pros, {n_cons} cons")

            # ─── ROUND 5: Entity Clustering (multi-model consensus) ──
            verified_facts = None
            if aspect_categories:
                await self._emit_log("INFO", "SIL", "Round 5: Clustering facts by entity...", progress=42)

                # 5a: Each model clusters independently (parallel)
                clustering_tasks = [
                    self._cluster_facts_by_entity(model, classified, aspect_categories, query)
                    for model in responding_models
                ]
                clustering_results = await asyncio.gather(*clustering_tasks, return_exceptions=True)

                all_clusterings: dict[str, dict] = {}
                for model, result in zip(responding_models, clustering_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Clustering failed for {model}: {result}")
                    else:
                        all_clusterings[model] = result

                if len(all_clusterings) >= 2:
                    await self._emit_log("INFO", "SIL", f"Round 5a: {len(all_clusterings)} models produced clusterings, judging...", progress=44)

                    # 5b: Cross-model judging (parallel)
                    judge_tasks = []
                    judge_pairs = []  # Track (judge, source) for each task
                    for judge in all_clusterings:
                        for source in all_clusterings:
                            if judge != source:
                                judge_tasks.append(
                                    self._judge_clustering(judge, source, classified, all_clusterings[source], query)
                                )
                                judge_pairs.append((judge, source))

                    judgment_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

                    # Restructure into judge -> {source -> judgments}
                    structured_judgments: dict[str, dict[str, dict]] = {}
                    for (judge, source), result in zip(judge_pairs, judgment_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Judging failed for {judge} -> {source}: {result}")
                            continue
                        structured_judgments.setdefault(judge, {})[source] = result

                    await self._emit_log("INFO", "SIL", "Round 5c: Computing entity consensus...", progress=46)

                    # 5c: Consensus
                    verified_facts = self._compute_clustering_consensus(
                        all_clusterings, structured_judgments, len(all_clusterings),
                        aspect_categories, classified,
                    )

                    n_entities = verified_facts.get("total_entities", 0)
                    n_specific = verified_facts.get("specific_count", 0)
                    n_generic = verified_facts.get("generic_count", 0)
                    await self._emit_log(
                        "INFO", "SIL",
                        f"Entity clustering complete: {n_entities} entities ({n_specific} specific, {n_generic} generic)",
                        progress=48,
                    )

                elif len(all_clusterings) == 1:
                    # Single model produced clustering — use it directly (no judging)
                    single_model = list(all_clusterings.keys())[0]
                    verified_facts = all_clusterings[single_model]
                    verified_facts.setdefault("total_entities", len(verified_facts.get("entities", [])))
                    verified_facts.setdefault("specific_count", sum(1 for e in verified_facts.get("entities", []) if e.get("type") != "generic"))
                    verified_facts.setdefault("generic_count", sum(1 for e in verified_facts.get("entities", []) if e.get("type") == "generic"))
                    await self._emit_log("WARN", "SIL", f"Only 1 model produced clustering — using {single_model}'s output directly")

                else:
                    await self._emit_log("WARN", "SIL", "No models produced valid clusterings — skipping entity clustering")

            context = SubjectContext(
                subject=query,
                features=classified["characteristics"],
                pros=classified["positives"],
                cons=classified["negatives"],
                use_cases=classified["use_cases"],
                mav_verified=True,
            )

            await self._emit_log("INFO", "SIL", "Subject intelligence gathering complete", progress=50)
            return MAVResult(
                context=context,
                model_data=model_data_list,
                total_facts_extracted=len(pooled_queries),
                facts_verified=len(verified_results),
                facts_rejected=len(unverified_results),
                query_pool_report=report,
                verified_facts=verified_facts,
            )

        # Fallback: Single model extraction without MAV
        return await self._single_model_fallback(query, additional_context)

    async def _single_model_fallback(
        self,
        query: str,
        additional_context: Optional[str] = None,
    ) -> MAVResult:
        """Fallback to single model extraction when MAV cannot run."""
        fallback_model = (
            self.mav_config.models[0]
            if self.mav_config.models
            else "anthropic/claude-sonnet-4"
        )

        try:
            understanding, search_content = await self._research_subject(
                fallback_model, query, additional_context
            )
            queries = await self._generate_factual_queries(
                fallback_model, query, search_content, additional_context
            )

            # Answer the queries with the single model
            pooled = [FactualQuery(id=f"q{i+1}", query=q, source_model=fallback_model) for i, q in enumerate(queries)]
            answers = await self._answer_queries(
                fallback_model, query, pooled, search_content, additional_context
            )

            # Classify answers directly (no consensus needed)
            pseudo_results = []
            for ans in answers:
                pq = next((q for q in pooled if q.id == ans.query_id), None)
                pseudo_results.append(QueryConsensusResult(
                    query_id=ans.query_id,
                    query=pq.query if pq else "",
                    answers=[ans],
                    consensus_reached=True,
                    consensus_answer=ans.response,
                    agreeing_models=[fallback_model],
                    pairwise_similarities={},
                    agreement_count=1,
                ))

            classified = await self._classify_verified_answers(pseudo_results, query)

            context = SubjectContext(
                subject=query,
                features=classified["characteristics"],
                pros=classified["positives"],
                cons=classified["negatives"],
                use_cases=classified["use_cases"],
                mav_verified=False,
            )

            model_data = MAVModelData(
                model=fallback_model,
                sanitized_model=self._sanitize_model_name(fallback_model),
                understanding=understanding,
                queries_generated=queries,
                answers=answers,
                search_content=search_content,
            )

            return MAVResult(
                context=context,
                model_data=[model_data],
                total_facts_extracted=len(queries),
                facts_verified=len(pseudo_results),
                facts_rejected=0,
            )
        except Exception:
            # Complete failure - return empty context
            context = SubjectContext(
                subject=query,
                features=[],
                pros=[],
                cons=[],
                use_cases=[],
                mav_verified=False,
            )
            return MAVResult(
                context=context,
                model_data=[],
                total_facts_extracted=0,
                facts_verified=0,
                facts_rejected=0,
            )
