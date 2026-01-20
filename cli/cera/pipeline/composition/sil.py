"""Subject Intelligence Layer (SIL) - Factual grounding through independent web research.

This module implements the redesigned MAV (Multi-Agent Verification) architecture where
each model independently:
1. Understands what type of subject is being researched
2. Generates its own search queries based on that understanding
3. Searches the web using its provider's infrastructure (or Tavily fallback)
4. Extracts facts in a consistent schema

The 2/3 majority voting then compares extractions across models to find consensus.
"""

from dataclasses import dataclass, field
from typing import Optional
import asyncio
import json
import re


@dataclass
class MAVConfig:
    """Multi-Agent Verification configuration."""

    enabled: bool = True
    models: list[str] = None  # 3 models for cross-validation
    similarity_threshold: float = 0.85

    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass
class SubjectContext:
    """Context document containing subject intelligence."""

    subject: str
    features: list[str]  # Called "characteristics" in extraction, mapped to features
    pros: list[str]  # Called "positives" in extraction
    cons: list[str]  # Called "negatives" in extraction
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
class ExtractionResult:
    """Result from a single model's extraction using consistent schema."""

    model: str
    characteristics: list[str]  # Key attributes (specs, ingredients, material, etc.)
    positives: list[str]  # Things reviewers praise
    negatives: list[str]  # Things reviewers complain about
    use_cases: list[str]  # When/where/how it's typically used
    availability: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    raw_response: str = ""


class SubjectIntelligenceLayer:
    """
    Subject Intelligence Layer (SIL) with Independent MAV.

    Provides factual grounding to reduce hallucination by gathering
    intelligence about the review subject through INDEPENDENT web research
    per model, then applying Multi-Agent Verification (MAV).

    New Flow (Independent Research per Model):
    1. Each MAV model independently understands the subject type
    2. Each model generates its own search queries
    3. Each model searches using its provider's infrastructure (:online) or Tavily
    4. Each model extracts facts using a consistent schema
    5. MAV applies 2/3 majority voting across extractions
    6. Return verified SubjectContext with high-confidence facts
    """

    # Prompt for understanding the subject (Step 1)
    UNDERSTAND_SUBJECT_PROMPT = """You are researching "{subject}" to gather factual information for review generation.

First, determine:
1. What TYPE of thing is this? (electronics, clothing, food, sports equipment, vehicle, service, etc.)
2. What aspects are RELEVANT to research for this type of subject?
3. What would reviewers typically praise or complain about?

Then generate 3-5 search queries to find factual information about this subject.
Focus on aspects that are relevant to the type of subject identified.

Return ONLY valid JSON with no other text:
{{
  "subject_type": "the type category",
  "relevant_aspects": ["aspect1", "aspect2", ...],
  "search_queries": ["query1", "query2", ...]
}}"""

    # Prompt for extracting facts with web search capability (Step 2-3)
    SEARCH_AND_EXTRACT_PROMPT = """You have web search capabilities. Research "{subject}" using the web.

Search for information about: {queries}

After researching, extract facts in this EXACT JSON format:
{{
  "characteristics": ["key attribute 1", "key attribute 2", ...],
  "positives": ["thing reviewers praise 1", "thing reviewers praise 2", ...],
  "negatives": ["thing reviewers complain about 1", "thing reviewers complain about 2", ...],
  "use_cases": ["typical use case 1", "typical use case 2", ...],
  "availability": "price/availability info or null"
}}

IMPORTANT:
- characteristics: Key attributes relevant to the subject type (specs for electronics, ingredients for food, material for clothing, etc.)
- positives: Things people praise in reviews (5-10 items)
- negatives: Things people complain about in reviews (3-5 items)
- use_cases: Typical scenarios where this is used (3-5 items)
- Only include information you found through web search
- Be specific and factual
- Return ONLY valid JSON, no other text"""

    # Prompt for extracting facts from provided search results (fallback)
    EXTRACT_FROM_RESULTS_PROMPT = """Extract factual information from these search results about "{subject}".

SEARCH RESULTS:
{search_content}

Extract facts in this EXACT JSON format:
{{
  "characteristics": ["key attribute 1", "key attribute 2", ...],
  "positives": ["thing reviewers praise 1", "thing reviewers praise 2", ...],
  "negatives": ["thing reviewers complain about 1", "thing reviewers complain about 2", ...],
  "use_cases": ["typical use case 1", "typical use case 2", ...],
  "availability": "price/availability info or null"
}}

IMPORTANT:
- characteristics: Key attributes relevant to the subject type (specs for electronics, ingredients for food, material for clothing, etc.)
- positives: Things people praise in reviews (5-10 items)
- negatives: Things people complain about in reviews (3-5 items)
- use_cases: Typical scenarios where this is used (3-5 items)
- Only include information supported by the search results
- Be specific and factual
- Return ONLY valid JSON, no other text"""

    def __init__(
        self,
        api_key: str,
        mav_config: Optional[MAVConfig] = None,
        tavily_api_key: Optional[str] = None,
    ):
        """
        Initialize SIL.

        Args:
            api_key: OpenRouter API key
            mav_config: Optional MAV configuration with 3 models for verification
            tavily_api_key: Optional Tavily API key for web search fallback
        """
        self.api_key = api_key
        self.mav_config = mav_config or MAVConfig(enabled=False)
        self.tavily_api_key = tavily_api_key

    def _supports_online(self, model: str) -> bool:
        """Check if a model supports web search via :online suffix or native search."""
        from cera.llm.openrouter import supports_online_search

        return supports_online_search(model)

    def _get_search_model(self, model: str) -> Optional[str]:
        """Get the model ID configured for web search."""
        from cera.llm.openrouter import get_search_model_id

        return get_search_model_id(model)

    async def _understand_subject(
        self,
        model: str,
        subject: str,
    ) -> SubjectUnderstanding:
        """
        Let a model independently understand the subject type.

        Each model determines:
        - What type of thing is this?
        - What aspects are relevant to research?
        - What search queries should be used?

        Args:
            model: Model ID to use
            subject: Subject being researched

        Returns:
            SubjectUnderstanding with queries to use
        """
        from cera.llm.openrouter import OpenRouterClient

        prompt = self.UNDERSTAND_SUBJECT_PROMPT.format(subject=subject)

        async with OpenRouterClient(self.api_key) as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,  # Slightly creative for query generation
            )

        # Parse JSON from response
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return SubjectUnderstanding(
                model=model,
                subject_type=data.get("subject_type", "general"),
                relevant_aspects=data.get("relevant_aspects", []),
                search_queries=data.get("search_queries", [f"{subject} reviews"]),
                raw_response=response,
            )
        except json.JSONDecodeError:
            # Fallback: generate basic queries
            return SubjectUnderstanding(
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

    async def _search_with_tavily(
        self,
        queries: list[str],
    ) -> tuple[str, list[str]]:
        """
        Search the web using Tavily with model-generated queries.

        Args:
            queries: List of search queries

        Returns:
            Tuple of (combined_content, source_urls)
        """
        from cera.llm.web_search import WebSearchClient

        async with WebSearchClient(
            openrouter_api_key=self.api_key,
            tavily_api_key=self.tavily_api_key,
        ) as search_client:
            # Run all queries and combine results
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

    async def _search_and_extract(
        self,
        model: str,
        subject: str,
        understanding: SubjectUnderstanding,
    ) -> ExtractionResult:
        """
        Search and extract facts using a model's understanding.

        Uses hybrid approach:
        - If model supports :online → use model's native search
        - Otherwise → use Tavily with model-generated queries

        Args:
            model: Model ID to use
            subject: Subject being researched
            understanding: Model's understanding with queries

        Returns:
            ExtractionResult with facts
        """
        from cera.llm.openrouter import OpenRouterClient

        search_model = self._get_search_model(model)
        queries_str = ", ".join(understanding.search_queries)
        sources = []

        if search_model:
            # Model supports web search - let it search directly
            prompt = self.SEARCH_AND_EXTRACT_PROMPT.format(
                subject=subject,
                queries=queries_str,
            )

            async with OpenRouterClient(self.api_key) as client:
                response = await client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=search_model,
                    temperature=0.0,  # Deterministic for fact extraction
                )
        else:
            # Model doesn't support web search - use Tavily fallback
            search_content, sources = await self._search_with_tavily(
                understanding.search_queries
            )

            prompt = self.EXTRACT_FROM_RESULTS_PROMPT.format(
                subject=subject,
                search_content=search_content[:15000],  # Truncate if too long
            )

            async with OpenRouterClient(self.api_key) as client:
                response = await client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                )

        # Parse JSON from response
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return ExtractionResult(
                model=model,
                characteristics=data.get("characteristics", []),
                positives=data.get("positives", []),
                negatives=data.get("negatives", []),
                use_cases=data.get("use_cases", []),
                availability=data.get("availability"),
                sources=sources,
                raw_response=response,
            )
        except json.JSONDecodeError:
            return ExtractionResult(
                model=model,
                characteristics=[],
                positives=[],
                negatives=[],
                use_cases=[],
                sources=sources,
                raw_response=response,
            )

    async def _gather_with_model(
        self,
        model: str,
        subject: str,
    ) -> ExtractionResult:
        """
        Complete pipeline for a single model's independent research.

        1. Understand the subject
        2. Search and extract facts

        Args:
            model: Model ID to use
            subject: Subject being researched

        Returns:
            ExtractionResult from this model's independent research
        """
        # Step 1: Understand the subject
        understanding = await self._understand_subject(model, subject)

        # Step 2: Search and extract
        extraction = await self._search_and_extract(model, subject, understanding)

        return extraction

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode([text1, text2])
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity[0][0])
        except ImportError:
            # Fallback: Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0

    def _find_consensus(
        self,
        items_per_model: list[list[str]],
        threshold: float,
    ) -> list[str]:
        """
        Find items that appear in at least 2/3 of models (majority voting).

        Args:
            items_per_model: List of item lists from each model
            threshold: Similarity threshold for matching items

        Returns:
            List of consensus items
        """
        if len(items_per_model) < 2:
            return items_per_model[0] if items_per_model else []

        # Flatten all items with their source model index
        all_items = []
        for model_idx, items in enumerate(items_per_model):
            for item in items:
                all_items.append((item, model_idx))

        # Group similar items
        groups = []
        used = set()

        for i, (item1, model1) in enumerate(all_items):
            if i in used:
                continue

            group = [(item1, model1)]
            used.add(i)

            for j, (item2, model2) in enumerate(all_items):
                if j in used or model1 == model2:
                    continue

                # Check similarity
                similarity = self._compute_similarity(item1, item2)
                if similarity >= threshold:
                    group.append((item2, model2))
                    used.add(j)

            groups.append(group)

        # Filter to items that appear in at least 2/3 of models
        min_models = max(2, len(items_per_model) * 2 // 3)
        consensus_items = []

        for group in groups:
            unique_models = len(set(model for _, model in group))
            if unique_models >= min_models:
                # Use the first item as the canonical form
                consensus_items.append(group[0][0])

        return consensus_items

    def _apply_mav(
        self,
        extractions: list[ExtractionResult],
        threshold: float,
    ) -> SubjectContext:
        """
        Apply Multi-Agent Verification to extraction results.

        Uses 2/3 majority voting across models to verify facts.

        Args:
            extractions: List of extraction results from each model
            threshold: Similarity threshold for matching

        Returns:
            Verified SubjectContext
        """
        if not extractions:
            return SubjectContext(
                subject="",
                features=[],
                pros=[],
                cons=[],
                use_cases=[],
                mav_verified=False,
            )

        # Gather items from each model (using consistent schema fields)
        characteristics_per_model = [e.characteristics for e in extractions]
        positives_per_model = [e.positives for e in extractions]
        negatives_per_model = [e.negatives for e in extractions]
        use_cases_per_model = [e.use_cases for e in extractions]

        # Find consensus through majority voting
        verified_characteristics = self._find_consensus(
            characteristics_per_model, threshold
        )
        verified_positives = self._find_consensus(positives_per_model, threshold)
        verified_negatives = self._find_consensus(negatives_per_model, threshold)
        verified_use_cases = self._find_consensus(use_cases_per_model, threshold)

        # For availability, use the most common non-null value
        availabilities = [e.availability for e in extractions if e.availability]
        availability = availabilities[0] if availabilities else None

        # Collect all sources
        all_sources = []
        for e in extractions:
            for source in e.sources:
                if source not in all_sources:
                    all_sources.append(source)

        return SubjectContext(
            subject="",  # Will be set by caller
            features=verified_characteristics,  # Map characteristics → features
            pros=verified_positives,  # Map positives → pros
            cons=verified_negatives,  # Map negatives → cons
            use_cases=verified_use_cases,
            availability=availability,
            mav_verified=True,
            search_sources=all_sources,
        )

    async def gather_intelligence(
        self,
        query: str,
        region: str = "united states",
        category: str = "general",
        feature_count: str = "5-10",
        sentiment_depth: str = "praise and complain",
        context_scope: str = "typical use cases",
    ) -> SubjectContext:
        """
        Gather intelligence about the subject using independent MAV research.

        New Flow:
        1. Each MAV model independently understands the subject type
        2. Each model generates its own search queries
        3. Each model searches using its provider's infrastructure or Tavily
        4. Each model extracts facts in consistent schema
        5. MAV applies 2/3 majority voting across extractions
        6. Return verified SubjectContext

        Args:
            query: The subject to research (e.g., "iPhone 15 Pro", "summer dress", "pad thai")
            region: Geographic region for context
            category: Product/service category hint
            feature_count: Range of features to extract (hint)
            sentiment_depth: Level of sentiment analysis (hint)
            context_scope: Scope of contextual information (hint)

        Returns:
            SubjectContext with gathered intelligence
        """
        # Check if MAV is enabled and properly configured
        if self.mav_config.enabled and len(self.mav_config.models) == 3:
            # Run independent research for all 3 MAV models in parallel
            research_tasks = [
                self._gather_with_model(model, query)
                for model in self.mav_config.models
            ]
            extractions = await asyncio.gather(*research_tasks, return_exceptions=True)

            # Filter out errors
            valid_extractions = [
                e for e in extractions if isinstance(e, ExtractionResult)
            ]

            if len(valid_extractions) >= 2:
                # Apply MAV verification (2/3 majority voting)
                context = self._apply_mav(
                    valid_extractions,
                    self.mav_config.similarity_threshold,
                )
                context.subject = query
                return context

        # Fallback: Single model extraction without MAV
        fallback_model = (
            self.mav_config.models[0]
            if self.mav_config.models
            else "anthropic/claude-sonnet-4"
        )

        extraction = await self._gather_with_model(fallback_model, query)

        return SubjectContext(
            subject=query,
            features=extraction.characteristics,
            pros=extraction.positives,
            cons=extraction.negatives,
            use_cases=extraction.use_cases,
            availability=extraction.availability,
            mav_verified=False,
            search_sources=extraction.sources,
        )

    async def verify_with_mav(
        self,
        claims: list[str],
        models: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Verify claims using Multi-Agent Verification (MAV).

        Uses 2/3 majority voting across multiple LLMs to verify
        factual claims with semantic similarity threshold τ.

        Args:
            claims: List of claims to verify
            models: Optional list of 3 model IDs to use (overrides config)

        Returns:
            List of verified claims (those passing MAV)
        """
        from cera.llm.mav import MultiAgentVerification

        mav_models = models or self.mav_config.models

        if not mav_models or len(mav_models) != 3:
            # Skip MAV if not properly configured
            return claims

        mav = MultiAgentVerification(
            api_key=self.api_key,
            models=mav_models,
            similarity_threshold=self.mav_config.similarity_threshold,
        )

        return await mav.filter_verified(claims)
