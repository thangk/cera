"""VDT (Vocabulary Diversity Tracker) - Tracks phrase frequency across entire generation run.

Complements NEB (which remembers last N full reviews via FIFO) by tracking cumulative
bigram/trigram frequencies. When phrases become overused (>5% of reviews), they are
injected into the system prompt as "avoid these phrases" guidance.

Subject-related terms (product features, domain keywords) are exempt from tracking
since repetition of product vocabulary is natural when reviewing a single subject.
"""

from collections import Counter
import re

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "when", "that", "this",
    "it", "its", "i", "my", "me", "we", "our", "you", "your", "they",
    "their", "them", "he", "she", "his", "her", "about", "also", "been",
    "like", "really", "pretty", "much", "even", "well", "still", "got",
})


class VocabDiversityTracker:
    """Tracks bigram/trigram frequencies across all generated reviews
    and provides 'avoid overused phrases' guidance for the LLM prompt.

    Subject terms (features, domain keywords) are exempt — repeating
    product-specific vocabulary is natural when all reviews target
    the same subject."""

    def __init__(self, top_k: int = 10, subject_terms: list[str] | None = None):
        self.bigram_counter: Counter = Counter()
        self.trigram_counter: Counter = Counter()
        self.review_count: int = 0
        self.top_k = top_k
        # Build set of lowercase subject tokens to exempt from overuse detection
        self._subject_tokens: frozenset[str] = frozenset()
        if subject_terms:
            tokens = set()
            for term in subject_terms:
                for word in re.findall(r"[a-z']+", term.lower()):
                    if word not in STOP_WORDS and len(word) > 2:
                        tokens.add(word)
            self._subject_tokens = frozenset(tokens)

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, extract words, filter stopwords."""
        words = re.findall(r"[a-z']+", text.lower())
        return [w for w in words if w not in STOP_WORDS and len(w) > 2]

    def _get_ngrams(self, tokens: list[str], n: int) -> list[tuple]:
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _contains_subject_term(self, ngram: tuple) -> bool:
        """Check if any token in the ngram is a subject-related term."""
        return any(token in self._subject_tokens for token in ngram)

    def update(self, review_text: str) -> None:
        """Add a completed review's vocabulary to the tracker."""
        tokens = self._tokenize(review_text)
        self.bigram_counter.update(self._get_ngrams(tokens, 2))
        self.trigram_counter.update(self._get_ngrams(tokens, 3))
        self.review_count += 1

    def get_overused_phrases(self) -> list[str]:
        """Return top-k most overused bigrams/trigrams.
        Only returns phrases appearing in >5% of reviews (min 3 occurrences).
        Skips phrases containing subject-related terms (natural product vocabulary)."""
        if self.review_count < 10:
            return []

        threshold = max(3, int(self.review_count * 0.05))

        candidates = []
        for ngram, count in self.bigram_counter.most_common(self.top_k * 3):
            if count >= threshold and not self._contains_subject_term(ngram):
                candidates.append((" ".join(ngram), count))
        for ngram, count in self.trigram_counter.most_common(self.top_k * 2):
            if count >= threshold and not self._contains_subject_term(ngram):
                candidates.append((" ".join(ngram), count))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in candidates[:self.top_k]]

    def get_formatted_context(self) -> str:
        """Format overused phrases as a prompt injection section.
        Returns empty string if no overused phrases detected yet."""
        phrases = self.get_overused_phrases()
        if not phrases:
            return ""

        lines = [
            "## Vocabulary Diversity",
            "The following phrases have been OVERUSED in previously generated reviews.",
            "AVOID these exact phrases and find alternative ways to express the same ideas:",
            "",
        ]
        for phrase in phrases:
            lines.append(f'- "{phrase}"')
        lines.append("")
        lines.append("Use fresh vocabulary, synonyms, and different sentence constructions.")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        return {
            "reviews_tracked": self.review_count,
            "unique_bigrams": len(self.bigram_counter),
            "overused_count": len(self.get_overused_phrases()),
            "subject_terms": len(self._subject_tokens),
        }
