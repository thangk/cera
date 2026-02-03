"""Multi-Dimensional Quality Assessment (MDQA) - Comprehensive evaluation metrics."""

from dataclasses import dataclass
from typing import Optional
from collections import Counter
import math


@dataclass
class MDQAMetrics:
    """MDQA evaluation metrics."""

    # Lexical Quality
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None

    # Semantic Similarity
    bertscore: Optional[float] = None
    moverscore: Optional[float] = None

    # Corpus Diversity
    distinct_1: Optional[float] = None
    distinct_2: Optional[float] = None
    self_bleu: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bleu": self.bleu,
            "rouge_l": self.rouge_l,
            "bertscore": self.bertscore,
            "moverscore": self.moverscore,
            "distinct_1": self.distinct_1,
            "distinct_2": self.distinct_2,
            "self_bleu": self.self_bleu,
        }

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        scores = []
        if self.bertscore is not None:
            scores.append(self.bertscore)
        if self.distinct_2 is not None:
            scores.append(self.distinct_2)
        if self.self_bleu is not None:
            # Lower self-BLEU is better, so invert
            scores.append(1 - self.self_bleu)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)


class MultiDimensionalQualityAssessment:
    """
    Multi-Dimensional Quality Assessment (MDQA).

    Evaluates synthetic reviews across three dimensions:
    1. Lexical Quality (BLEU, ROUGE-L)
    2. Semantic Similarity (BERTScore, MoverScore)
    3. Corpus Diversity (Distinct-n, Self-BLEU)

    All metrics can run on CPU (no GPU required), though GPU speeds up
    BERTScore and MoverScore significantly for large datasets.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize MDQA.

        Args:
            use_gpu: Use GPU for neural metrics (BERTScore, MoverScore).
                     If False, runs on CPU which is slower but works everywhere.
        """
        self.use_gpu = use_gpu
        self._device = "cuda" if use_gpu else "cpu"
        self._bertscore_model = None
        self._sentence_model = None  # Lazy-loaded and cached

    def _get_sentence_model(self):
        """Lazy-load and cache the SentenceTransformer model."""
        if self._sentence_model is None:
            import os
            import sys
            import logging
            import warnings
            from io import StringIO

            # Suppress all progress bars and verbose logging
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("safetensors").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

            from sentence_transformers import SentenceTransformer

            # Temporarily redirect stderr to suppress progress bar output
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                # Try to load from cache without network check
                try:
                    self._sentence_model = SentenceTransformer(
                        "all-MiniLM-L6-v2", device=self._device, local_files_only=True
                    )
                except Exception:
                    # Fallback: download if not cached
                    self._sentence_model = SentenceTransformer(
                        "all-MiniLM-L6-v2", device=self._device
                    )
            finally:
                sys.stderr = old_stderr
        return self._sentence_model

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: list[str], n: int) -> list[tuple]:
        """Extract n-grams from tokens."""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    # =========================================================================
    # LEXICAL QUALITY METRICS
    # =========================================================================

    def compute_bleu(
        self,
        generated: list[str],
        references: list[str],
        max_n: int = 4,
        smooth: bool = True,
    ) -> float:
        """
        Compute corpus-level BLEU score.

        BLEU measures n-gram precision of generated text against references.
        Uses smoothing to handle zero counts (BLEU+1 method).

        Args:
            generated: List of generated texts (hypotheses)
            references: List of reference texts
            max_n: Maximum n-gram order (default 4 for BLEU-4)
            smooth: Apply smoothing for zero counts

        Returns:
            BLEU score (0-1, higher is better)
        """
        if not generated or not references:
            return 0.0

        # Compute n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            matches = 0
            total = 0

            for hyp, ref in zip(generated, references):
                hyp_tokens = self._tokenize(hyp)
                ref_tokens = self._tokenize(ref)

                hyp_ngrams = Counter(self._get_ngrams(hyp_tokens, n))
                ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))

                # Clipped counts
                for ngram, count in hyp_ngrams.items():
                    matches += min(count, ref_ngrams.get(ngram, 0))
                total += sum(hyp_ngrams.values())

            if total == 0:
                precision = 0.0
            elif matches == 0 and smooth:
                # Smoothing: add 1 to numerator and denominator
                precision = 1.0 / (total + 1)
            else:
                precision = matches / total

            precisions.append(precision)

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0

        log_precision = sum(math.log(p) for p in precisions) / len(precisions)

        # Brevity penalty
        hyp_len = sum(len(self._tokenize(h)) for h in generated)
        ref_len = sum(len(self._tokenize(r)) for r in references)

        if hyp_len == 0:
            return 0.0

        if hyp_len < ref_len:
            bp = math.exp(1 - ref_len / hyp_len)
        else:
            bp = 1.0

        return bp * math.exp(log_precision)

    def compute_rouge_l(
        self,
        generated: list[str],
        references: list[str],
    ) -> float:
        """
        Compute ROUGE-L score (Longest Common Subsequence).

        ROUGE-L measures the longest common subsequence between
        generated and reference texts, capturing sentence-level similarity.

        Args:
            generated: List of generated texts
            references: List of reference texts

        Returns:
            ROUGE-L F1 score (0-1, higher is better)
        """
        if not generated or not references:
            return 0.0

        def lcs_length(x: list[str], y: list[str]) -> int:
            """Compute length of longest common subsequence."""
            m, n = len(x), len(y)
            # Use 1D DP to save memory
            prev = [0] * (n + 1)
            curr = [0] * (n + 1)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        curr[j] = prev[j - 1] + 1
                    else:
                        curr[j] = max(prev[j], curr[j - 1])
                prev, curr = curr, [0] * (n + 1)

            return prev[n]

        total_precision = 0.0
        total_recall = 0.0
        count = 0

        for hyp, ref in zip(generated, references):
            hyp_tokens = self._tokenize(hyp)
            ref_tokens = self._tokenize(ref)

            if not hyp_tokens or not ref_tokens:
                continue

            lcs_len = lcs_length(hyp_tokens, ref_tokens)

            precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0
            recall = lcs_len / len(ref_tokens) if ref_tokens else 0

            total_precision += precision
            total_recall += recall
            count += 1

        if count == 0:
            return 0.0

        avg_precision = total_precision / count
        avg_recall = total_recall / count

        if avg_precision + avg_recall == 0:
            return 0.0

        # F1 score
        return 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    # =========================================================================
    # SEMANTIC SIMILARITY METRICS
    # =========================================================================

    def compute_bertscore(
        self,
        generated: list[str],
        references: list[str],
    ) -> float:
        """
        Compute BERTScore.

        Uses BERT embeddings to measure semantic similarity
        between generated and reference texts. Works on CPU but
        GPU is significantly faster for large datasets.

        Args:
            generated: List of generated review texts
            references: List of reference review texts

        Returns:
            BERTScore F1 (0-1, higher is better)
        """
        try:
            from bert_score import score

            P, R, F1 = score(
                generated,
                references,
                lang="en",
                verbose=False,
                device=self._device,
            )
            return float(F1.mean())
        except ImportError:
            print("Warning: bert_score not installed. Install with: pip install bert-score")
            print("Falling back to ROUGE-L as semantic approximation.")
            return self.compute_rouge_l(generated, references)
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return self.compute_rouge_l(generated, references)

    def compute_moverscore(
        self,
        generated: list[str],
        references: list[str],
    ) -> float:
        """
        Compute MoverScore.

        MoverScore uses Word Mover's Distance with contextualized embeddings
        to measure semantic similarity. It captures soft semantic alignment
        between words, making it more robust to paraphrasing than BLEU.

        Works on CPU but GPU significantly speeds up computation.

        Args:
            generated: List of generated texts
            references: List of reference texts

        Returns:
            MoverScore (0-1, higher is better)
        """
        try:
            from moverscore_v2 import get_idf_dict, word_mover_score

            # Compute IDF weights from references
            idf_dict_ref = get_idf_dict(references)
            idf_dict_hyp = get_idf_dict(generated)

            scores = word_mover_score(
                references,
                generated,
                idf_dict_ref,
                idf_dict_hyp,
                stop_words=[],
                n_gram=1,
                remove_subwords=True,
                batch_size=48,
                device=self._device,
            )

            return sum(scores) / len(scores) if scores else 0.0

        except ImportError:
            # Fallback: Use sentence-transformers for semantic similarity
            try:
                from sentence_transformers import util

                model = self._get_sentence_model()
                gen_embeddings = model.encode(generated, convert_to_tensor=True, show_progress_bar=False)
                ref_embeddings = model.encode(references, convert_to_tensor=True, show_progress_bar=False)

                # Compute cosine similarity for each pair
                similarities = util.cos_sim(gen_embeddings, ref_embeddings)
                # Take diagonal (matching pairs)
                scores = [float(similarities[i][i]) for i in range(len(generated))]

                return sum(scores) / len(scores) if scores else 0.0

            except ImportError:
                print(
                    "Warning: Neither moverscore nor sentence-transformers installed."
                )
                print("Install with: pip install moverscore sentence-transformers")
                print("Falling back to ROUGE-L as semantic approximation.")
                return self.compute_rouge_l(generated, references)

        except Exception as e:
            print(f"Warning: MoverScore computation failed: {e}")
            return self.compute_rouge_l(generated, references)

    # =========================================================================
    # CORPUS DIVERSITY METRICS
    # =========================================================================

    def compute_distinct_n(self, texts: list[str], n: int = 1) -> float:
        """
        Compute Distinct-n metric.

        Measures lexical diversity as the ratio of unique n-grams
        to total n-grams across the corpus.

        Args:
            texts: List of review texts
            n: N-gram size (1 for unigrams, 2 for bigrams)

        Returns:
            Distinct-n score (0-1, higher is better)
        """
        all_ngrams = []
        for text in texts:
            tokens = self._tokenize(text)
            ngrams = self._get_ngrams(tokens, n)
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique = len(set(all_ngrams))
        total = len(all_ngrams)

        return unique / total

    def compute_self_bleu(self, texts: list[str], sample_size: int = 100) -> float:
        """
        Compute Self-BLEU metric.

        Measures corpus diversity by computing BLEU score of each
        text against all others. Lower is better (more diverse).

        Args:
            texts: List of review texts
            sample_size: Number of texts to sample for efficiency

        Returns:
            Self-BLEU score (0-1, lower is better)
        """
        if len(texts) < 2:
            return 0.0

        # Sample if corpus is large
        import random

        if len(texts) > sample_size:
            texts = random.sample(texts, sample_size)

        total_bleu = 0.0
        count = 0

        for i, text in enumerate(texts):
            references = texts[:i] + texts[i + 1 :]
            if not references:
                continue

            # Simplified BLEU calculation
            hypothesis = self._tokenize(text)
            if not hypothesis:
                continue

            ref_ngrams = Counter()
            for ref in references:
                ref_tokens = self._tokenize(ref)
                for n in range(1, 5):
                    ref_ngrams.update(self._get_ngrams(ref_tokens, n))

            hyp_ngrams = Counter()
            for n in range(1, 5):
                hyp_ngrams.update(self._get_ngrams(hypothesis, n))

            if not hyp_ngrams:
                continue

            # Precision
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            precision = matches / total if total > 0 else 0

            total_bleu += precision
            count += 1

        return total_bleu / count if count > 0 else 0.0

    # =========================================================================
    # MAIN EVALUATION METHODS
    # =========================================================================

    def evaluate(
        self,
        generated_texts: list[str],
        reference_texts: Optional[list[str]] = None,
    ) -> MDQAMetrics:
        """
        Run full MDQA evaluation.

        Args:
            generated_texts: List of generated review texts
            reference_texts: Optional list of reference texts for comparison

        Returns:
            MDQAMetrics with all computed scores
        """
        metrics = MDQAMetrics()

        # Corpus Diversity metrics (always computed)
        metrics.distinct_1 = self.compute_distinct_n(generated_texts, n=1)
        metrics.distinct_2 = self.compute_distinct_n(generated_texts, n=2)
        metrics.self_bleu = self.compute_self_bleu(generated_texts)

        # Reference-based metrics (require references)
        if reference_texts:
            # Ensure same length for comparison
            min_len = min(len(generated_texts), len(reference_texts))
            gen_subset = generated_texts[:min_len]
            ref_subset = reference_texts[:min_len]

            # Lexical Quality
            metrics.bleu = self.compute_bleu(gen_subset, ref_subset)
            metrics.rouge_l = self.compute_rouge_l(gen_subset, ref_subset)

            # Semantic Similarity
            metrics.bertscore = self.compute_bertscore(gen_subset, ref_subset)
            metrics.moverscore = self.compute_moverscore(gen_subset, ref_subset)

        return metrics

    def evaluate_from_reviews(
        self,
        reviews: list,
        text_field: str = "text",
        reference_texts: Optional[list[str]] = None,
    ) -> MDQAMetrics:
        """
        Evaluate from review objects.

        Args:
            reviews: List of review objects/dicts
            text_field: Field name containing review text
            reference_texts: Optional reference texts

        Returns:
            MDQAMetrics with all computed scores
        """
        if isinstance(reviews[0], dict):
            texts = [r[text_field] for r in reviews]
        else:
            texts = [getattr(r, text_field) for r in reviews]

        return self.evaluate(texts, reference_texts)
