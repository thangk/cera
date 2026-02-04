"""Noise Injection - Adding realistic imperfections using nlpaug."""

import random
import copy
from typing import Optional


class NoiseInjector:
    """
    Noise Injector for adding human-like imperfections.

    Uses nlpaug library for realistic text augmentation:
    - Character level: QWERTY keyboard typos, random char errors
    - Word level: Spelling mistakes, synonym replacements
    - Sentence level: Contextual modifications

    Pipeline: Clean → Character → Word → Sentence → Noisy
    """

    def __init__(
        self,
        typo_rate: float = 0.01,
        colloquialism: bool = True,
        grammar_errors: bool = True,
    ):
        """
        Initialize NoiseInjector.

        Args:
            typo_rate: Rate of character-level typos (0-1, default 0.01 = 1%)
            colloquialism: Enable word-level colloquial replacements
            grammar_errors: Enable sentence-level grammar modifications
        """
        self.typo_rate = typo_rate
        self.colloquialism = colloquialism
        self.grammar_errors = grammar_errors

        # Lazy-load augmenters to avoid import overhead
        self._keyboard_aug = None
        self._spelling_aug = None
        self._random_char_aug = None

    def _get_keyboard_aug(self):
        """Get or create keyboard typo augmenter."""
        if self._keyboard_aug is None:
            import nlpaug.augmenter.char as nac

            self._keyboard_aug = nac.KeyboardAug(
                aug_char_p=self.typo_rate,
                aug_word_p=0.1,  # 10% of words affected
                include_special_char=False,
                include_numeric=False,
                lang="en",
            )
        return self._keyboard_aug

    def _get_random_char_aug(self):
        """Get or create random character augmenter."""
        if self._random_char_aug is None:
            import nlpaug.augmenter.char as nac

            self._random_char_aug = nac.RandomCharAug(
                action="substitute",
                aug_char_p=self.typo_rate * 0.5,  # Half the typo rate
                aug_word_p=0.05,
            )
        return self._random_char_aug

    def _get_spelling_aug(self):
        """Get or create spelling augmenter for colloquialisms."""
        if self._spelling_aug is None:
            import nlpaug.augmenter.word as naw

            self._spelling_aug = naw.SpellingAug(
                aug_p=0.05,  # 5% of words
                aug_min=1,
                aug_max=3,
            )
        return self._spelling_aug

    def _safe_augment(self, aug, text: str) -> str:
        """Safely apply augmentation with error handling."""
        try:
            result = aug.augment(text)
            if isinstance(result, list):
                return result[0] if result else text
            return result if result else text
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            return text

    def inject_character_noise(self, text: str) -> str:
        """
        Apply character-level noise (typos).

        Uses nlpaug KeyboardAug for QWERTY-based realistic typos.
        """
        if self.typo_rate <= 0:
            return text

        aug = self._get_keyboard_aug()
        return self._safe_augment(aug, text)

    def inject_word_noise(self, text: str) -> str:
        """
        Apply word-level noise (spelling/colloquialisms).

        Uses nlpaug SpellingAug for realistic word-level errors.
        """
        if not self.colloquialism:
            return text

        aug = self._get_spelling_aug()
        return self._safe_augment(aug, text)

    def inject_sentence_noise(self, text: str) -> str:
        """
        Apply sentence-level noise (grammar, punctuation).

        Adds minor grammar imperfections for authenticity.
        """
        if not self.grammar_errors:
            return text

        result = text

        # Random chance to remove trailing punctuation
        if random.random() < 0.1:
            result = result.rstrip(".")

        # Random chance to add double space
        if random.random() < 0.05:
            words = result.split()
            if len(words) > 3:
                idx = random.randint(1, len(words) - 2)
                words[idx] = words[idx] + " "
                result = " ".join(words)

        # Random chance to add filler words
        if random.random() < 0.05:
            fillers = ["like", "you know", "basically", "honestly"]
            words = result.split()
            if len(words) > 4:
                idx = random.randint(2, len(words) - 2)
                words.insert(idx, random.choice(fillers))
                result = " ".join(words)

        return result

    def inject_noise(self, text: str) -> str:
        """
        Apply full noise pipeline to text.

        Order: Sentence → Word → Character
        (Reverse order to preserve structure for dependent augmentations)
        """
        result = text

        # 1. Sentence-level (structure modifications)
        if self.grammar_errors:
            result = self.inject_sentence_noise(result)

        # 2. Word-level (colloquialisms, spelling)
        if self.colloquialism:
            result = self.inject_word_noise(result)

        # 3. Character-level last (typos)
        if self.typo_rate > 0:
            result = self.inject_character_noise(result)

        return result

    def inject_noise_batch(
        self,
        reviews: list,
        text_field: str = "text",
    ) -> tuple[list, list]:
        """
        Apply noise to a batch of reviews.

        Returns both noisy and clean versions for comparison/ablation studies.

        Args:
            reviews: List of review objects/dicts
            text_field: Field name containing review text

        Returns:
            Tuple of (noisy_reviews, clean_reviews)
        """
        noisy = []
        clean = []

        for review in reviews:
            if isinstance(review, dict):
                clean_copy = review.copy()
                noisy_copy = review.copy()
                noisy_copy[text_field] = self.inject_noise(review[text_field])
            else:
                # Dataclass or object with attributes
                clean_copy = copy.deepcopy(review)
                noisy_copy = copy.deepcopy(review)
                setattr(noisy_copy, text_field, self.inject_noise(getattr(review, text_field)))

            clean.append(clean_copy)
            noisy.append(noisy_copy)

        return noisy, clean


class AdvancedNoiseInjector(NoiseInjector):
    """
    Advanced noise injector with additional nlpaug augmenters.

    Includes:
    - OCR-style errors (simulates scanned text artifacts)
    - Back-translation noise (via contextual embeddings)
    - Contextual word substitutions (BERT-based)
    """

    def __init__(
        self,
        typo_rate: float = 0.01,
        colloquialism: bool = True,
        grammar_errors: bool = True,
        use_ocr_errors: bool = False,
        use_contextual: bool = False,
    ):
        """
        Initialize AdvancedNoiseInjector.

        Args:
            typo_rate: Rate of character-level typos
            colloquialism: Enable word-level modifications
            grammar_errors: Enable sentence-level modifications
            use_ocr_errors: Enable OCR-style character errors
            use_contextual: Enable BERT-based contextual word substitution
        """
        super().__init__(typo_rate, colloquialism, grammar_errors)
        self.use_ocr_errors = use_ocr_errors
        self.use_contextual = use_contextual
        self._ocr_aug = None
        self._contextual_aug = None

    def _get_ocr_aug(self):
        """Get or create OCR error augmenter."""
        if self._ocr_aug is None:
            import nlpaug.augmenter.char as nac

            self._ocr_aug = nac.OcrAug(
                aug_char_p=self.typo_rate,
                aug_word_p=0.1,
            )
        return self._ocr_aug

    def _get_contextual_aug(self):
        """Get or create contextual word augmenter (BERT-based)."""
        if self._contextual_aug is None:
            import nlpaug.augmenter.word as naw

            self._contextual_aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased",
                action="substitute",
                aug_p=0.1,
                device="cpu",  # Use "cuda" if GPU available
            )
        return self._contextual_aug

    def inject_ocr_noise(self, text: str) -> str:
        """
        Apply OCR-style character errors.

        Simulates artifacts from scanned/OCR'd text.
        """
        if not self.use_ocr_errors:
            return text

        aug = self._get_ocr_aug()
        return self._safe_augment(aug, text)

    def inject_contextual_noise(self, text: str) -> str:
        """
        Apply contextual word substitutions.

        Uses BERT to find contextually appropriate word replacements.
        """
        if not self.use_contextual:
            return text

        aug = self._get_contextual_aug()
        return self._safe_augment(aug, text)

    def inject_noise(self, text: str) -> str:
        """
        Apply full advanced noise pipeline.

        Order: Sentence → Contextual → Word → Character → OCR
        """
        result = text

        # 1. Sentence-level
        if self.grammar_errors:
            result = self.inject_sentence_noise(result)

        # 2. Contextual word substitution (before other word-level)
        if self.use_contextual:
            result = self.inject_contextual_noise(result)

        # 3. Word-level
        if self.colloquialism:
            result = self.inject_word_noise(result)

        # 4. Character-level
        if self.typo_rate > 0:
            result = self.inject_character_noise(result)

        # 5. OCR errors (additional character-level)
        if self.use_ocr_errors:
            result = self.inject_ocr_noise(result)

        return result


NOISE_PRESETS = {
    "none": {"typo_rate": 0.0, "colloquialism": False, "grammar_errors": False},
    "light": {"typo_rate": 0.005, "colloquialism": False, "grammar_errors": True},
    "moderate": {"typo_rate": 0.01, "colloquialism": True, "grammar_errors": True},
    "heavy": {"typo_rate": 0.03, "colloquialism": True, "grammar_errors": True},
}


def create_noise_pipeline(
    typo_rate: float = 0.01,
    colloquialism: bool = True,
    grammar_errors: bool = True,
    advanced: bool = False,
    use_ocr: bool = False,
    use_contextual: bool = False,
    preset: Optional[str] = None,
) -> NoiseInjector:
    """
    Factory function to create appropriate noise injector.

    Args:
        typo_rate: Rate of character-level typos (0-1)
        colloquialism: Enable word-level modifications
        grammar_errors: Enable sentence-level modifications
        advanced: Use AdvancedNoiseInjector with extra features
        use_ocr: Enable OCR-style errors (requires advanced=True)
        use_contextual: Enable BERT-based substitution (requires advanced=True)
        preset: Noise preset name ('none', 'light', 'moderate', 'heavy')
                If specified, overrides typo_rate, colloquialism, grammar_errors

    Returns:
        Configured NoiseInjector instance
    """
    # Apply preset if specified
    if preset is not None:
        if preset == "ref_dataset":
            # ref_dataset preset uses the passed-in values (extracted from reference dataset)
            # Don't override typo_rate, colloquialism, grammar_errors - use what was passed
            pass
        elif preset not in NOISE_PRESETS:
            raise ValueError(f"Invalid preset '{preset}'. Valid: {list(NOISE_PRESETS.keys()) + ['ref_dataset']}")
        else:
            preset_config = NOISE_PRESETS[preset]
            typo_rate = preset_config["typo_rate"]
            colloquialism = preset_config["colloquialism"]
            grammar_errors = preset_config["grammar_errors"]

    if advanced or use_ocr or use_contextual:
        return AdvancedNoiseInjector(
            typo_rate=typo_rate,
            colloquialism=colloquialism,
            grammar_errors=grammar_errors,
            use_ocr_errors=use_ocr,
            use_contextual=use_contextual,
        )
    else:
        return NoiseInjector(
            typo_rate=typo_rate,
            colloquialism=colloquialism,
            grammar_errors=grammar_errors,
        )
