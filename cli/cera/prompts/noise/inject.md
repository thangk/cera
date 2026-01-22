# Noise Injection Guidelines

This document describes the types of noise injected into generated reviews to make them more authentic.

## Noise Types

### 1. Typos (typo_rate: {typo_rate})
Common keyboard errors:
- Adjacent key substitutions (teh → the)
- Missing letters (becuse → because)
- Doubled letters (untill → until)
- Transposed letters (recieve → receive)

### 2. Colloquialisms (enabled: {colloquialism})
Informal language patterns:
- Contractions (I am → I'm, do not → don't)
- Casual phrases (gonna, wanna, kinda)
- Abbreviated words (prob, def, tbh)
- Regional expressions based on reviewer's region

### 3. Grammar Variations (enabled: {grammar_errors})
Minor grammatical imperfections:
- Subject-verb agreement in casual speech
- Run-on sentences
- Missing punctuation
- Comma splices
- Starting sentences with conjunctions

## Application Rules
- Noise is applied post-generation to preserve LLM output quality
- Noise density varies by preset (none, light, moderate, heavy)
- Advanced noise includes OCR-style errors and contextual substitutions
- Noise is distributed naturally, not uniformly
