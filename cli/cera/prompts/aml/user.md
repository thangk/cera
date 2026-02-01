Write a review for "{subject}" from the perspective of the following reviewer:

## Reviewer Profile
- Age: {age}
- Sex: {sex}
- Region: {region}
- Background: {additional_context}

## Review Requirements
- Length: {min_sentences}-{max_sentences} sentences
- Include specific details about the product/service
- Reference relevant features naturally in your review

## Sentence Sentiment Distribution
Distribute sentence-level sentiments according to these proportions:
- Positive sentences: {polarity_positive}%
- Neutral sentences: {polarity_neutral}%
- Negative sentences: {polarity_negative}%

For a {min_sentences}-{max_sentences} sentence review, include a mix of sentiments. For example:
- 4 sentences with 60% positive → ~2-3 positive, ~1 neutral, ~0-1 negative
- Balance naturally - not every review needs all sentiment types

## Aspect Categories
Use ONLY these aspect categories for annotations:
{aspect_categories}

## Output Format
Output the review as a JSON object with per-sentence aspect annotations.
{dataset_mode_instruction}

```json
{output_example}
```

Rules:
- Each sentence must have at least one opinion annotation
- Use ONLY categories from the list above
- Polarity per opinion can be "positive", "neutral", or "negative"
- Mix sentiments naturally across sentences based on the distribution above
- Write authentically as the reviewer described above
- Output ONLY the JSON object, no other text
