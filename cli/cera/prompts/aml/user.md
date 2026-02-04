Write a review for "{subject}" from the perspective of the following reviewer:

## Reviewer Profile
{reviewer_profile}

## Review Requirements
- Write exactly {num_sentences} sentences
- Include specific details about the product/service
- Reference relevant features naturally in your review

## Sentence-Level Sentiment Requirements
Each sentence must match the specified sentiment below:
{sentence_polarities}

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
- Write EXACTLY {num_sentences} sentences, each matching its specified sentiment
- Each sentence must have at least one opinion annotation
- Use ONLY categories from the list above
- Polarity per opinion must match the sentiment specified for that sentence
- Write authentically as the reviewer described above
- Output ONLY the JSON object, no other text
