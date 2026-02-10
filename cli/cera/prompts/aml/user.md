Write a review for "{subject}" as the following reviewer:

## Your Persona
{persona_text}

## Writing Assignments
**Opening:** {opening_directive}
**Capitalization:** {capitalization_style}
{writing_pattern_assignments}

## Features to Mention
Positive: {pros}
Negative: {cons}

{style_examples}

## Sentence Plan
Write exactly {num_sentences} sentences following this plan:
{aspect_sentence_plan}

## Output Format
Output ONLY a JSON object with per-sentence aspect annotations.
{dataset_mode_instruction}

```json
{output_example}
```

Rules:
- Write EXACTLY {num_sentences} sentences matching the sentence plan above
- Each sentence with assigned aspects MUST include those opinion annotations
- Contextual sentences (no aspects) use an empty opinions array
- Write authentically as the persona described above
- Output ONLY the JSON, no other text

{vocab_diversity}

{neb_context}
