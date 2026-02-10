Generate {variant_count} distinct review structure templates for writing authentic {domain} reviews about "{subject}".

These templates define HOW a review is organized — the flow, connective style, evidence approach, and sentiment arc. Each template should produce a noticeably different review shape when followed.

## Context
Region: {region}
{reviewer_context}

## Dimensions to Vary
For each template, make a distinct combination of these dimensions:

1. **Review Flow** — The macro structure of the review
   - Examples: "intro → body → verdict", "problem → solution", "comparison story", "feature tour", "stream of consciousness", "before/after narrative"

2. **Sentiment Arc** — How the overall feeling progresses through the review
   - Examples: "consistently positive", "starts critical then won over", "positive but ends with caveat", "mixed throughout", "builds frustration", "neutral/informative"

3. **Connective Style** — How sentences relate to each other
   - Examples: "logical (However, Therefore, As a result)", "narrative (Then, Eventually, After that)", "abrupt/fragmented (no connectives, just jumps)", "conversational (So, And, But, Like)", "comparative (Unlike, Compared to, On the other hand)"

4. **Evidence Approach** — How opinions are supported
   - Examples: "bare assertions (just states opinions)", "specific details (cites numbers, measurements)", "comparative (references alternatives or past experience)", "anecdotal (tells mini-stories)", "mixed (some backed up, some just stated)"

5. **Sentence Rhythm** — The pattern of sentence lengths
   - Examples: "short punchy (most sentences 5-12 words)", "long flowing (compound sentences with multiple clauses)", "varied (alternates short and long)", "builds up (starts short, gets longer)", "front-loaded (long detail sentence then short reactions)"

## Partial Templates
NOT every template needs all 5 dimensions. About 30-40% of templates should intentionally omit 1-2 dimensions — set those fields to null. This represents reviewers who don't have a conscious approach to every dimension (e.g., someone who just dumps opinions with no particular flow or sentiment arc). The remaining 60-70% should have all dimensions filled.

## Requirements
- Each template must feel like a different person's natural writing approach
- Templates should be usable across any review polarity (positive, negative, mixed)
- Keep descriptions concise — 1 sentence per dimension
- Avoid overlap: if one template uses "logical connectives + feature tour", another should NOT

## Output Format
Return a JSON array. Output ONLY the JSON, no other text.

```json
[
  {{
    "id": "structure-01",
    "name": "The Storyteller",
    "flow": "Narrative arc: sets the scene, describes the experience chronologically, ends with current feelings",
    "sentiment_arc": "Starts neutral/curious, builds to strong opinion by the end",
    "connectives": "Narrative transitions: 'So', 'Then', 'After a week', 'Eventually'",
    "evidence": "Anecdotal — supports opinions with mini-stories and specific moments",
    "rhythm": "Varied — short reactions mixed with longer descriptive sentences"
  }},
  {{
    "id": "structure-02",
    "name": "The Quick Dumper",
    "flow": null,
    "sentiment_arc": null,
    "connectives": "Abrupt — no connectives, just jumps between thoughts",
    "evidence": "Bare assertions — states opinions without supporting details",
    "rhythm": "Short punchy — most sentences under 10 words"
  }}
]
```
