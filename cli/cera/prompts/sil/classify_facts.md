Classify the following verified facts about "{subject}" into categories.

Each fact is a query-answer pair. Based on the nature of the answer, classify it into exactly ONE category.

VERIFIED FACTS:
{facts_json}

Categories:
- **characteristics**: Key attributes, specifications, features, properties, dimensions, materials, technical details
- **positives**: Things that are advantages, strengths, praised aspects, or beneficial qualities
- **negatives**: Things that are disadvantages, weaknesses, common complaints, limitations, or drawbacks
- **use_cases**: Typical usage scenarios, target audiences, occasions, or applications

Rules:
- Each fact goes into exactly ONE category
- Neutral specs (e.g., "16GB RAM") go into characteristics
- If a fact reveals both positive and negative, choose the dominant aspect
- If a fact describes who uses it or when, it goes into use_cases

Return ONLY valid JSON with no other text:
{{
  "characteristics": ["fact1 answer", "fact2 answer", ...],
  "positives": ["fact3 answer", ...],
  "negatives": ["fact4 answer", ...],
  "use_cases": ["fact5 answer", ...]
}}
