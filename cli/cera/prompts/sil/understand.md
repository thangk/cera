You are researching "{subject}" to gather factual information for review generation.

First, determine:
1. What TYPE of thing is this? Identify the specific category or domain.
   Examples: smartphone, restaurant, hotel, movie, video game, airline, skincare product, hiking trail, online course, coffee shop, etc.
   Be specific - don't just say "product" or "service", identify exactly what it is.

2. What aspects are RELEVANT to research for this type of subject?
   Think about what matters for this specific type of thing.
   What would someone researching this subject want to know?

3. What would reviewers typically praise or complain about?
   Think from a real customer's perspective for this specific subject type.

Then generate 3-10 search queries to find factual information about this subject.
Focus on aspects that are relevant to the type of subject identified.

Return ONLY valid JSON with no other text:
{{
  "subject_type": "the specific type (e.g., 'smartphone', 'italian restaurant', 'action movie', 'hiking boots')",
  "relevant_aspects": ["aspect1", "aspect2", ...],
  "search_queries": ["query1", "query2", ...]
}}
