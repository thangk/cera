You are researching "{subject}" to gather factual information for review generation.

First, determine:
1. What TYPE of thing is this? (electronics, clothing, food, sports equipment, service, etc.)
2. What aspects are RELEVANT to research for this type of subject?
3. What would reviewers typically praise or complain about for this type of subject?

Then generate 3-5 search queries to find factual information about this subject.
Focus on aspects that are relevant to the type of subject identified.

Return your response as JSON in this exact format:
```json
{{
  "subject_type": "the category/type of the subject",
  "relevant_aspects": ["aspect1", "aspect2", "aspect3"],
  "search_queries": ["query1", "query2", "query3"]
}}
```

Be specific and practical in your queries. Target real product specifications, verified reviews, and factual information.
