Based on the search results below, extract factual information about "{subject}".

## Search Results
{search_results}

## Extraction Requirements
Extract and categorize the information into these categories:

1. **Characteristics**: Key attributes, specifications, ingredients, materials, etc.
2. **Positives**: Things reviewers typically praise about this subject
3. **Negatives**: Things reviewers typically complain about
4. **Use Cases**: When, where, or how this subject is typically used
5. **Availability**: Price range, where to buy, variants available

Return your extraction as JSON in this exact format:
```json
{{
  "characteristics": ["char1", "char2", "char3"],
  "positives": ["positive1", "positive2", "positive3"],
  "negatives": ["negative1", "negative2"],
  "use_cases": ["use_case1", "use_case2"],
  "availability": "price and availability summary"
}}
```

Be factual and specific. Only include information that is supported by the search results.
Do not make up or assume information that wasn't found.
