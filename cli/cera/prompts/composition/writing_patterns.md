Analyze the domain "{domain}" for the subject "{subject}" and generate natural writing variation patterns that real reviewers use.

{reference_context}

## Your Task
Identify domain-specific terms, measurements, brand names, and quantities that reviewers commonly mention, then provide multiple natural ways real people write each one.

## Requirements
For each pattern category:
- **context**: When this pattern applies (1 sentence)
- **options**: 4-6 natural variations real reviewers would use, ranging from formal to casual

## Guidelines
- Only include patterns relevant to this specific domain and subject
- Include both formal ("16GB RAM") and casual ("16 gigs") variants
- Include common misspellings or informal abbreviations real people use
- Consider the region ({region}) for currency, units, and terminology
- Generate 5-10 pattern categories that are most relevant to this subject

## Output Format
Return a JSON object. Output ONLY the JSON, no other text.

```json
{{
  "domain": "{domain}",
  "patterns": {{
    "category_name": {{
      "context": "When referring to ...",
      "options": ["formal version", "casual version", "abbreviated", "colloquial"]
    }}
  }}
}}
```
