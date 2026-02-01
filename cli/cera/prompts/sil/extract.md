Extract factual information from these search results about "{subject}".

SEARCH RESULTS:
{search_content}

Extract facts in this EXACT JSON format:
{{
  "characteristics": ["key attribute 1", "key attribute 2", ...],
  "positives": ["thing reviewers praise 1", "thing reviewers praise 2", ...],
  "negatives": ["thing reviewers complain about 1", "thing reviewers complain about 2", ...],
  "use_cases": ["typical use case 1", "typical use case 2", ...],
  "availability": "price/availability info or null"
}}

IMPORTANT:
- characteristics: Key attributes relevant to this specific subject (5-10 items)
  Think: What defines this? What are its main features/specs/properties?

- positives: Things people praise in reviews (5-10 items)
  Think: What do satisfied customers highlight? What works well?

- negatives: Things people complain about in reviews (3-5 items)
  Think: What are common complaints? What disappoints people?

- use_cases: Typical scenarios where this is used/experienced (3-5 items)
  Think: When/why/how do people use or interact with this?

- availability: Price range, where to buy/find, variants available

- Only include information found in the search results
- Be specific and factual
- Return ONLY valid JSON, no other text
