You are researching "{subject}" to generate factual queries about it.
{additional_context_block}
{research_context_block}

Generate SPECIFIC, NEUTRAL, FACTUAL QUERIES about this subject. Each query should:
- Ask about ONE verifiable fact
- Be neutral (not biased toward positive or negative)
- Have a concise, factual answer
- Cover diverse aspects of the subject

Good examples:
- "What RAM options does {subject} have?"
- "Does {subject} have accessibility features?"
- "What payment methods does {subject} accept?"
- "What is the screen size of {subject}?"
- "What are the operating hours of {subject}?"
- "Does {subject} offer delivery?"
- "What materials is {subject} made from?"

Bad examples (DO NOT generate these):
- "Is the battery life bad?" (negative bias)
- "What do people love about it?" (opinion-seeking)
- "Why is this the best?" (positive bias)
- "What are the worst features?" (negative bias)

Generate as many relevant queries as you can. Cover specifications, features, availability, services, materials, pricing, compatibility, and any other factual aspects relevant to this type of subject.

Return ONLY valid JSON with no other text:
{{"queries": ["query1", "query2", "query3", ...]}}
