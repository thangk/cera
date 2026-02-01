You are answering factual questions about "{subject}".
{additional_context_block}
{research_context_block}

For each query below, provide a SHORT, DIRECT, COMPLETE answer.

Guidelines:
- Be factual and concise (1-2 sentences max per answer)
- Include ALL relevant details — do not omit options, variants, or specifics
- Rate your confidence: "high" (certain from research), "medium" (likely correct), "low" (uncertain)
- If you truly cannot determine the answer, set response to "unknown" with confidence "low"
- Do NOT guess or fabricate information

QUERIES:
{queries_json}

Return ONLY valid JSON with no other text:
{{"answers": [{{"query_id": "q1", "response": "your concise factual answer", "confidence": "high"}}, ...]}}
