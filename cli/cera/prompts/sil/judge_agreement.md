You are judging answer agreement for a factual research verification task about "{subject}".

You are **{judge_model}**. Below is a combined report of all queries and all researchers' answers, including yours.

For each query, carefully compare YOUR answer with each other researcher's answer and score agreement. Think through whether the core factual claims match before scoring.

## Agreement Rules
- **1 (agree)**: The other answer conveys the same core factual information as yours
- **0 (disagree)**: The other answer contradicts yours or provides fundamentally different information
- **Numerical tolerance**: For quantities, measurements, or statistics, allow up to 15% deviation (e.g., "16 hours" and "14.5 hours" = agree; "16 hours" and "6 hours" = disagree)
- **Completeness**: If one answer is a subset of another (e.g., lists 3 of 5 items), this is AGREEMENT — the core facts match, one is just more complete
- **Wording**: Ignore surface wording differences. "MacBook Pro" vs "macbook pro" vs "MBP" = agree
- **Scope differences**: If both answers are factually correct but emphasize different aspects of the same topic, this is AGREEMENT as long as no claims directly contradict each other

## Judging Process
For each query:
1. Identify the core factual claims in YOUR answer
2. Identify the core factual claims in the other researcher's answer
3. Check if any claims directly contradict — if none contradict, lean toward agreement
4. Apply numerical tolerance for quantitative comparisons
5. Score: 1 if core facts align, 0 only if there is a genuine factual contradiction

COMBINED REPORT:
{topics_json}

For each query, score EACH other researcher's answer: 1 = agree, 0 = disagree. Do NOT score yourself.

Return ONLY valid JSON with no other text:
{{"judgments": [{{"query_id": "q1", "scores": {{"other_model_name": 1, "another_model_name": 0}}}}, ...]}}
