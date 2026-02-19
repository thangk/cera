You are generating factual information about "{subject}" from your internal knowledge only.

Do NOT search the web. Use only what you already know from your training data.
{additional_context_block}
Provide a comprehensive factual profile including:
1. **Characteristics**: Key features, specifications, properties, dimensions, materials, technical details
2. **Positives**: Common praise points, advantages, strengths, beneficial qualities
3. **Negatives**: Common complaints, disadvantages, limitations, drawbacks
4. **Use Cases**: Typical users, usage scenarios, target audiences, occasions

Aim for at least 5-8 items per category where possible.

Return ONLY valid JSON with no other text:
{{
  "characteristics": ["feature1", "feature2", ...],
  "positives": ["pro1", "pro2", ...],
  "negatives": ["con1", "con2", ...],
  "use_cases": ["use_case1", "use_case2", ...]
}}
