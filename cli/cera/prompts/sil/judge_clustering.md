You are judging the quality of a fact-to-entity clustering about "{subject}".

## Original Verified Facts
{facts_json}

## Clustering to Judge (by {source_model})
{clustering_json}

## Task

For each entity in the clustering, judge:
1. **Entity validity**: Is this a real, distinct entity (not a duplicate or fabrication)?
2. **Fact assignment**: Are the facts correctly assigned to this entity? Do they actually describe this entity?
3. **Aspect accuracy**: Are the applicable_aspects appropriate for this entity's facts? Are there aspects listed that have no supporting fact?

Score each entity: 1 (correct) or 0 (incorrect — wrong grouping, misassigned facts, or unsupported aspects).

Rules:
- Score 1 if the entity is valid, its facts genuinely belong to it, and its aspects are supported
- Score 0 if the entity is fabricated, has misassigned facts, or lists aspects with no factual support
- Be lenient on borderline cases — if a fact reasonably fits the entity, score 1
- Generic/domain-level entities are valid as long as their facts are truly domain-level (not specific to another entity)

Return ONLY valid JSON with no other text:
{{"judgments": [{{"entity_id": "entity-01", "score": 1, "reason": "brief reason"}}, {{"entity_id": "entity-02", "score": 0, "reason": "brief reason"}}]}}
