You are analyzing verified facts about "{subject}" to identify distinct entities.

## Verified Facts
{facts_json}

## Available Aspect Categories
{aspect_categories}

## Task

Group these facts by the distinct entities they describe. An entity can be:
- A specific named item (product, place, person, establishment, etc.)
- A generic/domain-level observation not tied to one specific item

For each entity:
1. **Identify** it with a clear name and short description
2. **Group** all facts (characteristics, positives, negatives) that belong to it
3. **Assign applicable aspects** from the available categories — only include aspects that are supported by this entity's facts. If there's no fact related to an aspect, don't include it.

Rules:
- Every fact must be assigned to exactly ONE entity
- If a fact compares multiple entities, assign it to the entity it's primarily about, or to a "generic" entry
- Generic entries capture domain-level observations not tied to a specific entity
- There should be at least 1 entity (even if all facts are generic)
- Each entity must have at least 1 applicable aspect
- Do NOT fabricate facts — only use information present in the verified facts above

Return ONLY valid JSON with no other text:
{output_schema}
