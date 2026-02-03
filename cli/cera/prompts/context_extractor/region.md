Analyze the following sample reviews to identify the geographic region(s) or location context.

## Sample Reviews

{reviews_json}

## Your Task

Identify the geographic region(s) based on:
- Place names, cities, neighborhoods mentioned
- Currency symbols or references ($, £, etc.)
- Language patterns (British vs American English spellings)
- Cultural references, local terminology
- Units of measurement (miles vs kilometers, Fahrenheit vs Celsius)

## Important: Multiple Regions

Reviews may span **multiple distinct geographic regions**. Do NOT force a single location when the data suggests multiple regions are present.

- If reviews clearly come from ONE region → return that specific region
- If reviews span MULTIPLE identifiable regions → list ALL identified regions (comma-separated or as a descriptive phrase)
- If reviews span many regions or are globally diverse → use a broad descriptor like "Multiple regions worldwide", "Various European cities", "International", etc.

## Output Format

Return a JSON object with:
- "region": The identified region(s). Can be:
  - A single region: "United States", "United Kingdom"
  - Multiple specific regions: "Amsterdam, Paris, London", "Western Europe (Netherlands, France, UK)"
  - A broad descriptor: "Multiple European cities", "International", "Various North American cities"
- "confidence": Your confidence level from 0.0 to 1.0
- "reason": Brief explanation, especially useful for multi-region results

If the region cannot be determined, return:
{{"region": null, "confidence": 0.0, "reason": "No clear geographic indicators found"}}

Example outputs:
{{"region": "United Kingdom", "confidence": 0.85, "reason": null}}
{{"region": "San Francisco Bay Area", "confidence": 0.92, "reason": null}}
{{"region": "Amsterdam, Paris, Berlin", "confidence": 0.78, "reason": "Reviews mention multiple European capitals"}}
{{"region": "Western Europe (Netherlands, France, Germany)", "confidence": 0.80, "reason": "Reviews span several Western European countries"}}
{{"region": "International", "confidence": 0.70, "reason": "Reviews from diverse global locations including US, Europe, and Asia"}}

Return ONLY the JSON object, nothing else.
