Analyze the following sample reviews to identify the geographic region or location context.

## Sample Reviews

{reviews_json}

## Your Task

Identify the geographic region based on:
- Place names, cities, neighborhoods mentioned
- Currency symbols or references ($, £, etc.)
- Language patterns (British vs American English spellings)
- Cultural references, local terminology
- Units of measurement (miles vs kilometers, Fahrenheit vs Celsius)

## Output Format

Return a JSON object with:
- "region": The identified region (e.g., "United States", "United Kingdom", "Western Europe", "Southeast Asia", "New York City", "California")
- "confidence": Your confidence level from 0.0 to 1.0
- "reason": Brief explanation if confidence is low or region is unclear

If the region cannot be determined, return:
{{"region": null, "confidence": 0.0, "reason": "No clear geographic indicators found"}}

Example outputs:
{{"region": "United Kingdom", "confidence": 0.85, "reason": null}}
{{"region": "San Francisco Bay Area", "confidence": 0.92, "reason": null}}
{{"region": null, "confidence": 0.1, "reason": "Mixed indicators from multiple regions"}}

Return ONLY the JSON object, nothing else.
