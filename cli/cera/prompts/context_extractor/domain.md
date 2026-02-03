Analyze the following sample reviews to identify the domain/category of the subject being reviewed.

## Sample Reviews

{reviews_json}

## Your Task

Identify the most specific domain that describes what is being reviewed.

Common domains include (but are not limited to):
- Restaurant, Cafe, Bar, Fast Food
- Hotel, Hostel, Resort, Vacation Rental
- Laptop, Smartphone, Tablet, Electronics, Appliances
- Software, App, SaaS, Website
- Movie, Book, Music, Game
- Clothing, Beauty, Health
- Automotive, Travel, Airlines
- Service (specify type)

## Output Format

Return a JSON object with:
- "domain": The specific domain name (1-3 words, e.g., "Italian Restaurant", "Budget Hotel", "Gaming Laptop")
- "confidence": Your confidence level from 0.0 to 1.0

Example output:
{{"domain": "Fine Dining Restaurant", "confidence": 0.95}}

Return ONLY the JSON object, nothing else.
