
from app.services.gemini_service import generate_text


def generate_crop_calendar(
    crop: str,
    location: str,
    season: str,
    start_month: int,
) -> str:
    prompt = f"""
You are an agricultural planning assistant for Indian farmers.

Create a practical crop calendar using these details:
- Crop: {crop}
- Location: {location}
- Season: {season}
- Starting month: {start_month}

Include these stages in chronological order:
1. Land preparation
2. Seed selection and sowing
3. Irrigation and water management
4. Nutrient and fertilizer management
5. Weed, pest, and disease monitoring
6. Crop growth and maintenance
7. Harvesting and post-harvest handling

Instructions:
- Give approximate timing for each stage.
- Adapt recommendations to the specified crop, location, and season.
- Do not invent official government recommendations or claim to have
  checked live weather or local agricultural data.
- Clearly label timing as general guidance where local conditions
  may change the schedule.
- Mention that rainfall, soil type, variety, and local advice can
  affect the calendar.
- Keep the language simple and farmer-friendly.
- Use a table with columns: Stage, Approximate timing, Activities.
- Finish with a short section called "Important advice".
"""

    return generate_text(prompt)
