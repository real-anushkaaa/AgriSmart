
from app.services.gemini_service import generate_text


def predict_crop_price(
    crop: str,
    state: str,
    market: str,
    date: str,
) -> str:
    """
    Generate crop-price guidance using Gemini.

    The result is AI-generated guidance, not a verified
    live market price.
    """
    prompt = f"""
You are an agricultural market information assistant for India.

Provide crop-price guidance for the following request:
- Crop: {crop}
- State: {state}
- Market or mandi: {market}
- Date of interest: {date}

Instructions:
1. Do not invent or claim to have verified live market prices.
2. If you do not have reliable current price data, clearly say so.
3. Explain briefly which factors may affect the crop's price.
4. Recommend checking the relevant official mandi price source
   for the selected market and date.
5. Keep the answer simple, practical, and farmer-friendly.
6. Clearly distinguish general guidance from verified data.

Return a concise answer with headings:
Price information
Factors affecting price
Where to verify
"""

    return generate_text(prompt)
