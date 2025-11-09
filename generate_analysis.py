import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_trade_report(stock, trend, entry, stop, target, proba):
    prompt = f"""
    Write a short, professional trading note for {stock}.
    Data:
    - Trend: {trend}
    - Entry: ₹{entry}
    - Stop-Loss: ₹{stop}
    - Target: ₹{target}
    - Model Probability: {proba:.0%}
    Include:
    1. Trend overview
    2. Trade setup (entry, stop, target)
    3. Risk–Reward estimate
    4. One-paragraph commentary.
    Keep under 150 words.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a professional stock market analyst."},
                {"role":"user","content":prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating report: {e}"
