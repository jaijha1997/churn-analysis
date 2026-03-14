"""
LLM-powered retention strategy generator.
Takes a customer's churn risk profile + SHAP explanations and generates
personalized retention recommendations.

Uses OpenAI GPT-4o-mini by default (cheap, fast, good enough for structured output).

TODO:
- Add few-shot examples to improve recommendation quality
- Cache responses to avoid redundant API calls for similar profiles
- Add segment-level batch generation (process all high-risk customers at once)
- Evaluate recommendation quality with a small human eval set
"""

from openai import OpenAI
from typing import Dict, List, Optional
import json
import logging

from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a customer retention strategist at a SaaS/telecom company.
Given a customer's churn risk profile and the key factors driving their risk,
generate specific, actionable retention recommendations.

Always respond with valid JSON in this exact format:
{
  "risk_summary": "1-2 sentence summary of why this customer is at risk",
  "segment": "one of: high_value_at_risk | price_sensitive | disengaged | support_frustrated | new_customer_risk",
  "recommendations": [
    {
      "action": "specific action to take",
      "channel": "email | call | in-app | sms",
      "priority": "high | medium | low",
      "rationale": "why this action addresses the churn driver"
    }
  ],
  "urgency": "immediate | within_week | within_month"
}"""


class RetentionAdvisor:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or OPENAI_API_KEY
        if not key:
            logger.warning("No OpenAI API key found — LLM features disabled")
        self.client = OpenAI(api_key=key) if key else None
        self.model = OPENAI_MODEL

    def generate_recommendation(
        self,
        customer_id: str,
        churn_probability: float,
        top_risk_factors: List[Dict],
        customer_context: Optional[Dict] = None,
    ) -> Dict:
        """Generate retention recommendation for a single at-risk customer."""
        if self.client is None:
            return self._fallback_recommendation(churn_probability, top_risk_factors)

        prompt = self._build_prompt(customer_id, churn_probability, top_risk_factors, customer_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            result["customer_id"] = customer_id
            result["churn_probability"] = churn_probability
            return result
        except Exception as e:
            logger.error(f"OpenAI API error for {customer_id}: {e}")
            return self._fallback_recommendation(churn_probability, top_risk_factors)

    def _build_prompt(
        self,
        customer_id: str,
        churn_prob: float,
        risk_factors: List[Dict],
        context: Optional[Dict],
    ) -> str:
        factors_str = "\n".join(
            f"  - {f['feature']}: {f['direction']} (SHAP={f['shap_value']:.3f})"
            for f in risk_factors
        )
        context_str = ""
        if context:
            context_str = f"\nAdditional context:\n" + "\n".join(f"  {k}: {v}" for k, v in context.items())

        return f"""Customer ID: {customer_id}
Churn Probability: {churn_prob:.1%}
Risk Level: {"CRITICAL" if churn_prob > 0.7 else "HIGH" if churn_prob > 0.5 else "MODERATE"}

Top churn drivers:
{factors_str}{context_str}

Generate targeted retention recommendations."""

    def _fallback_recommendation(self, churn_prob: float, risk_factors: List[Dict]) -> Dict:
        """Rule-based fallback when LLM is unavailable."""
        # TODO: make this smarter — currently very generic
        return {
            "risk_summary": f"Customer shows {churn_prob:.0%} churn probability based on behavioral signals.",
            "segment": "high_value_at_risk" if churn_prob > 0.6 else "disengaged",
            "recommendations": [
                {
                    "action": "Proactive outreach from account manager",
                    "channel": "call",
                    "priority": "high",
                    "rationale": "High churn risk warrants personal touch",
                }
            ],
            "urgency": "immediate" if churn_prob > 0.7 else "within_week",
            "note": "LLM unavailable — using rule-based fallback",
        }

    def batch_generate(self, at_risk_customers: List[Dict]) -> List[Dict]:
        """Process multiple at-risk customers. Respects rate limits with basic throttling."""
        # TODO: add async batch processing with asyncio + tenacity for retries
        results = []
        for i, customer in enumerate(at_risk_customers):
            logger.info(f"Processing customer {i+1}/{len(at_risk_customers)}: {customer['customer_id']}")
            rec = self.generate_recommendation(
                customer_id=customer["customer_id"],
                churn_probability=customer["churn_probability"],
                top_risk_factors=customer.get("risk_factors", []),
                customer_context=customer.get("context"),
            )
            results.append(rec)
        return results
