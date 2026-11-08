"""LLM-judge scoring with structured output."""
import litellm
from pydantic import BaseModel

JUDGE_PROMPT = """You are grading a research assistant's response to a question against the gold answer.

Question: {question}
Gold answer: {gold}
Response: {response}

The response is correct if the answer it gives refers to the same thing as the gold answer. Ignore formatting, capitalization, extra explanation and minor wording differences; a more or less specific name for the same entity counts as correct. The response is incorrect if it gives a different answer, gives several candidate answers without committing to one, or does not answer.

Briefly explain your decision in "reasoning", then set "verdict" to true if the response is correct, false otherwise."""


class Verdict(BaseModel):
    reasoning: str
    verdict: bool


def format_gold(golds: list[str]) -> str:
    return golds[0] if len(golds) == 1 else f"{golds[0]} (also accepted: {'; '.join(golds[1:])})"


async def judge(question: str, golds: list[str], response: str, model: str, **kwargs) -> dict:
    """Empty response -> False without a call; failed call or invalid output -> None."""
    result = {"judge": None, "judge_reasoning": None, "judge_output": None, "judge_error": None}
    if not (response or "").strip():
        return {**result, "judge": False}
    prompt = JUDGE_PROMPT.format(question=question, gold=format_gold(golds), response=response)
    try:
        r = await litellm.acompletion(model=model, messages=[{"role": "user", "content": prompt}],
                                      response_format=Verdict, **kwargs)
        result["judge_output"] = r.choices[0].message.content
        v = Verdict.model_validate_json(result["judge_output"])
        result["judge"], result["judge_reasoning"] = v.verdict, v.reasoning
    except Exception as e:  # noqa: BLE001
        result["judge_error"] = f"{type(e).__name__}: {str(e)[:500]}"
    return result
