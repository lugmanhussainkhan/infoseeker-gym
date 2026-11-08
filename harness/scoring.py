"""Answer extraction and scoring: normalized exact match, containment, optional LLM judge."""
import re
import string

import litellm

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def extract_answer(content: str) -> tuple[str, bool]:
    """Last <answer>...</answer> in the final message; falls back to the stripped content."""
    content = content or ""
    found = ANSWER_RE.findall(content)
    if found:
        return found[-1].strip(), True
    return content.strip()[-300:], False


def normalize(s: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s or "")          # trailing disambiguator
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return bool(p) and any(p == normalize(g) for g in golds)


def contains(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return bool(p) and any(normalize(g) and normalize(g) in p for g in golds)


JUDGE_PROMPT = """Decide whether the predicted answer refers to the same thing as the gold answer.
Ignore formatting, capitalization and minor wording differences; a more or less specific name for the same entity counts as correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Reply with exactly one word: CORRECT or INCORRECT."""


async def judge(question: str, gold: str, pred: str, model: str, **kwargs) -> bool | None:
    if not pred.strip():
        return False
    try:
        r = await litellm.acompletion(
            model=model, temperature=0, max_tokens=8,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}],
            **kwargs)
        out = (r.choices[0].message.content or "").strip().upper()
        return out.startswith("CORRECT")
    except Exception:  # noqa: BLE001
        return None
