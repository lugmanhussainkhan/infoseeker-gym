"""Append-only agent loop: each reply and its tool results are appended; the full history is sent every turn."""
import time

import litellm

from .env import TOOLS, SearchEnv

SYSTEM_PROMPT = """You are a research assistant with access to a search tool. Research the user's question using the tools, and answer based on what you find. You can make at most {max_tool_calls} tool calls.

When you are done, reply to the user with your answer without calling a tool."""

USER_TEMPLATE = "{question}"


def as_dict(msg) -> dict:
    """The reply as a dict, with provider_specific_fields (e.g. original `reasoning`) restored at top level."""
    d = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else dict(msg)
    d = {k: v for k, v in d.items() if v is not None}
    for k, v in (d.pop("provider_specific_fields", None) or {}).items():
        if v is not None:
            d.setdefault(k, v)
    return d


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {}
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    out = {"prompt_tokens": d.get("prompt_tokens") or 0, "completion_tokens": d.get("completion_tokens") or 0}
    details = d.get("completion_tokens_details") or {}
    if isinstance(details, dict) and details.get("reasoning_tokens"):
        out["reasoning_tokens"] = details["reasoning_tokens"]
    return out


async def run_episode(question: str, env: SearchEnv, model: str, completion_kwargs: dict,
                      max_tool_calls: int = 20, max_turns: int = 40) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls)},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    tool_log, usage_log = [], []
    tool_calls_made, status, error = 0, "max_turns", None
    t0 = time.time()

    for _ in range(max_turns):
        try:
            resp = await litellm.acompletion(model=model, messages=messages, tools=TOOLS,
                                             tool_choice="auto", **completion_kwargs)
        except litellm.ContextWindowExceededError as e:
            status, error = "context_limit", str(e)[:500]
            break
        except Exception as e:  # noqa: BLE001 - record and stop this episode
            status, error = "model_error", f"{type(e).__name__}: {str(e)[:500]}"
            break

        choice = resp.choices[0]
        usage_log.append(_usage(resp))
        msg = as_dict(choice.message)
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            status = "answered" if choice.finish_reason != "length" else "truncated"
            break

        for tc in calls:
            name, args = tc["function"]["name"], tc["function"]["arguments"]
            if tool_calls_made >= max_tool_calls:
                text, meta = ("Tool budget exhausted. Do not call tools again; "
                              "give your final answer now."), {"error": "budget"}
            else:
                text, meta = await env.call(name, args)
                tool_calls_made += 1
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": text})
            tool_log.append({"name": name, "arguments": args, **meta})

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "final_content": (messages[-1].get("content") or "") if messages[-1]["role"] == "assistant" else "",
        "num_turns": sum(m["role"] == "assistant" for m in messages),
        "num_tool_calls": tool_calls_made,
        "tool_log": tool_log,
        "usage": {
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_log),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_log),
            "reasoning_tokens": sum(u.get("reasoning_tokens", 0) for u in usage_log),
            "max_prompt_tokens": max((u.get("prompt_tokens", 0) for u in usage_log), default=0),
        },
        "seconds": round(time.time() - t0, 2),
    }
