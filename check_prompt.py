"""Server check: does the chat template actually see each turn's reasoning?

Renders a stored episode through the server's /tokenize + /detokenize (vLLM) and reports, per
assistant turn, whether the reasoning appears in the prompt. Two extra renders show which
message field the server forwards to the template, to help fix the serving side if needed.

Usage: python check_prompt.py runs/smoke.jsonl --server http://localhost:8000 [--index 0] [--show]
"""
import argparse
import json

import httpx

from harness.env import TOOLS


def render(srv, model, msgs, extra):
    body = {"model": model, "messages": msgs, "tools": TOOLS, "add_generation_prompt": False, **extra}
    r = httpx.post(f"{srv}/tokenize", json=body, timeout=60)
    r.raise_for_status()
    tokens = r.json()["tokens"]
    text = httpx.post(f"{srv}/detokenize", json={"model": model, "tokens": tokens}, timeout=60).json()["prompt"]
    return text, len(tokens)


def reasoning_of(m):
    return m.get("reasoning_content") or m.get("reasoning")


def only_field(msgs, field):
    out = []
    for m in msgs:
        m = dict(m)
        r = m.pop("reasoning_content", None) or m.pop("reasoning", None)
        if m["role"] == "assistant" and r:
            m[field] = r
        out.append(m)
    return out


def report(text, msgs):
    return ["none stored" if not reasoning_of(m) else
            ("RENDERED" if reasoning_of(m).strip()[:200] in text else "missing")
            for m in msgs if m["role"] == "assistant"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_file")
    ap.add_argument("--server", required=True, help="server base URL (with or without /v1)")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--extra-body", default=None, help="e.g. chat_template_kwargs, same as run.py")
    ap.add_argument("--show", action="store_true", help="print the rendered prompt")
    a = ap.parse_args()

    msgs = [json.loads(l) for l in open(a.run_file) if l.strip()][a.index]["messages"]
    srv = a.server.rstrip("/").removesuffix("/v1")
    model = httpx.get(f"{srv}/v1/models").json()["data"][0]["id"]
    extra = json.loads(a.extra_body) if a.extra_body else {}

    for name, v in [("as stored", msgs),
                    ("field `reasoning_content`", only_field(msgs, "reasoning_content")),
                    ("field `reasoning`", only_field(msgs, "reasoning"))]:
        text, n = render(srv, model, v, extra)
        if name == "as stored" and a.show:
            print(text + "\n" + "=" * 80)
        print(f"{name:<26} {n:>6} tokens   per assistant turn: {report(text, v)}")
    print("\n'as stored' should be RENDERED on every turn. If not, the server is not passing the "
          "reasoning to the chat template; the other two lines show which field name it forwards.")


if __name__ == "__main__":
    main()
