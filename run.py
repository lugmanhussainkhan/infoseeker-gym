"""Run the search agent over a question file and write one JSON line per question.

Example (Qwen served by vLLM on another machine):
  python run.py \
    --questions data/infoseek_hard_6k.jsonl --limit 500 \
    --model openai/Qwen/Qwen3.8-27B --api-base http://gpu-box:8001/v1 \
    --index-url http://index-box:8000 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": true}}' \
    --out runs/qwen3.8-27b.jsonl --concurrency 32

Re-running with the same --out skips questions already written (resume).
"""
import argparse
import asyncio
import json
import os
import random
import sys

import litellm

from harness.agent import run_episode
from harness.env import SearchEnv
from harness.scoring import contains, exact_match, extract_answer, judge

litellm.drop_params = True
litellm.suppress_debug_info = True


def load_questions(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            golds = r.get("answer_aliases") or [r["answer"]]
            rows.append({"id": r.get("id", f"q{i:05d}"), "question": r["question"],
                         "answer": r["answer"], "golds": golds,
                         "meta": {k: r[k] for k in ("num_nodes", "depth", "shape") if k in r}})
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="litellm model string, e.g. openai/<served-name>")
    ap.add_argument("--api-base", default=os.environ.get("MODEL_API_BASE"))
    ap.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", "EMPTY"))
    ap.add_argument("--index-url", default=os.environ.get("INDEX_URL", "http://localhost:8000"))
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192, help="per model turn")
    ap.add_argument("--extra-body", default=None, help="JSON passed to the server, e.g. chat_template_kwargs")
    ap.add_argument("--max-tool-calls", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--search-k", type=int, default=5)
    ap.add_argument("--page-words", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per model call")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--judge-model", default=None, help="litellm model for LLM-judge scoring (optional)")
    ap.add_argument("--judge-api-base", default=None)
    ap.add_argument("--judge-api-key", default=None)
    a = ap.parse_args()

    qs = load_questions(a.questions)
    if a.shuffle_seed is not None:
        random.Random(a.shuffle_seed).shuffle(qs)
    if a.limit:
        qs = qs[: a.limit]
    done = set()
    if os.path.exists(a.out):
        # keep finished episodes; drop ones that failed on a server/connection error so they rerun
        with open(a.out) as fh:
            prev = [json.loads(l) for l in fh if l.strip()]
        kept = [r for r in prev if r["status"] != "model_error"]
        if len(kept) < len(prev):
            print(f"Retrying {len(prev) - len(kept)} episodes that ended in model_error")
            with open(a.out, "w") as fh:
                for r in kept:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = {r["id"] for r in kept}
    todo = [q for q in qs if q["id"] not in done]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    env = SearchEnv(a.index_url, k=a.search_k, page_words=a.page_words)
    try:
        print("index server:", await env.health())
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Cannot reach index server at {a.index_url}: {e}")

    kwargs = {"api_base": a.api_base, "api_key": a.api_key, "temperature": a.temperature,
              "top_p": a.top_p, "max_tokens": a.max_tokens, "timeout": a.timeout}
    if a.extra_body:
        kwargs["extra_body"] = json.loads(a.extra_body)
    judge_kwargs = {k: v for k, v in {"api_base": a.judge_api_base, "api_key": a.judge_api_key}.items() if v}
    config = {k: v for k, v in vars(a).items() if k not in ("api_key", "judge_api_key")}

    print(f"{len(qs)} questions, {len(done)} already done, running {len(todo)}")
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    stats = {"n": 0, "em": 0, "contains": 0}

    async def one(q):
        async with sem:
            ep = await run_episode(q["question"], env, a.model, kwargs, a.max_tool_calls, a.max_turns)
            pred, tagged = extract_answer(ep["final_content"]) if ep["status"] == "answered" else ("", False)
            rec = {
                "id": q["id"], "question": q["question"], "answer": q["answer"], "golds": q["golds"],
                **q["meta"], "model": a.model, "pred": pred, "answer_tagged": tagged,
                "em": exact_match(pred, q["golds"]), "contains": contains(pred, q["golds"]),
                **{k: ep[k] for k in ("status", "error", "num_turns", "num_tool_calls", "usage", "seconds", "tool_log")},
                "messages": ep["messages"], "config": config,
            }
            if a.judge_model:
                # Untagged answers: let the judge read the whole final message, so format
                # failures (answer_tagged=False) can be separated from wrong answers.
                judge_input = pred if tagged else ep["final_content"][-2000:]
                rec["judge"] = await judge(q["question"], q["answer"], judge_input, a.judge_model, **judge_kwargs)
        async with lock:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["n"] += 1; stats["em"] += rec["em"]; stats["contains"] += rec["contains"]
            n = stats["n"]
            if n % 10 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] EM {stats['em']/n:.1%}  contains {stats['contains']/n:.1%}", flush=True)

    await asyncio.gather(*(one(q) for q in todo))
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
