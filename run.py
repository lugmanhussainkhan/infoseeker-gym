"""Run the search agent over a question file and write one JSON line per question.

Example:
  python run.py \
    --questions data/test.jsonl --limit 500 \
    --llm openai/Qwen/Qwen3.8-27B \
    --llm-args '{"api_base": "http://gpu-box:8001/v1", "api_key": "EMPTY",
                 "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}' \
    --judge-llm anthropic/claude-sonnet-5-5 \
    --index-url http://index-box:8000 \
    --out runs/qwen3.8-27b.jsonl --concurrency 32

--llm-args / --judge-llm-args are JSON kwargs for litellm.acompletion, merged over the defaults.
Re-running with the same --out resumes.
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
from harness.scoring import judge

litellm.drop_params = True
litellm.suppress_debug_info = True

LLM_DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192, "timeout": 600}
JUDGE_DEFAULTS = {"temperature": 0, "timeout": 600}


def json_object(text: str) -> dict:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}")
    if not isinstance(d, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return d


def redact(args: dict) -> dict:
    return {k: v for k, v in args.items() if not k.lower().endswith(("key", "secret", "password", "token"))}


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
    ap.add_argument("--llm", required=True, help="litellm model string for the agent, e.g. openai/<served-name>")
    ap.add_argument("--llm-args", type=json_object, default={},
                    help=f"JSON kwargs for the agent's litellm calls, merged over {LLM_DEFAULTS}")
    ap.add_argument("--judge-llm", required=True, help="litellm model string for the judge")
    ap.add_argument("--judge-llm-args", type=json_object, default={},
                    help=f"JSON kwargs for the judge's litellm calls, merged over {JUDGE_DEFAULTS}")
    ap.add_argument("--index-url", default=os.environ.get("INDEX_URL", "http://localhost:8000"))
    ap.add_argument("--max-tool-calls", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--search-k", type=int, default=5)
    ap.add_argument("--page-words", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    a = ap.parse_args()
    llm_kwargs = {**LLM_DEFAULTS, **a.llm_args}
    judge_kwargs = {**JUDGE_DEFAULTS, **a.judge_llm_args}
    config = {**vars(a), "llm_args": redact(llm_kwargs), "judge_llm_args": redact(judge_kwargs)}

    qs = load_questions(a.questions)
    if a.shuffle_seed is not None:
        random.Random(a.shuffle_seed).shuffle(qs)
    if a.limit:
        qs = qs[: a.limit]
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            prev = [json.loads(l) for l in fh if l.strip()]
        kept = [r for r in prev if r["status"] != "model_error"]
        if len(kept) < len(prev):
            print(f"Retrying {len(prev) - len(kept)} episodes that ended in model_error")
        rejudge = [r for r in kept if r.get("judge", False) is None]
        if rejudge:
            print(f"Re-judging {len(rejudge)} episodes whose judge call failed")

            async def rejudge_one(r):
                async with sem:
                    return await judge(r["question"], r["golds"], r.get("response", ""), a.judge_llm, **judge_kwargs)

            results = await asyncio.gather(*(rejudge_one(r) for r in rejudge))
            for r, v in zip(rejudge, results):
                r.update(v, judge_model=a.judge_llm)
        if len(kept) < len(prev) or rejudge:
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

    print(f"{len(qs)} questions, {len(done)} already done, running {len(todo)}")
    stats = {"n": 0, "correct": 0, "judge_failed": 0}

    async def one(q):
        async with sem:
            ep = await run_episode(q["question"], env, a.llm, llm_kwargs, a.max_tool_calls, a.max_turns)
            response = ep["final_content"] if ep["status"] == "answered" else ""
            verdict = await judge(q["question"], q["golds"], response, a.judge_llm, **judge_kwargs)
            rec = {
                "id": q["id"], "question": q["question"], "answer": q["answer"], "golds": q["golds"],
                **q["meta"], "model": a.llm, "judge_model": a.judge_llm, "response": response, **verdict,
                **{k: ep[k] for k in ("status", "error", "num_turns", "num_tool_calls", "usage", "seconds", "tool_log")},
                "messages": ep["messages"], "config": config,
            }
        async with lock:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["n"] += 1
            stats["correct"] += rec["judge"] is True
            stats["judge_failed"] += rec["judge"] is None
            n = stats["n"]
            if n % 10 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] judge accuracy {stats['correct']/n:.1%}"
                      f"  (judge failed on {stats['judge_failed']})", flush=True)

    await asyncio.gather(*(one(q) for q in todo))
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
