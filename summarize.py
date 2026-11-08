"""Summarize one or more run files: accuracy, behaviour and cost, overall and per group.

Usage:
  python summarize.py runs/qwen3.8-27b.jsonl runs/gemma4-31b.jsonl \
      [--coverage ../wiki_bm25/infoseek_hard_6k.coverage.jsonl] [--by num_nodes]
With --coverage, rows are also grouped by retrieval bucket (answer page rank for the raw question).
"""
import argparse
import collections
import json
import statistics


def bucket(rank):
    if rank is None:
        return "multi-hop (>50)"
    return "direct (1-5)" if rank <= 5 else "near (6-50)"


def summarize(rows, label):
    n = len(rows)
    if not n:
        return
    acc = lambda key: sum(bool(r.get(key)) for r in rows) / n
    judged = [r for r in rows if r.get("judge") is not None]
    tc = [r["num_tool_calls"] for r in rows]
    toks = [r["usage"]["completion_tokens"] for r in rows]
    ctx = [r["usage"]["max_prompt_tokens"] for r in rows]
    status = collections.Counter(r["status"] for r in rows)
    calls = collections.Counter(t["name"] for r in rows for t in r["tool_log"])
    line = (f"{label:<28} n={n:<5} EM {acc('em'):6.1%}  contains {acc('contains'):6.1%}"
            + (f"  judge {sum(r['judge'] for r in judged)/len(judged):6.1%}" if judged else "")
            + f"  tagged {acc('answer_tagged'):6.1%}"
            + f" | tool calls mean {statistics.mean(tc):4.1f} (search {calls['search']/n:.1f}, open {calls['open_page']/n:.1f})"
            + f" | out tokens mean {statistics.mean(toks):6.0f}  max ctx {max(ctx):6d}"
            + f" | " + ", ".join(f"{k} {v}" for k, v in status.most_common()))
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--coverage")
    ap.add_argument("--by", default="num_nodes")
    a = ap.parse_args()
    ranks = {}
    if a.coverage:
        for l in open(a.coverage):
            d = json.loads(l)
            if "answer_rank" in d:
                ranks[d["id"]] = d["answer_rank"]
    for path in a.runs:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        print(f"\n== {path} ({rows[0]['model'] if rows else ''})")
        summarize(rows, "all")
        groups = collections.defaultdict(list)
        for r in rows:
            groups[f"{a.by}={r.get(a.by)}"].append(r)
        for k in sorted(groups):
            summarize(groups[k], k)
        if ranks:
            groups = collections.defaultdict(list)
            for r in rows:
                if r["id"] in ranks:
                    groups[bucket(ranks[r["id"]])].append(r)
            for k in sorted(groups):
                summarize(groups[k], k)


if __name__ == "__main__":
    main()
