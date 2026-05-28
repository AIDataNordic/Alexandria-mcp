"""
eval_alexandria.py — Smoke test for Alexandria MCP server

Tester tre dimensjoner:
  1. Relevans      — kjente spørsmål skal treffe riktig forfatter i topp-N
  2. Filtre        — author/language/date_to-filtre skal virke korrekt
  3. Latens        — realistisk responstid

Kjøring:
    venv/bin/python3 eval_alexandria.py
    venv/bin/python3 eval_alexandria.py --url http://localhost:8005/mcp
    venv/bin/python3 eval_alexandria.py --verbose
"""

import argparse
import asyncio
import json
import sys
import time

from fastmcp import Client

ALEXANDRIA_URL = "http://localhost:8005/mcp"

# (query, expected_author_substring, beskrivelse)
# expected_author er case-insensitiv substring-match mot "creator"-feltet i svaret
RELEVANCE_CASES = [
    ("categorical imperative duty ethics",           "kant",       "Kant — kategorisk imperativ"),
    ("will to power eternal recurrence Zarathustra", "nietzsche",  "Nietzsche — vilje til makt"),
    ("theory of forms the Good beautiful",           "plat",       "Platon — formteorien"),
    ("eudaimonia flourishing virtue ethics",         "aristotle",  "Aristoteles — eudaimonia"),
    ("cogito ergo sum I think therefore I am",       "descartes",  "Descartes — cogito"),
    ("critique of pure reason synthetic a priori",   "kant",       "Kant — ren fornuft"),
    ("dialectics spirit history absolute idea",      "hegel",      "Hegel — dialektikk"),
    ("greatest happiness principle utility",         "mill",       "Mill — utilitarisme"),
    ("tabula rasa empiricism ideas sensation",       "lock",       "Locke — tabula rasa"),
    ("stoic virtue sage indifferent externals",      "epict",      "Epiktet — stoisk visdom"),
    ("divine grace predestination original sin",     "augustin",   "Augustin — nåde/synd"),
    ("social contract state of nature sovereign",    "hobbes",     "Hobbes — samfunnskontrakten"),
    ("the unexamined life is not worth living",      "plat",       "Platon/Sokrates — det uutforskede liv"),
    ("slave morality ressentiment noble values",     "nietzsche",  "Nietzsche — slavemoralen"),
    ("pliktetikk kategorisk imperativ",              "kant",       "Kant — norsk query (multilingual)"),
]

# (beskrivelse, tool-kall-kwargs, assertion-funksjon som tar results og returnerer (ok, melding))
FILTER_CASES = [
    (
        "author-filter: Aristotle — alle resultater skal ha 'aristotle' i creator",
        {"query": "virtue soul good life happiness", "author": "Aristotle", "limit": 5},
        lambda r: (
            all("aristotle" in _creator_str(x) for x in r),
            f"{sum(1 for x in r if 'aristotle' in _creator_str(x))}/{len(r)} har Aristotle i creator"
        ),
    ),
    (
        "language-filter: ger — alle resultater skal ha 'ger' i language",
        {"query": "kategorischer Imperativ Pflicht", "language": "ger", "limit": 5},
        lambda r: (
            all("ger" in str(x.get("language") or "").lower() for x in r),
            f"{sum(1 for x in r if 'ger' in str(x.get('language') or '').lower())}/{len(r)} har language=ger"
        ),
    ),
    (
        "date_to-filter: ≤400 — kun antikke tekster",
        {"query": "virtue soul good", "date_to": 400, "limit": 5},
        lambda r: (
            all(int(str(x.get("date") or "0")[:4] or 0) <= 400 for x in r if x.get("date")),
            ", ".join(f"{_creator_str(x)} ({x.get('date','?')})" for x in r[:3])
        ),
    ),
    (
        "date_from-filter: ≥1800 — kun moderne tekster",
        {"query": "consciousness freedom existence", "date_from": 1800, "limit": 5},
        lambda r: (
            all(int(str(x.get("date") or "0")[:4] or 0) >= 1800 for x in r if x.get("date")),
            ", ".join(f"{_creator_str(x)} ({x.get('date','?')})" for x in r[:3])
        ),
    ),
    (
        "kombinert: author=Kant + language=ger",
        {"query": "Pflicht Vernunft", "author": "Kant", "language": "ger", "limit": 5},
        lambda r: (
            all("kant" in _creator_str(x) for x in r),
            f"{len(r)} resultater, creators: {list({_creator_str(x) for x in r})}"
        ),
    ),
]


def _parse_results(resp) -> list[dict]:
    """Extract list[dict] from a fastmcp CallToolResult."""
    try:
        text = resp.content[0].text if resp and resp.content else None
        return json.loads(text) if text else []
    except Exception:
        return []


def _ping_ok(resp) -> bool:
    try:
        return bool(resp and resp.content and resp.content[0].text)
    except Exception:
        return False


def _creator_str(r: dict) -> str:
    c = r.get("creator") or ""
    if isinstance(c, list):
        return " ".join(c).lower()
    return str(c).lower()


def check_author_hit(results: list[dict], expected_author: str, at_k: int) -> bool:
    for r in results[:at_k]:
        if expected_author.lower() in _creator_str(r):
            return True
    return False


async def run_eval(url: str, verbose: bool):
    print(f"\nAlexandria smoke test — {url}")
    print("=" * 60)

    async with Client(url) as client:

        # --- ping ---
        print("\n[ping]")
        t0 = time.time()
        resp = await client.call_tool("ping", {})
        ping_ms = round((time.time() - t0) * 1000)
        ping_ok = _ping_ok(resp)
        print(f"  {'OK' if ping_ok else 'FEIL'} ({ping_ms} ms)")

        # --- relevanstest ---
        print(f"\n[relevans] {len(RELEVANCE_CASES)} spørsmål, limit=5")
        print(f"  {'Beskrivelse':<45} {'H@1':>4} {'H@3':>4} {'H@5':>4} {'ms':>6}  top-creator")
        print("  " + "-" * 80)

        hits1 = hits3 = hits5 = 0
        latencies = []
        relevance_details = []

        for query, expected, desc in RELEVANCE_CASES:
            t0 = time.time()
            resp = await client.call_tool("search_texts", {"query": query, "limit": 5})
            elapsed_ms = round((time.time() - t0) * 1000)
            latencies.append(elapsed_ms)

            results = _parse_results(resp)

            h1 = check_author_hit(results, expected, 1)
            h3 = check_author_hit(results, expected, 3)
            h5 = check_author_hit(results, expected, 5)
            hits1 += h1
            hits3 += h3
            hits5 += h5

            top_creator = _creator_str(results[0])[:25] if results else "–"
            mark1 = "✓" if h1 else ("~" if h3 else ("≈" if h5 else "✗"))

            print(f"  {mark1} {desc:<43} {str(h1)[0]:>4} {str(h3)[0]:>4} {str(h5)[0]:>4} {elapsed_ms:>5}ms  {top_creator}")

            if verbose and results:
                for i, r in enumerate(results[:3]):
                    score = r.get("rerank_score", "?")
                    title = (r.get("title") or "")[:50]
                    creator = _creator_str(r)[:25]
                    print(f"       [{i+1}] score={score:.3f}  {creator} — {title}")

            relevance_details.append({
                "query": query,
                "expected": expected,
                "desc": desc,
                "hit@1": h1, "hit@3": h3, "hit@5": h5,
                "elapsed_ms": elapsed_ms,
                "results": results if verbose else [{"creator": r.get("creator"), "title": r.get("title"), "rerank_score": r.get("rerank_score")} for r in results],
            })

        n = len(RELEVANCE_CASES)
        avg_ms = round(sum(latencies) / len(latencies))
        print(f"\n  Totalt: hits@1={hits1}/{n}  hits@3={hits3}/{n}  hits@5={hits5}/{n}  snitt {avg_ms}ms/søk")

        # --- filtertest ---
        print(f"\n[filtre] {len(FILTER_CASES)} testtilfeller")
        filter_results = []
        filter_pass = 0

        for desc, kwargs, assert_fn in FILTER_CASES:
            t0 = time.time()
            resp = await client.call_tool("search_texts", kwargs)
            elapsed_ms = round((time.time() - t0) * 1000)

            results = _parse_results(resp)
            if not results:
                ok, detail = False, "ingen resultater"
            else:
                ok, detail = assert_fn(results)

            mark = "✓" if ok else "✗"
            filter_pass += ok
            print(f"  {mark} {desc}")
            print(f"       {detail}  ({elapsed_ms}ms)")

            filter_results.append({"desc": desc, "kwargs": kwargs, "ok": ok, "detail": detail, "elapsed_ms": elapsed_ms})

        print(f"\n  Totalt: {filter_pass}/{len(FILTER_CASES)} filtertester bestått")

    # --- oppsummering ---
    print("\n" + "=" * 60)
    print("OPPSUMMERING")
    print(f"  ping:       {'OK' if ping_ok else 'FEIL'}")
    print(f"  hits@1:     {hits1}/{n}  ({round(hits1/n*100)}%)")
    print(f"  hits@3:     {hits3}/{n}  ({round(hits3/n*100)}%)")
    print(f"  hits@5:     {hits5}/{n}  ({round(hits5/n*100)}%)")
    print(f"  filtre:     {filter_pass}/{len(FILTER_CASES)} OK")
    print(f"  latens:     snitt {avg_ms}ms, max {max(latencies)}ms")
    print()

    return {
        "ping_ok": ping_ok,
        "relevance": relevance_details,
        "filters": filter_results,
        "summary": {
            "hits_at_1": hits1, "hits_at_3": hits3, "hits_at_5": hits5,
            "n": n,
            "filter_pass": filter_pass, "filter_total": len(FILTER_CASES),
            "avg_latency_ms": avg_ms, "max_latency_ms": max(latencies),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Alexandria MCP smoke test")
    parser.add_argument("--url", default=ALEXANDRIA_URL)
    parser.add_argument("--verbose", action="store_true", help="Vis topp-3 resultater per spørsmål")
    parser.add_argument("--out", default="eval_alexandria_results.json", help="Lagre JSON-resultater til fil")
    args = parser.parse_args()

    results = asyncio.run(run_eval(args.url, args.verbose))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Resultater lagret til {args.out}")


if __name__ == "__main__":
    main()
