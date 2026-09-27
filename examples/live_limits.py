"""Live demo against tokenjuggler.yaml's tiny test limits (rpm=2 per deployment).

    REDIS_URL=redis://localhost:6379/15 uv run python examples/live_limits.py

Watch calls fill a deployment, spill to the next one, fall back to another
model, and finally be refused - without a single provider 429.
"""

import asyncio
import os
import time

from dotenv import load_dotenv

from tokenjuggler import QuotaExceeded, TokenJuggler

load_dotenv(os.path.join(os.getcwd(), ".env"))
PROMPT = "Reply with one word: ok"


async def unified(tj, model):
    started = time.perf_counter()
    try:
        r = await tj.generate(model, PROMPT)
        route = " -> ".join(f"{a.deployment}({a.outcome})" for a in r.attempts)
        print(f"  generate({model!r:20}) served by {r.deployment:30} "
              f"{(time.perf_counter() - started):4.1f}s  {r.usage.input_tokens}in/"
              f"{r.usage.output_tokens}out  route: {route}")
    except QuotaExceeded as exc:
        print(f"  generate({model!r:20}) REFUSED before calling any provider: {exc}")


async def main():
    tj = TokenJuggler.from_config("tokenjuggler.yaml", project="live-test")
    # Start from empty buckets.
    keys = await tj.backend._redis.keys("{tj-test}*")
    if keys:
        await tj.backend._redis.delete(*keys)

    print("\n1) native OpenAI SDK -> gpt-5.4 (Responses API via Databricks)")
    oai = tj.openai("gpt-5.4")
    resp = await oai.responses.create(model="gpt-5.4", input=PROMPT, max_output_tokens=64)
    print(f"  responses.create -> {resp.output_text!r} ({resp.usage.input_tokens}in/"
          f"{resp.usage.output_tokens}out)")

    print("\n2) unified gpt-5.4: 1 more fits on Databricks, then fallback model gemini-3.7-flash")
    for _ in range(2):
        await unified(tj, "gpt-5.4")

    print("\n3) native google-genai SDK -> gemini-3.7-flash")
    gem = tj.genai("gemini-3.7-flash")
    out = await gem.aio.models.generate_content(model="gemini-3.7-flash", contents=PROMPT)
    print(f"  generate_content -> {out.text!r}")

    print("\n4) unified gemini-3.7-flash until every route is full")
    for _ in range(3):
        await unified(tj, "gemini-3.7-flash")

    print("\n5) claude-sonnet-5 (single Databricks route): 2 fit, 3rd refused")
    for _ in range(3):
        await unified(tj, "claude-sonnet-5")

    print("\n6) usage recorded for project 'live-test'")
    for row in sorted(await tj.usage(), key=lambda r: r["deployment"]):
        print(f"  {row['deployment']:30} requests={row['requests']} errors={row['errors']} "
              f"in={row['input_tokens']} out={row['output_tokens']} cost=${row['cost_usd']:.4f}")
    await tj.aclose()


asyncio.run(main())
