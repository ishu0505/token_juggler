# token_daddy
unified llm provider interface and token tracking

## LLM gate (rate-limited, async)

Every call goes through `llm_gate`, which enforces per provider/model TPM, RPM,
concurrency, workspace output TPM and a per-job token budget in Redis:

```python
from token_daddy.llm import client_for
from token_daddy.llm.gate import estimate_tokens, llm_gate

client, model = client_for("gemini")  # or "openai"
async with llm_gate("gemini", model, job_id="job-1",
                    estimated_tokens=estimate_tokens(prompt)) as slot:
    response = await client.call_structured(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_schema=schema,
    )
    slot.record_actual(response.total_tokens, response.output_tokens)
```

## Unified clients (sync)

```python
from token_daddy.llm_clients import Attachment, get_client

client = get_client("gemini")  # or "openai" / "anthropic"
reply = client.generate("What is this?", attachments=[Attachment.from_path("a.pdf")])
print(reply.text, reply.usage.total_tokens, reply.estimated_cost_usd)
```
