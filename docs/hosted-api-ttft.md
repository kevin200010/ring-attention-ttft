# Lowering TTFT on hosted APIs (Claude, OpenAI, …)

On a hosted API you can't choose Ring Attention or any other prefill strategy:
the provider decides how prompts are spread across its hardware. TTFT there
comes from **prefill (reading new prompt tokens)**, **thinking time before the
first visible token**, and **network**. These levers address them, roughly in
order of impact for long-context RAG.

## 1. Prompt caching

A cached prefix is not prefilled again, which is the biggest TTFT win when the
same documents or system prompt repeat.

- **Claude:** mark the stable part with `cache_control` (optionally `"ttl": "1h"`).
  Check `usage.cache_read_input_tokens > 0` to confirm hits.
- **OpenAI:** automatic for long prompts. Keep the beginning of the prompt identical.
- **Prompt order:** instructions → documents → question. Anything that changes
  (timestamps, user IDs) goes after the cached part, because one changed byte
  before it is a cache miss.
- **Pre-warm:** send a throwaway request with the documents when a user opens a
  workspace, so the real question hits a warm cache.

```python
response = client.messages.create(
    model="claude-opus-5-5", max_tokens=1024,
    system=[{"type": "text", "text": DOCS, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    messages=[{"role": "user", "content": question}],
)
print(response.usage.cache_read_input_tokens)
```

## 2. Send fewer tokens

Prefill time grows faster than linearly with length (this repo's single-Mac
throughput: 2,600 → 1,150 tokens/s from 4k to 32k).

- Rerank and send the top 5–10 chunks, not 50
- Remove duplicate and near-duplicate chunks
- Strip boilerplate and markup
- Drop tool definitions the request doesn't need

## 3. Less thinking before the first word

Reasoning models may think for seconds before the first visible token. For
lookup-style RAG, use low effort (`output_config={"effort": "low"}` on Claude,
`reasoning_effort="low"` on OpenAI) or a non-reasoning model.

## 4. Smaller models where good enough

Smaller models prefill faster. "Find the answer in these documents" often works
fine on a Haiku/Sonnet- or mini-class model.

## 5. Connection and network

- **Stream** the response, and measure TTFT as the time to the first text delta
- **Reuse one client** so connections stay open
- **Deploy near the provider's region**

## 6. Change the request flow

- **Retrieve while the user types:** start embedding and retrieval early
- **Two-step answers:** a fast model replies immediately while a larger one works
- **Split, query in parallel, combine:** split documents across parallel requests
  and merge the answers. It's an application-level ring, with one catch: the model
  can't connect facts across groups.

## Measuring TTFT through an API

```python
import time, anthropic
client = anthropic.Anthropic()
t0 = time.perf_counter()
with client.messages.stream(model="claude-opus-5-5", max_tokens=200,
                            messages=[{"role": "user", "content": rag_prompt}]) as stream:
    for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(f"TTFT {time.perf_counter() - t0:.2f}s")
            break
```

Repeat each measurement several times and report the median: API latency varies
with load.
