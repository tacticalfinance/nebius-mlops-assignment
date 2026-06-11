# Report: LLM inference + o11y (text-to-SQL on Qwen3-30B-A3B)

## 1. Serving configuration (Phase 1)

Hardware: 1× H100 80GB. Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, 30B total / ~3B active params).

Workload shape driving the choices: 1.5–3K-token prompts (mostly repeated DB schemas), short structured outputs (SQL / small JSON, ≤256 tokens), 2–3 *sequential* vLLM calls per agent run, SLO of p95 < 5s end-to-end at 10+ RPS (so ~20–30 LLM calls/s at the serving layer).

Baseline flags (`scripts/start_vllm.sh`) and rationale:

| Flag | Value | Why |
|---|---|---|
| `--dtype` | `bfloat16` | Native checkpoint precision; ~57 GB of weights fits in 80 GB with room for KV cache. Quantization (e.g. FP8) is deliberately kept in reserve as a Phase 6 lever if KV headroom or decode speed becomes the bottleneck. |
| `--max-model-len` | `8192` | Qwen3's native window (262K) would make vLLM budget KV for sequences we never see. Prompts top out ~3K + 256 output tokens, so 8K is generous; capping it lets far more concurrent sequences fit in the same KV pool — concurrency is what the 10 RPS target actually stresses. |
| `--gpu-memory-utilization` | `0.90` | Standard starting point: maximizes the KV-cache pool while leaving headroom for activations and CUDA graphs. |
| `--max-num-seqs` | `256` | vLLM's default, set explicitly so it's a documented tuning lever. With 3B active params per token, MoE decode is cheap per-sequence — batch capacity should not be the first ceiling we hit. |
| `--enable-prefix-caching` | on | The biggest workload-specific win: every request embeds a rendered DB schema, and each agent run makes 2–3 calls sharing that same prefix. Cached prefixes turn most prefill work into cache hits → lower TTFT and less compute per request. (Validated by the prefix-cache-hit-ratio panel in Grafana.) |
| `--disable-log-requests` | on | Keeps server logs readable during 5-minute load tests; per-request logging is pure noise at 20+ calls/s. |

What's intentionally *not* tuned yet (Phase 6 candidates): FP8 quantization / FP8 KV cache, `--max-num-batched-tokens` (prefill chunking vs TTFT), scheduler knobs, speculative decoding.

**Sanity check:** server up, manual eval-set queries return sensible SQL (e.g. `SELECT c.lat, c.lng FROM circuits c JOIN races r ON c.circuitId = r.circuitId WHERE r.name = 'Australian Grand Prix'`), and the server log already shows a 27–40% prefix-cache hit rate on repeated-schema prompts — see `screenshots/vllm_manual_query.png`.

## 2. Observability dashboard (Phase 2)

Dashboard: `infra/grafana/provisioning/dashboards/serving.json`, three rows mirroring the three questions an on-call would ask:

- **Latency** — E2E request latency (p50/p95/p99), lifecycle breakdown (queue vs prefill vs decode, p95), TTFT, inter-token latency. Answers "is it slow, and *where* in the lifecycle?"
- **Throughput** — running/waiting requests, generated tokens/s, prompt tokens/s, completed requests/s. Queue depth is the earliest saturation signal.
- **KV cache** — usage fraction (headroom / eviction risk) and prefix-cache hit ratio (validates `--enable-prefix-caching` for this schema-heavy workload).

Validated with a 3 RPS / 180s agent-level burst (540 requests, 0 errors) — every panel reacts (`screenshots/grafana_serving.png`). Readings at 3 RPS: agent E2E p50 0.81s / p95 2.90s; lifecycle p95 is decode-dominated (~1.2s) with queue ≈ 0; TTFT p95 ~70ms; ~300 gen tok/s out, ~4K prompt tok/s in; vLLM completes ~8 req/s (≈ 3 agent runs/s × 2–3 calls each, as expected); **KV cache usage peaked ~2%** and prefix-cache hit ratio climbed to ~70%. Takeaway for Phase 6: at low load the system is decode-bound with enormous KV headroom — concurrency, not memory, will be the thing to stress.

## 3. Agent (Phase 3)

LangGraph graph: `attach_schema → generate_sql → execute → verify →` (ok → END | not ok → `revise → execute → verify`, capped at `MAX_ITERATIONS`). Verify asks for strict one-line JSON `{ok, issue}`, parsed defensively; execution errors are never accepted regardless of the verifier's opinion (fail closed). Revise sees the failing SQL, its execution result, the verifier's issue, and all prior attempts so it doesn't repeat them.

Revise trigger observed in interactive testing (10 eval questions → 1 revise): on *"What is the average fastest lap time in seconds for Lewis Hamilton…"* (`formula_1`), the first generation referenced a non-existent column and errored; verify failed it with *"the column 'l.fastestLapTime' does not exist, making the result unusable"*, revise corrected the column and the second execution succeeded (2 iterations total). This is the loop working as designed: execution errors are the highest-signal failure class and are always routed to revise.

## 4. Tracing (Phase 4)

Langfuse (local, docker-compose) captures every agent run via the LangChain callback handler. `screenshots/langfuse_trace.png` shows a full revise-loop waterfall on the Lewis Hamilton question: `attach_schema → generate_sql → execute → verify (fail) → revise → execute`, each LLM span carrying its prompt, response, latency and token count (1.47s total, 3.7K prompt / 233 completion tokens — the prompt weight is the rendered schema, which is what makes prefix caching matter). Traces are tagged via the `langfuse_tags` metadata key with `phase:*` (`phase4_smoke` / `eval` / `load_test`) and `db_id:*`, so Phase 6 load-test traces are filterable from eval traces (`screenshots/langfuse_tags.png`).

## 5. Baseline eval (Phase 5)

Method: execution accuracy — agent SQL and gold SQL both run against the target DB, result sets compared after canonicalization (rows sorted, cells stringified, NULL→''). Scored per iteration from the agent's history with carry-forward, so `iter_k` = "pass rate had we stopped after k+1 attempts".

Getting to the baseline took three prompt/context-engineering iterations against the real 30B (final tuning on the real endpoint, per the assignment):

| Agent version | overall | iter_0 → iter_2 |
|---|---|---|
| v0: bare schema (names+types only) | 30% | 30% → 30% (flat) |
| v1: + value examples & BIRD column descriptions | 26.7% | flat — annotations fixed literals (`'M'`, `A15`, date formats) but inflated DDL blew the 4400-char pruning budget; dropped tables caused hallucinated columns |
| v2: budget 9000, redundant-desc filter | 36.7% | flat |
| **v3 (baseline): budget 12000 + 5 targeted SQL rules** | **50%** | **43.3% → 50%** |

The dominant failure class was the model guessing data it cannot see (literal casing `'m'`/`'M'`, labels `'carcinogenic'`/`'+'`, invented set codes, cryptic columns like `financial.A15`). Annotating the schema with sampled example values + BIRD column descriptions fixed most of it — but only once the schema budget was large enough that the pruner never drops a needed table.

**Does the loop earn its keep?** In v0–v2: no — verify/revise never flipped an outcome (revise had no signal to fix a bad literal it couldn't see either). In the final baseline: yes — iter_0→iter_1 is +6.7pp (2/30 questions recovered, mostly execution-error repairs now that revise can consult example values); iter_2 adds nothing. Remaining failures are largely gold-SQL quirks (gold returns `id` where the question asks for names, paraphrased answer strings, gold's column order differing from the question's own order) — honest eval noise rather than agent defects.

Artifacts: `results/eval_baseline.json`, `screenshots/grafana_eval_run.png`.

## 6. SLO: load test & tuning (Phase 6)

Target: **p95 end-to-end agent latency < 5s at 10+ RPS over 5 minutes.**

**Baseline (10 RPS / 300s, 3000 requests):** p50 4.4s, **p95 55.4s**, p99 61.4s — SLO missed by 11×. (`results/load_test_baseline.json`, `screenshots/grafana_before.png`.)

Iteration log (saw X → hypothesized Y → changed Z → result W):

1. **Saw**: agent p95 55s while the dashboard showed vLLM perfectly healthy — vLLM-side E2E p95 only ~5–6s, queue depth oscillating 0–30 and recovering, KV cache ≤40%, TTFT p95 ~200ms, and a sawtooth "running" curve (waves of work, then dips to 0). **Hypothesized**: the ~50s gap lives in the agent layer, not the GPU — `/answer` is a sync FastAPI endpoint, so each request holds one of anyio's ~40 threadpool threads for its whole multi-second graph run; at 10 RPS × ~5s that needs 50+ concurrent slots → requests queue inside the agent server. **Changed**: `uvicorn --workers 8` (one change; no vLLM flags touched). **Result**: p95 55.4s → **9.8s** (5.6×), p50 4.4s → 1.7s; dashboard now shows a steady ~20–25 running plateau and flat ~25 completed req/s — the sawtooth gone, full offered load reaching vLLM (`screenshots/grafana_after.png`, `results/load_test_iter1.json`).
2. **Saw**: agent p95 (9.8s) ≈ 2× vLLM per-call p95 (~5s); lifecycle breakdown decode-dominated; the agent's tail is the revise path — `MAX_ITERATIONS=3` means a worst-case run chains 6 sequential vLLM calls. **Hypothesized**: the 2nd revise is pure tail latency, because Phase 5 measured iter_1 = iter_2 = 50% (zero accuracy from the 3rd attempt). **Changed**: `MAX_ITERATIONS` 3 → 2 (worst case 6 → 4 sequential calls). **Result**: p95 9.8s → **6.3s**, p99 22.3s → 14.4s, 3000/3000 ok — pure tail win, zero quality cost by construction (`results/load_test_iter2.json`).
3. **Saw**: p95 6.3s, still 1.3s over SLO; lifecycle p95 decode-dominated with ITL p95 ~50ms / p99 ~80ms at batch ~25, while TTFT (~200ms) and KV (≤40%) have huge headroom. **Hypothesized**: with ~40K prompt tokens/s of prefill being chunked into the same engine steps as decode (default budget 8192 tokens/step), every decode step also carries large prefill chunks — inflating ITL for all running requests; trading some of our abundant TTFT headroom for smoother decode should cut the dominant component. **Changed**: `--max-num-batched-tokens 2048` (one flag; model unchanged). **Result**: p95 6.31s → 6.20s, p99 14.4s → 13.2s — essentially flat. Honest lesson: the targeted interference effect was secondary; prefill chunking was not what the SLO was waiting on (`results/load_test_iter3.json`).
4. **Saw**: latency now tracks raw decode time — ~100–150 output tokens × 30–50ms ITL ≈ 3–5s per generate call, ×2–4 sequential calls per run. **Hypothesized**: SQL output mostly copies tokens already present in the prompt (schema identifiers, question literals) — ideal for n-gram prompt-lookup speculative decoding, which drafts multi-token spans from the prompt and verifies them in one step (lossless, no quality risk). **Changed**: `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}'`. **Result**: p95 6.20s → **5.57s**; ITL p99 80→68ms / p50 ~20ms confirmed speculation accepting drafts — but the dashboard also showed a *new* symptom: "requests waiting" sustained at 15–25, as large as "running" (`results/load_test_iter4.json`).
5. **Saw**: sustained queue depth ≈ running depth (~15–25 each) with ~40K prompt tok/s offered; by Little's law that is ~0.7–1.1s of queueing per call — while lifecycle decode improved. **Hypothesized**: iteration 3's `--max-num-batched-tokens 2048`, neutral before speculation, became the binding constraint once decode sped up: prefill admission is now the bottleneck. **Changed**: reverted to the default prefill budget (dropped the flag), keeping speculation. **Result**: _pending._

Quality after tuning: _TODO — `results/eval_after_tuning.json` vs baseline._

Verdict: _TODO — SLO hit or missed, gap quantified._

## 7. Agent value

_TODO — one paragraph citing the per-iteration pass rates._

## 8. What I'd do with more time

_TODO._
