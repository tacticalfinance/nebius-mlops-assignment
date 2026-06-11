# Report: LLM inference + o11y (text-to-SQL on Qwen3-30B-A3B)

## 1. Serving configuration (Phase 1)

1× H100 80GB, `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, 30B total / ~3B active). Workload: 1.5–3K-token prompts (repeated DB schemas), short structured outputs (≤256 tokens), 2–4 *sequential* vLLM calls per agent run, SLO p95 < 5s at 10+ RPS (~20–30 LLM calls/s).

| Flag | Value | Why |
|---|---|---|
| `--dtype` | `bfloat16` | Native precision; ~57 GB fits in 80 GB with KV room. FP8 deliberately held in reserve as a Phase 6 lever. |
| `--max-model-len` | `8192` | Qwen3's 262K window would budget KV for sequences we never see; prompts top out ~3K + 256 out. Capping it buys concurrency, which is what 10 RPS stresses. |
| `--gpu-memory-utilization` | `0.90` | Maximize KV pool, leave headroom for activations/CUDA graphs. |
| `--max-num-seqs` | `256` | Default, made explicit as a tuning lever; 3B-active MoE decode is cheap per sequence, so batch capacity shouldn't be the first ceiling. |
| `--enable-prefix-caching` | on | The workload-specific win: every request embeds a rendered schema and each run's 2–4 calls share that prefix. Validated by the hit-ratio panel (~85% under load). |
| `--disable-log-requests` | on | Readable logs at 20+ calls/s. |
| `--speculative-config` | ngram, 5 tokens | *Added in Phase 6 iteration 4* — SQL output copies schema/question tokens from the prompt; prompt-lookup drafts verify losslessly. See §6. |

Sanity check: manual eval-set queries return correct SQL; server log already showed 27–40% prefix-cache hits (`screenshots/vllm_manual_query.png`).

## 2. Observability dashboard (Phase 2)

`infra/grafana/provisioning/dashboards/serving.json` — three rows for the three on-call questions: **Latency** (E2E p50/p95/p99, queue/prefill/decode lifecycle breakdown, TTFT, ITL — "is it slow and *where*?"), **Throughput** (running/waiting, gen & prompt tokens/s, completed req/s — queue depth is the earliest saturation signal), **KV cache** (usage fraction, prefix-hit ratio).

Validated with a 3 RPS / 180s burst, all panels reacting (`screenshots/grafana_serving.png`): decode-dominated lifecycle, queue ≈ 0, TTFT p95 ~70ms, KV peak ~2%, prefix hits ~70%. Read-across for Phase 6: decode-bound with huge KV headroom — concurrency, not memory, is what will break.

## 3. Agent (Phase 3)

LangGraph: `attach_schema → generate_sql → execute → verify` → ok ? END : `revise → execute → verify`, capped at `MAX_ITERATIONS`. Verify returns strict one-line JSON `{ok, issue}`, parsed defensively; execution errors are never accepted regardless of the verifier (fail closed). Revise sees the failing SQL, execution output, verifier issue, and all prior attempts.

Observed revise trigger: *"average fastest lap time for Lewis Hamilton"* — first SQL referenced a non-existent column, verify failed it (*"column 'l.fastestLapTime' does not exist"*), revise fixed it, second execution succeeded.

## 4. Tracing (Phase 4)

Langfuse (local) captures every run via the LangChain callback. `screenshots/langfuse_trace.png`: full revise-loop waterfall with per-span prompt/response/latency/tokens (1.47s, 3.7K prompt / 233 completion — prompt weight is the schema, which is why prefix caching matters). Traces tagged via `langfuse_tags` with `phase:*` and `db_id:*`, so load-test traffic is filterable from eval runs (`screenshots/langfuse_tags.png`).

## 5. Baseline eval (Phase 5)

Execution accuracy: agent SQL and gold SQL run against the target DB, row sets compared canonicalized (sorted, stringified, NULL→''). Per-iteration scoring with carry-forward: `iter_k` = pass rate had we stopped after k+1 attempts. Reaching the baseline took three tuning iterations against the real 30B:

| Agent version | overall | per-iteration |
|---|---|---|
| v0: bare schema (names+types) | 30% | flat |
| v1: + sampled value examples & BIRD column descriptions | 26.7% | flat — fixed literals but bigger DDL blew the schema-pruning budget; dropped tables → hallucinated columns |
| v2: bigger budget, redundant-desc filter | 36.7% | flat |
| **v3 (baseline): 12K-char budget + 5 targeted SQL rules** | **50%** | **43.3% → 50%** |

The dominant failure class was the model guessing data it can't see (`'m'` vs `'M'`, `'carcinogenic'` vs `'+'`, invented set codes, cryptic `financial.A15`). Schema annotations fixed it — but only once the budget guaranteed no needed table is pruned.

**Does the loop earn its keep?** v0–v2: no — flat curve; revise had no signal to fix unseen literals. v3: yes — +6.7pp at iter_1 (2/30 recovered, mostly execution-error repairs), iter_2 adds 0pp. Remaining failures are largely gold-SQL quirks (gold returns `id` for "list cards", paraphrased answer strings, gold column order differing from the question's own order). Artifacts: `results/eval_baseline.json`, `screenshots/grafana_eval_run.png`.

## 6. SLO: load test & tuning (Phase 6)

Target: **p95 < 5s at 10+ RPS over 5 min.** Baseline (10 RPS / 300s): p50 4.4s, **p95 55.4s**, p99 61.4s, 2997/3000 ok (3 client-side connection errors) — missed 11× (`results/load_test_baseline.json`, `screenshots/grafana_before.png`).

Iteration log — every entry: saw → hypothesized → changed (one thing) → result:

1. **Saw** agent p95 55s while vLLM looked healthy (vLLM E2E p95 ~5–6s, queue recovering, KV ≤40%, sawtooth "running" curve). **Hypothesized** the gap is agent-side: sync `/answer` holds one of ~40 threadpool threads per request; 10 RPS × ~5s needs 50+. **Changed** `uvicorn --workers 8`. **Result** p95 → **9.8s** (5.6×); sawtooth gone, steady ~25 req/s into vLLM.
2. **Saw** p95 ≈ 2× vLLM per-call latency; tail = revise path (`MAX_ITERATIONS=3` ⇒ worst case 6 sequential calls). **Hypothesized** the 2nd revise is pure tail: Phase 5 measured iter_1 = iter_2 = 50%. **Changed** cap 3 → 2. **Result** p95 → **6.3s**, p99 22→14s; zero quality cost by construction.
3. **Saw** decode-dominated lifecycle, ITL p95 ~50ms, TTFT/KV with huge headroom; ~40K prompt tok/s of prefill chunked into decode steps (8192/step budget). **Hypothesized** prefill chunks inflate every decode step; trade TTFT headroom for ITL. **Changed** `--max-num-batched-tokens 2048`. **Result** p95 6.31→6.20s — **flat; hypothesis wrong** (prefill interference was secondary).
4. **Saw** latency tracking raw decode (~100–150 tokens × 30–50ms × 2–4 sequential calls). **Hypothesized** SQL output copies prompt tokens → ideal for lossless n-gram speculative decoding. **Changed** `--speculative-config` (ngram, 5 draft tokens). **Result** p95 → **5.57s**, ITL p99 80→68ms — but a *new* symptom appeared: "waiting" sustained at 15–25 ≈ "running".
5. **Saw** sustained queue ≈ running; Little's law ⇒ ~0.7–1.1s queueing per call. **Hypothesized** iteration 3's 2048 budget, neutral before speculation, became binding once decode sped up: admission is now the bottleneck. **Changed** reverted to default budget, kept speculation. **Result** p95 → **4.79s** — SLO met (p50 1.49s, p99 12.1s, 3000/3000 ok at 10 RPS / 300s).

Before/after: `screenshots/grafana_before.png` (sawtooth, 55s p95) vs `grafana_after.png` (queue ≈ 0, steady ~23 req/s, ITL p95 ~48ms).

**Quality after tuning** (`results/eval_after_tuning.json`): 50%, per-iteration 43.3% → 50% — identical to baseline; expected (cap removed a measured-useless attempt; speculation is lossless) but measured, not assumed.

**Verdict: SLO met** — p95 4.79s vs 55.4s baseline (11.6×), quality exactly preserved. Caveats kept honest: p99 is 12.1s (the revise path can't fit 5s; a p99 SLO would need an architecture change, see §8), and iteration 3 was a wrong hypothesis whose flag later *became* the bottleneck — only the queue-depth panel exposed that regime shift.

## 7. Agent value

The loop earns its keep, but only once it had data to act on. Final: iter_0 43.3% → iter_1 50% (+6.7pp, 2/30 — execution-error repairs and zero-row fixes where schema example values supply the correct literal); iter_2 adds 0pp, which is why `MAX_ITERATIONS=2` — that measurement converted directly into a 3.5s p95 cut in Phase 6. In the pre-annotation agent the curve was flat: a revise loop is only as good as its signal. Costs: one extra sequential verify call (~1s) per request and the 12s p99 tail; defensible for a PoC, but the +6.7pp should be re-validated on a larger eval set (30 questions ⇒ wide confidence intervals).

## 8. What I'd do with more time

- **Parallel self-consistency**: 3 samples at temp>0 *in parallel*, execute all, majority-vote on canonicalized rows — attacks plausible-but-wrong SQL (which verify can't catch) without sequential latency; the KV headroom for it is visibly idle.
- **Merge verify into generate** (`{sql, self_check}` in one response), keeping a separate verify only for errors/0-rows — removes ~1s from every request and attacks the p99 tail.
- **Eval statistical power**: 30 questions ⇒ ±18pp at 50%; scale to 200+ BIRD dev questions and stratify per DB before trusting prompt changes.
- **Schema linking by embedding retrieval** instead of token overlap, measured by needed-table recall — v1's regression showed table-dropping is the most destructive failure mode.
- **FP8 A/B**: official FP8 checkpoint vs bf16 — one load test + one eval to quantify ITL gain vs quality cost; it was the next lever if iteration 5 had missed.
- **Mine the Langfuse tags**: per-node latency aggregation over `phase:load_test` traces to attribute the 12.1s p99 (which node, which iteration) and confirm the ~51s max outliers are cold prefix-cache misses on first-seen DBs.
