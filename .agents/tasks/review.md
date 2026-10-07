# Multimodal ONNX inference engine for text + image toxicity scoring

The change introduces `src/models/inference.py`, `model_utils.py`, `warmup.py`, and `scripts/export_onnx_stubs.py` — a self-contained inference layer that tokenises text and fetches/decodes images concurrently via `asyncio.gather`, routes each modality through a dedicated ONNX session, and returns `ModerationResult` objects with label, confidence, per-modality scores, L2-normalised embeddings, and wall-clock latency. ONNX sessions are loaded once at startup and cached; CPU-bound session calls are offloaded to a thread-pool executor. The stub exporter produces valid ONNX graphs that mirror the exact input/output shapes the engine expects, enabling CI to run without real model weights.

Watch for: `asyncio.get_event_loop()` is deprecated in Python 3.10+ and will raise in some contexts (confirmed); the per-item latency timestamp is captured before the item is appended to the result list, so all items in a batch record different latency values even though the batch ran as a unit (confirmed); and there are zero automated tests — the smoke-test evidence referenced in the task spec was not found in the repository (confirmed).

**Verdict**: NEEDS_CHANGES

---

## High-level view

The concurrency model is structurally correct: `asyncio.gather` fans out text and image work, and the two ONNX `.run()` calls are wrapped in `run_in_executor` so they don't block the event loop. The one correctness problem is that both helpers call `asyncio.get_event_loop()` rather than `asyncio.get_running_loop()` — the latter is the correct API inside a coroutine and the former is deprecated since Python 3.10 with a `DeprecationWarning` that becomes an error in 3.12.

The per-item `latency_ms` field captures a snapshot mid-loop rather than the batch wall time. Every item in a batch records a different value, and items near the end of the list will record something close to true batch latency while items near the start will record nearly zero. Callers observing `results[i].latency_ms` for SLO monitoring will get misleading numbers for all but the last item.

The zero vector substituted for a failed image fetch is indistinguishable in the `ModerationResult` from a legitimately all-dark image; there is no `fetch_failed` flag on the result to let callers know the image path was a fallback.

There are no tests anywhere in the repository. The warmup utility makes real outbound HTTP requests to `picsum.photos` during startup, which means CI and local dev environments without internet access will hang on warmup.

---

<details>
<summary>Issues (5)</summary>

1. **`get_event_loop()` deprecation** — Both `_run_text_inference` and `_run_image_inference` call `asyncio.get_event_loop()`. Inside a running coroutine this should be `asyncio.get_running_loop()`. `get_event_loop()` emits `DeprecationWarning` in Python 3.10+ and raises `RuntimeError` in 3.12 when no current event loop is set. Replace both calls with `asyncio.get_running_loop()`.

2. **Per-item latency timestamp is wrong** — `latency_ms` is stamped inside the per-item `for` loop against `start`, so `results[0].latency_ms` ≈ 0 and only the last item approaches true batch latency. Capture `total_latency_ms` after the loop finishes and assign it to every `ModerationResult`, or move it to a batch-level field.

3. **No `fetch_failed` flag on image fallback** — When `_safe_fetch` returns `None`, the result is silently scored with a zero-pixel image. Downstream consumers cannot distinguish a genuine all-dark image from a fetch failure. Add a boolean `image_fetch_failed: bool` field to `ModerationResult` and set it when the zero-vector fallback is used.

4. **Warmup makes unconditional outbound HTTP requests** — `warmup.py` hits `https://picsum.photos/224` on every startup. This will block or fail in air-gapped CI environments. Accept an optional `image_url` parameter (defaulting to the picsum URL) or a flag to skip image warmup, and document the network dependency.

5. **No tests** — The repository contains zero test files. At minimum: a unit test for `_run_text_inference` / `_run_image_inference` using the ONNX stubs from `export_onnx_stubs.py`; a test that `_safe_fetch` returning `None` produces a `ModerationResult` with all-zero image embedding and (once added) `image_fetch_failed=True`; and a latency assertion that `latency_ms` on all results in a batch are equal.

</details>

---

<details>
<summary>Details</summary>

### `asyncio.get_event_loop()` inside coroutines

Both `_run_text_inference` and `_run_image_inference` obtain the event loop with:

```python
loop = asyncio.get_event_loop()
logits, embeddings = await loop.run_in_executor(None, _run)
```

Since these methods are `async def` and are always awaited from within a running event loop, the correct call is `asyncio.get_running_loop()`. The `get_event_loop()` form was deprecated in Python 3.10 (PEP 644) and raises `DeprecationWarning` in 3.10/3.11. In Python 3.12 it raises `RuntimeError` when called in a context where no current event loop has been set by the caller — which is the norm for asyncio programs that use `asyncio.run()`. This is a one-line fix in each method but it is blocking for any deployment targeting Python 3.12.

### Per-item latency stamping

`run_batch` starts a `perf_counter` before `asyncio.gather`. Inside the post-gather loop it computes:

```python
latency_ms = (time.perf_counter() - start) * 1000.0
```

This is evaluated once per iteration, so `results[0].latency_ms` captures the time elapsed up through processing item 0 (effectively the gather overhead plus the cost of a single dictionary lookup), while `results[N-1].latency_ms` captures something closer to true batch latency. The `total_latency_ms` variable computed after the loop is only used for logging — it is not stored in any result field. Any SLO or p99 measurement based on `result.latency_ms` will be systematically wrong for all items except the last.

### Image fetch fallback observability gap

`_run_image_inference` tracks which indices got `None` from `_safe_fetch` and fills them with `np.zeros((3, 224, 224))`. The zero tensor feeds the ONNX session and produces a valid (though arbitrary) logit and embedding. The `ModerationResult` for that item is returned without any marker indicating the image path was a fallback. A moderation system that needs to quarantine or re-queue items with missing images has no way to filter them from results. Adding `image_fetch_failed: bool = False` to `ModerationResult` and setting it to `True` in the zero-fill branch is the minimal fix; alternatively, the zero-vector embedding and its downstream confidence could be suppressed entirely and the result could rely only on the text path.

### Warmup network dependency

`warmup.py` hardcodes `https://picsum.photos/224` and issues four concurrent fetches per pass, three passes by default. This is a real outbound HTTP dependency at startup. There is no timeout beyond the httpx 5-second per-request default, so a flaky external host can delay startup by up to 15 seconds. In CI or air-gapped environments the warmup will either hang or fail entirely. The warmup function signature should accept an `image_url` override or a `skip_image_warmup: bool` flag.

### Missing test coverage

No test files exist in the repository. The ONNX stub exporter in `scripts/export_onnx_stubs.py` produces deterministic models suitable for unit testing without GPU or real weights, but nothing uses them in a test context. The areas most in need of coverage are: the `asyncio.gather` concurrency path (verifying that text and image inference run concurrently and that results are matched to inputs in the correct order), the zero-vector fallback path (verifying that a `None` return from `_safe_fetch` does not abort the batch), and the `latency_ms` correctness bug described above (which a test would immediately expose).

</details>

---

<details>
<summary>File map</summary>

| File | What changed |
|---|---|
| `src/models/inference.py` | New: multimodal inference engine, `ModerationResult` schema, preprocessing helpers, singleton |
| `src/models/model_utils.py` | New: ONNX session cache, sigmoid, L2 normalizer, async HTTP image loader |
| `src/models/warmup.py` | New: startup warmup coroutine using dummy payloads |
| `scripts/export_onnx_stubs.py` | New: deterministic ONNX stub exporter for text and image models |

Full diff: `git diff main -- src/models/ scripts/export_onnx_stubs.py`

</details>
