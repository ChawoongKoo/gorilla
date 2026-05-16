# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **Gorilla** monorepo from UC Berkeley, hosting several subprojects related to LLM tool/function calling. The directories at the repo root are largely independent projects with their own dependencies and tooling:

- **`berkeley-function-call-leaderboard/`** — BFCL: the evaluation harness and dataset for benchmarking function-calling LLMs. **This is the active focus of the current work.**
- `gorilla/` — original Gorilla paper code (inference + eval against APIBench)
- `openfunctions/` — Gorilla OpenFunctions model serving + utils
- `goex/` — Gorilla Execution Engine (runtime for safely executing LLM-generated actions)
- `raft/` — Retrieval-Augmented Fine-tuning recipe and Azure AI Studio scripts
- `agent-arena/` — Agent Arena evaluation/client code
- `data/` — APIBench dataset (apizoo, api, apibench)

The subprojects do not share a top-level Python package; each manages its own install. Work scoped to BFCL stays inside `berkeley-function-call-leaderboard/` unless you explicitly need another subproject.

## BFCL: Berkeley Function Calling Leaderboard

All BFCL paths below are relative to `berkeley-function-call-leaderboard/` unless noted. The installable package is `bfcl_eval` (PyPI name: `bfcl-eval` — **not** the unrelated `bfcl` package). Entry-point script is `bfcl`. Current dataset version prefix is `BFCL_v4` (set in `bfcl_eval/constants/category_mapping.py:VERSION_PREFIX`).

### Install

```bash
# From berkeley-function-call-leaderboard/
conda create -n BFCL python=3.10
conda activate BFCL
pip install -e .                       # core install (API-based models only)
pip install -e .[oss_eval_vllm]        # add vLLM backend for self-hosted models
pip install -e .[oss_eval_sglang]      # add SGLang backend (faster but requires SM 80+ GPU)
pip install -e .[wandb]                # optional: WandB logging
```

Python 3.10+ required. Pinned deps include `numpy==1.26.4`, `tree_sitter==0.21.3`, `vllm==0.8.5` (extra), `mistralai==1.7.0`, `anthropic>=0.75.0`.

### Project root, results, and `.env`

`bfcl_eval/constants/eval_config.py` resolves paths from `BFCL_PROJECT_ROOT` (env var). For an editable install this defaults to the `berkeley-function-call-leaderboard/` directory; for a PyPI install you **must** set it explicitly or results end up inside `site-packages/`.

Under the resolved project root, BFCL reads/creates:

- `result/MODEL_NAME/BFCL_v4_TEST_CATEGORY_result.json` — model responses (per-test-case)
- `score/MODEL_NAME/BFCL_v4_TEST_CATEGORY_score.json` — per-category eval output
- `score/data_overall.csv`, `data_live.csv`, `data_non_live.csv`, `data_multi_turn.csv`, `data_agentic.csv`, `data_format_sensitivity.csv` — aggregate CSVs
- `.env` — API keys + endpoint config (copy from `bfcl_eval/.env.example`)
- `test_case_ids_to_generate.json` — only consulted with `--run-ids`; copy from `bfcl_eval/test_case_ids_to_generate.json.example`
- `.file_locks/` — concurrent-IO locks (auto-created)

The `result/`, `score/`, and `.file_locks/` dirs are created automatically when `eval_config` is imported.

`bfcl_eval/.env.example` enumerates every supported provider key (OpenAI, Anthropic, Google, NVIDIA, Grok, Cohere, DeepSeek, Qwen, GLM, Kimi, Mistral, Fireworks, Writer, GoGoAgent, Nanbeige, Mining, DMCito, AWS SSO for Nova, Ling) plus `SERPAPI_API_KEY` (required for `web_search` category), `LOCAL_SERVER_ENDPOINT`/`LOCAL_SERVER_PORT` (vLLM/SGLang defaults: `localhost:1053`), and `REMOTE_OPENAI_BASE_URL`/`REMOTE_OPENAI_API_KEY`/`REMOTE_OPENAI_TOKENIZER_PATH` for arbitrary OpenAI-compatible endpoints.

### Running the benchmark

Two-phase workflow: **generate** model responses, then **evaluate** them.

```bash
# Discover what's available
bfcl models                    # list all registered model IDs
bfcl test-categories           # list test groups and individual categories

# Phase 1: generate responses (writes to result/)
bfcl generate --model MODEL_ID --test-category CATEGORY[,CATEGORY...] --num-threads N

# Phase 2: score responses (writes to score/)
bfcl evaluate --model MODEL_ID --test-category CATEGORY[,CATEGORY...]

# Quick views
bfcl results                   # list models with response files in result/
bfcl scores                    # print data_overall.csv as a table
```

CLI quirks worth remembering:

- `bfcl generate --model A,B --test-category x,y` uses **commas** to separate values. The legacy `python -m bfcl_eval.openfunctions_evaluation ...` / `python -m bfcl_eval.eval_checker.eval_runner ...` and the root-level `openfunctions_evaluation.py` shim use **spaces** instead. The Typer CLI is preferred; the script paths are kept for backward compatibility.
- Default `--test-category` is `all` (which includes the non-scoring `format_sensitivity`). Use `all_scoring` to limit to leaderboard-scoring categories.
- Available test groups (`bfcl_eval/constants/category_mapping.py`): `all`, `all_scoring`, `single_turn`, `multi_turn`, `live`, `non_live`, `python`, `non_python`, `agentic`, `memory`, `web_search`. Individual categories are listed in `TEST_CATEGORIES.md`.
- `--num-threads` defaults to `1` for API models (respect provider rate limits) and `100` for OSS models (the `LOCAL_SERVER_MAX_CONCURRENT_REQUEST` constant in `eval_config.py`).
- Generation is **resumable**: existing `result/MODEL_NAME/*_result.json` entries are preserved unless `--allow-overwrite`/`-o` is passed. Generated files are sorted by `id` at the end of each run.
- `--run-ids` reads `test_case_ids_to_generate.json` from the project root (not the CWD) and ignores `--test-category`. Sample at `bfcl_eval/test_case_ids_to_generate.json.example`.
- `--include-input-log` adds the fully-transformed inference input to the result file (verbose; see `LOG_GUIDE.md`). `--exclude-state-log` drops backend-state snapshots from multi-turn logs.
- For local OSS models: `bfcl generate --model M --backend {vllm|sglang} --num-gpus 1 --gpu-memory-utilization 0.9 [--local-model-path /path] [--enable-lora --max-lora-rank 128 --lora-modules name=/path ...]`. Use `--skip-server-setup` to point at an already-running OpenAI-compatible endpoint via the `LOCAL_SERVER_*` or `REMOTE_OPENAI_*` env vars.
- `bfcl evaluate --partial-eval` lets the scorer silently skip IDs missing from a result file (useful when paired with `--run-ids`); the resulting accuracy will not match the official leaderboard.

### Single-category / single-test iteration

Run one test ID against one model:

```bash
# 1. Copy the template (once)
cp bfcl_eval/test_case_ids_to_generate.json.example ./test_case_ids_to_generate.json
# 2. Edit it to list only the IDs you want, e.g. {"simple_python": ["simple_python_102"]}
# 3. Generate + evaluate that subset
bfcl generate --model gpt-4o-2024-11-20-FC --run-ids --allow-overwrite
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category simple_python --partial-eval
```

Helper scripts under `bfcl_eval/scripts/`:

- `visualize_multi_turn_ground_truth_conversation.py` — simulate ground-truth multi-turn conversations into `bfcl_eval/scripts/ground_truth_conversation/`. Useful to understand what a multi-turn entry is *actually* asking for.
- `check_func_doc_format.py`, `check_illegal_python_param_name.py`, `compile_multi_turn_func_doc.py` — dataset hygiene utilities.

## Architecture

### High-level flow

1. **Dataset load.** Prompts live in `bfcl_eval/data/BFCL_v4_<category>.json`; ground truths in `bfcl_eval/data/possible_answer/BFCL_v4_<category>.json`; multi-turn function specs in `bfcl_eval/data/multi_turn_func_doc/<api>.json`; memory pre-requisite conversations in `bfcl_eval/data/memory_prereq_conversation/`. The pair `(prompts, ground_truths)` is aligned by *index* in the JSON list, not by ID — the evaluator preserves this when it filters with `_subset_entries_by_model_ids` (`bfcl_eval/eval_checker/eval_runner.py`).
2. **Handler dispatch.** `bfcl_eval/constants/model_config.py` maps every supported `--model` ID to a `ModelConfig(model_handler=..., is_fc_model=..., underscore_to_dot=..., input_price=..., output_price=...)`. The `model_handler` field is a class from `bfcl_eval/model_handler/api_inference/` or `local_inference/`. `_llm_response_generation.build_handler` instantiates it with the registry name (the user-facing ID) and the underlying API/HF `model_name`.
3. **Generation.** `_llm_response_generation.generate_results` schedules test cases through a `ThreadPoolExecutor` whose width is `--num-threads`. Memory and multi-turn test cases carry `depends_on` lists; a topological queue (`heapq` + `dependencies`/`children_of` dicts) ensures predecessor entries finish before dependent ones start. A dedicated writer thread serializes file writes through a `queue.Queue` so multiple workers never collide on the same JSON file.
4. **Inference.** `BaseHandler.inference` (in `bfcl_eval/model_handler/base_handler.py`) routes to one of four pipelines based on `is_fc_model` (or `"FC" in registry_name`) and whether the test ID indicates multi-turn: `inference_single_turn_FC`, `inference_multi_turn_FC`, `inference_single_turn_prompting`, `inference_multi_turn_prompting`. FC-mode methods are marked `@final` on the base class; subclasses implement the abstract `_query_FC` / `_parse_query_response_FC` / `_compile_tools` / `_pre_query_processing_FC` / `_add_*_message_FC` / `decode_ast` / `decode_execute` hooks (and their `_prompting` analogues). Each step's transcript flows into `inference_log` (see `LOG_GUIDE.md` for the role taxonomy: `user`, `assistant`, `tool`, `state_info`, `inference_input`, `handler_log`).
5. **Evaluation.** `bfcl_eval/eval_checker/eval_runner.runner` walks `result/` per model and dispatches each result file to one of the checkers based on category type:
   - **AST checker** (`ast_eval/ast_checker.py`) — single-turn `simple_*`, `parallel*`, `multiple`, `live_*`, plus `format_sensitivity`. Uses Python/Java/JavaScript AST parsing (via `tree-sitter`) under `ast_eval/type_convertor/`.
   - **Relevance/irrelevance checker** — same file; binary decision on whether the model produced any function call.
   - **Multi-turn checker** (`multi_turn_eval/multi_turn_checker.py`) — replays decoded function calls against simulated backends in `multi_turn_eval/func_source_code/` (`gorilla_file_system.py`, `math_api.py`, `message_api.py`, `posting_api.py`, `ticket_api.py`, `trading_bot.py`, `travel_booking.py`, `vehicle_control.py`, `long_context.py`).
   - **Agentic checker** (`agentic_eval/agentic_checker.py`) — for `memory_*` and `web_search_*`. Memory backends live in `multi_turn_eval/func_source_code/memory_{kv,vector,rec_sum}.py` and inherit from `memory_api_metaclass.py`. Web search uses SerpAPI through `web_search.py`.
   - Skipped categories (still in code but excluded from current iteration): `exec_*`, `rest`, `sql`, `chatable`, memory prerequisites.

### Naming conventions for model registry IDs

`MODEL_NAME` on the CLI is the **registry name** (the key in `MODEL_CONFIG_MAPPING`), not the vendor's model name. Convention from `SUPPORTED_MODELS.md`:

- IDs ending in `-FC` invoke the model in native Function-Calling mode (`is_fc_model=True`).
- IDs without the `-FC` suffix use Prompting mode — function specs are injected into the system prompt and the model is expected to emit a function-call-shaped string the handler then parses.
- Many models are registered twice (with and without `-FC`) so both modes can be benchmarked.
- Slashes in IDs (e.g. `meta-llama/Llama-3.1-8B-Instruct`) become underscores in directory names; the `bfcl results` command undoes the substitution for display.
- `evaluate` accepts the slash form (BFCL replaces it with `_` internally); `generate` and result directories use the underscore form. The `_subset_entries_by_model_ids` helper requires the model name match the registry.

### Adding a new model

Walked in detail in `CONTRIBUTING.md`. Summary:

1. Add a handler class in `bfcl_eval/model_handler/api_inference/<vendor>.py` (subclass `BaseHandler`) or `local_inference/<model>.py` (subclass `OSSHandler` from `local_inference/base_oss_handler.py`).
2. Implement at minimum `decode_ast` (returns `[{"func": {"arg": val}}, ...]`) and `decode_execute` (returns `["func(arg=val)", ...]`) — the evaluator calls these on every raw response.
3. For API models, implement the `_FC` and/or `_prompting` hooks in `base_handler.py`. For local models, the `OSSHandler` provides most of the prompting pipeline and you typically only implement `_format_prompt`.
4. Register the model in `bfcl_eval/constants/model_config.py` (`MODEL_CONFIG_MAPPING`) **and** in `bfcl_eval/constants/supported_models.py`. Add a row to `SUPPORTED_MODELS.md`.
5. Set `underscore_to_dot=True` if your FC API rejects `.` in function names; the evaluator transparently rewrites function names with dots to underscores for comparison.

### Files to know

- `bfcl_eval/__main__.py` — Typer CLI. `cli` is the Typer app exposed as the `bfcl` console script.
- `bfcl_eval/_llm_response_generation.py` — generation phase (called from `bfcl generate`).
- `bfcl_eval/eval_checker/eval_runner.py` — evaluation phase (called from `bfcl evaluate`).
- `bfcl_eval/utils.py` — shared helpers: `parse_test_category_argument`, `load_dataset_entry`, `load_ground_truth_entry`, `sort_key`, `is_multi_turn`/`is_memory`/`is_format_sensitivity`/etc. predicates, `populate_initial_settings_for_*`, file-locking wrappers.
- `bfcl_eval/eval_checker/eval_runner_helper.py` — `save_eval_results`, `generate_leaderboard_csv`, `record_cost_latency`, `update_leaderboard_table_with_local_score_file`.
- `bfcl_eval/constants/default_prompts.py` — system prompts injected for prompting-mode models; `MAXIMUM_STEP_LIMIT` cap on multi-turn steps.
- `bfcl_eval/constants/executable_backend_config.py` — which simulated-API classes are stateless vs. need state snapshots.
- `bfcl_eval/openfunctions_evaluation.py` (root-level shim) — legacy entrypoint; calls into `_llm_response_generation.main`. Will be removed in the next major release.

### Conventions

- Each dataset version bumps the `VERSION_PREFIX` constant; downstream filenames (`BFCL_v4_*.json`, `RESULT_FILE_PATTERN`) derive from it. When updating to v5, search for `VERSION_PREFIX` first.
- Result files are JSONL with one object per line; the generator sorts by `id` after each run (`sort_file_content_by_id`).
- Temperatures default to `0.001` — effectively greedy; retrying a failed inference will not change the output, which is why inference errors are recorded as the model's "response" rather than retried (see `multi_threaded_inference` in `_llm_response_generation.py`).
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `TOKENIZERS_PARALLELISM` are forced low in `_llm_response_generation.main` to avoid segfaults from the vector-memory backend; do not undo this casually. The script also forces multiprocessing's start method to `spawn`.

## Vision / Geoguessr (branch `hans-vision`)

`main` is text-only. The `hans-vision` branch adds a multi-modal extension to BFCL — vision (geoguessr + image-augmented web search), true-audio, and text-audio categories. The notes below cover the geoguessr workflow specifically, which is the active focus.

### What's different on this branch

- **Category names are namespaced.** Bare names like `simple_python` become `text:simple_python`; the new modalities are `vision:*`, `true_audio:*`, `text_audio:*`. See `bfcl_eval/constants/category_mapping.py`.
- **`--test-category all` no longer means "everything"** — `category_mapping.py:130` currently points `"all"` at `TEXT_FAILING_TOOLS_CATEGORY`. To get vision, pass `vision:geoguessr`, `all_vision`, or one of `vision:geoguessr_type{1,1_competition,2,3}`.
- **New top-level groups:** `all_vision`, `all_text`, `all_text_scoring`, `all_true_audio`, `all_text_audio`, `all_scoring` (which is now *all modalities* scoring). `vision:web_search` and `vision:geoguessr` are the two vision sub-groups.
- **New optional dep `geopy`** (and transitive `geographiclib`). Already covered by `pip install -e .` on this branch.
- **New evaluator module:** `bfcl_eval/eval_checker/vision_eval/vision_checker.py`. Dispatched from `eval_runner.runner` via `vision_geoguessr_runner` when `is_geoguessr(test_category)` is true. Output CSV adds `score_vision_overall.csv` to the score dir.

### Geoguessr categories

Four variants, each is a single-turn agentic task where the model can drive a streetview camera and must finally emit `(lat, lon)`:

- `vision:geoguessr_type1` — basic
- `vision:geoguessr_type1_competition` — competition split
- `vision:geoguessr_type2`, `vision:geoguessr_type3` — harder variants

Prompts live in `bfcl_eval/data/vision/geoguessr_type*.json` (JSONL, one object per line). Each entry looks like:

```json
{"id": "geoguessr_type1_0",
 "question": [[{"role":"user","content":"You are an agent competing in the GeoGuesser tournament. ... Output the answer in this format: (latitude, longitude). Use decimal degrees (WGS84) ..."}]],
 "initial_config": {"StreetViewAPI": {"lat": "78.22295351095083", "lng": "15.627656919871827"}},
 "involved_classes": ["StreetViewAPI"]}
```

Ground truths are in `bfcl_eval/data/possible_answer/vision/geoguessr_type*.json`: `{"id": "...", "ground_truth": ["lat, lng"]}` — one string with the coordinate pair.

### How scoring works

`vision_eval/vision_checker.py`:

1. `extract_coordinates_from_response` regex-matches `(lat, lon)` from the model's final non-function-call message.
2. `geopy.distance.geodesic(pred, truth).kilometers` → distance.
3. `score = 5000 * exp(-10 * distance_km / 14916.862)` — the GeoGuessr-style points formula (5000 = perfect, decays exponentially over ~half the Earth's circumference). Non-extractable response → `score=0`, `error_type="vision_geoguessr:coordinate_extraction_failed"`.

Per-entry scores are aggregated by the runner into `score/<model>/vision/geoguessr_type*_score.json` and rolled up into `score/score_vision_overall.csv`.

### `StreetViewAPI` client → your server (the HTTP contract)

`bfcl_eval/eval_checker/multi_turn_eval/func_source_code/street_view.py` is **a thin HTTP client**. There is no in-process simulation — every method makes an HTTP call to a streetview backend you must run separately. The base URL defaults to `http://127.0.0.1:18000` and is overridden by `GEOGUESSR_SERVER_URL`. All requests carry `X-Session-ID` once `/connect` has returned one.

The client retries `_connect_host` every 5s in a `while True` loop on connection failure (`street_view.py:120-121`) — meaning if the server isn't up, `bfcl generate` will hang silently, not crash. Start the server first.

**Response envelope (every endpoint):** `{"ok": <bool>, "updates": {...}, "error": {"message": "..."}}`. Non-ok envelopes raise `RuntimeError` with the message. The client unwraps `updates` and reads:
- `updates.session_id` — pinned into the `X-Session-ID` header for all subsequent calls.
- `updates.available_moves` — list of currently-permitted move directions (used to advertise the action space to the model on the next turn).
- `updates.description` — used by `check_direction`.
- `updates.image_base64` — used by `capture_view` to construct an `ImageResult(image_base64=..., mime_type="image/jpeg")`. Scroll/zoom endpoints call `capture_view` after applying their delta, so they too end up reading `image_base64`.

**Endpoints the client calls:**

| Method | Path                | Called by                  | Request body                                                                 | Required in `updates`                          |
|--------|---------------------|----------------------------|------------------------------------------------------------------------------|------------------------------------------------|
| POST   | `/connect`          | `_connect_host`            | `{api_key?, url_signing_secret?, session_id?}` (keys from `GOOGLE_MAPS_*` env) | `session_id`                                   |
| POST   | `/init_panorama`    | `_load_scenario`           | the entry's `initial_config["StreetViewAPI"]`, e.g. `{"lat":"78.22","lng":"15.62"}` | `available_moves`                              |
| GET    | `/check/direction`  | `check_direction`          | —                                                                            | `description`, `available_moves`               |
| POST   | `/capture/view`     | `capture_view` (also scroll/zoom) | —                                                                     | `image_base64`                                 |
| POST   | `/move/{north,northeast,east,southeast,south,southwest,west,northwest}` | `move_*` | — | `available_moves` |
| POST   | `/scroll/{left,right,up,down}` | `scroll_*`         | `{"delta": <degrees>}`                                                       | `available_moves` (and then `capture_view` is called) |
| POST   | `/zoom/{in,out}`    | `zoom_*`                   | `{"delta": <amount>}`                                                        | `available_moves` (and then `capture_view` is called) |
| POST   | `/end_session`      | `_end_session`             | —                                                                            | —                                              |

Notes for the server implementer:

- Pitch is clamped server-side: `scroll_up` so pitch ≤ 90, `scroll_down` so pitch ≥ −90. Zoom-out is clamped at 0. The client passes through negative deltas as their absolute value (the docstrings promise this); your server can rely on receiving positive numbers but should also defensively handle either.
- `available_moves` should be the list of directions where adjacent panoramas exist. The handler surfaces this back to the LLM each turn — if empty, the model can still scroll/zoom/capture but can't move.
- `capture_view` is the *only* endpoint the model uses to actually look at the world; the model decides when to call it. Make sure it returns a valid JPEG base64 (no data URL prefix — the client wraps it with `mime_type="image/jpeg"` separately).
- Session IDs: `/connect` mints them. Multiple BFCL workers (`--num-threads > 1`) will each open their own session; isolate state per `X-Session-ID`.
- `initial_config` is what BFCL hands you in `/init_panorama`. For `geoguessr_type1` it's `{"lat": "...", "lng": "..."}` (strings, not floats). Coerce as needed.

### Required env vars

In `berkeley-function-call-leaderboard/.env`:

```
# Where StreetViewAPI talks to (your server)
GEOGUESSR_SERVER_URL=http://127.0.0.1:18000

# Forwarded to the server inside POST /connect body if set
GOOGLE_MAPS_API_KEY=
GOOGLE_MAPS_URL_SIGNING_SECRET=

# Plus the API key for whichever vision-capable model you run
OPENAI_API_KEY=...     # for gpt-4o*/gpt-5*/gpt-4.1*
ANTHROPIC_API_KEY=...  # for claude-*-sonnet/opus
GOOGLE_API_KEY=...     # for gemini-*
```

`GOOGLE_MAPS_*` are only consulted inside `_connect_host` and forwarded to your server via the `/connect` body. BFCL itself never calls Google Maps — that's your server's concern. Leave them blank if your server doesn't need them.

### Running geoguessr end-to-end

```bash
# 0. Start your streetview server (your code, not BFCL's). Verify reachable:
curl http://127.0.0.1:18000/connect -X POST -H 'content-type: application/json' -d '{}'

# 1. Activate env, cd to BFCL
conda activate BFCL
cd berkeley-function-call-leaderboard

# 2. Set GEOGUESSR_SERVER_URL + provider key in .env

# 3. Smoke test against one ID
cp bfcl_eval/test_case_ids_to_generate.json.example ./test_case_ids_to_generate.json
# edit it to: {"vision:geoguessr_type1": ["geoguessr_type1_0"]}
bfcl generate --model gpt-4o-2024-11-20-FC --run-ids --allow-overwrite
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category vision:geoguessr_type1 --partial-eval

# 4. Full run
bfcl generate --model gpt-4o-2024-11-20-FC --test-category vision:geoguessr --num-threads 4
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category vision:geoguessr
```

Results: `result/<model>/vision/geoguessr_type*_result.json`.
Scores: `score/<model>/vision/geoguessr_type*_score.json` plus `score/score_vision_overall.csv`.

### Tips when integrating your own server

- Run the smoke test against one ID first. If the client hangs on "Failed to connect", `GEOGUESSR_SERVER_URL` is wrong or the server isn't bound to that interface.
- If `available_moves` is missing from your `/init_panorama` response, the model will see no directional options on turn 1 and either pick `capture_view`/scroll/zoom only, or guess immediately. Always include it.
- If you return `image_base64=""`, scoring still runs but the model has nothing to look at. Watch for this in `result/.../*_result.json` → `inference_log`.
- The `requests.Session` in the client persists across calls *per worker thread*. Your server can keep panorama state in-memory keyed by `X-Session-ID`; the client calls `/end_session` at the end of each test entry to let you free it.
- `_timeout = (15, None)` — 15s connect timeout, no read timeout. A slow `/capture/view` won't time out, but a hung TCP handshake will.
- The model talks to BFCL, which talks to your server. BFCL passes each `capture_view` image into the *next* assistant turn via the multi-turn FC pipeline (see `BaseHandler.inference_multi_turn_FC`). The model handler you pick must support image inputs in its tool-result channel — `OpenAIResponsesHandler`, `ClaudeHandler`, `GeminiHandler` all do; many local handlers do not.

## Other subprojects (quick orientation)

You usually won't need these for BFCL work, but here's the lay of the land:

- `gorilla/inference/` and `gorilla/eval/` — original Gorilla CLI/server + APIBench eval scripts. Their own README under `gorilla/inference/README.md`.
- `openfunctions/openfunctions-v1/` — training/eval for OpenFunctions v1; `openfunctions/utils/` — shared helpers.
- `goex/` — Go Execution engine: `exec_engine/` (runtime), `authorizations/` (OAuth), `docker/` (sandbox images), `demo/`, `function/`.
- `raft/` — RAFT fine-tuning pipeline; `raft/azure-ai-studio-ft/` for Azure jobs, `raft/sample_data/`, `raft/tests/`.
- `agent-arena/` — `client/` and `evalutation/` (note the spelling in-repo) for the Agent Arena leaderboard.
- `data/` — APIBench (`api/`, `apibench/`, `apizoo/`); used by the original `gorilla/` paper code.

Each of these has its own README; do not assume BFCL's tooling/conventions carry over.
