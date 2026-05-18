# 🏛️ AI Local Executive Team

**A fully local multi-agent AI board — five specialized models collaborating like a real executive team.**

No cloud. No API keys. No subscriptions. Runs 100% on your own GPU.

---

## What This Is

Most people running local AI do this:

```
Open LM Studio → load one model → chat with it
```

This project does something different:

```
Load multiple specialized models simultaneously
Route each question to the right expert automatically
Stream all five executives responding in sequence
Serve any OpenAI or Ollama compatible client
Switch between model presets live without restarting
Learn what works on your hardware and improve automatically
```

Instead of one model wearing every hat, you get a **board of directors** — each model optimized for its role, each thinking about your question from a different angle.

---

## ✨ Highlights

- **True multi-agent collaboration** — CEO opens, CTO+CFO+CPO run in parallel, COO plans execution, CEO closes with a final decision
- **Smart intent routing** — full board or single agent picked automatically from your message
- **6 quality presets** — Fastest (30s) to Nuclear (40 min), switch with a single keypress
- **Live hotkeys** while running — `SPACE`, `O`, `T`, `X`, `S`
- **Self-learning system** — tracks your hardware's real performance and adjusts timeouts automatically
- **Built-in model auto-tuner** — `T` key benchmarks context/cache/attention configs and saves the winner
- **Dual API** — speaks both OpenAI and Ollama format simultaneously
- **Works with every major local AI client** out of the box

---

## The Executive Board

| Role | Model (Balanced Preset) | Cloud Equivalent | Speed |
|------|------------------------|-----------------|-------|
| 👑 **CEO** | Gemma 4 31B | GPT-4o (strategic) | 4–6 tok/s |
| ⚙️ **CTO** | Qwen3 Coder 30B | Claude Sonnet (coding) | 4–7 tok/s |
| 📊 **CFO** | DeepSeek R1 32B | o1 Preview (reasoning) | 3–5 tok/s |
| 🎯 **CPO** | Gemma 4 26B A4B | Claude Sonnet (product) | 4–7 tok/s |
| 📋 **COO** | Qwen2.5 Coder 14B | GitHub Copilot Pro | 35–45 tok/s |

The CEO speaks first (frames the problem), CTO+CFO+CPO run in parallel (each from their angle), COO synthesizes into an execution plan, and the CEO closes with a final recommendation.

Monthly cost: **$0**

---

## Hardware

Designed and tested on:
- **GPU:** NVIDIA RTX 3060 — 12 GB VRAM
- **RAM:** 64 GB system RAM
- **CPU:** Ryzen 9 5900X
- **Backend:** LM Studio (local server on `localhost:1234`)

Large models (CEO 31B, CFO 32B) run in **hybrid mode** (GPU + RAM offload). Small models (COO 14B, Autocomplete 2B) run **fully on GPU**.

Works on any NVIDIA GPU with 8GB+ VRAM. 24GB+ cards run all models at full GPU speed.

---

## Quick Start

**Windows (recommended):**
```
1. Install Python 3.11+ from https://python.org
2. Install LM Studio from https://lmstudio.ai
3. In LM Studio: Local Server tab → Start Server (port 1234)
4. Download at least one model (see Recommended Models below)
5. Double-click install.bat  ← first time only
6. Double-click start.bat    ← every time after
```

**Manual:**
```bash
pip install -r requirements.txt
copy config.yaml.example config.yaml   # Windows
# cp config.yaml.example config.yaml  # Linux/Mac
python swarm_bridge_server.py
```

**Connect a client:**
```
OpenWebUI  → add as Ollama connection:  http://localhost:5555
Continue   → add as OpenAI provider:    http://localhost:5555/v1
AnythingLLM → Generic OpenAI:          http://localhost:5555/v1
Any client → any OpenAI-compatible URL: http://localhost:5555/v1
```

---

## Recommended First Models

> For RTX 3060 12GB + 64GB RAM. Mileage varies with different hardware.

| Model | Purpose | VRAM | Notes |
|-------|---------|------|-------|
| `google/gemma-3-12b` | General — all roles | ~8.2GB | **Best first model. Most stable.** |
| `qwen/qwen2.5-coder-14b-instruct` | Coding / COO | ~9GB | Great for programming tasks |
| `microsoft/phi-4-reasoning-plus` | Fast reasoning / CFO | ~9.1GB | Fits fully on GPU, quick |
| `google/gemma-4-e2b` | Autocomplete | ~4.4GB | Tiny and very fast |

**Start with Gemma 3 12B** — load it in LM Studio with these settings for the best experience on a 12GB card:
- Context length: **8192**
- GPU offload layers: **48**
- KV cache: **K = Q8_0 / V = Q4_0**

This keeps VRAM under 12GB while maximizing context and speed. Use the **gemma12b preset** first — it runs one model for all five roles, most reliable way to test the full board meeting.

---

## Presets

Switch any time with `SPACE` → number while the bridge is running, or run `tools/swap.bat`.

| Preset | Models | Time | Best For |
|--------|--------|------|----------|
| **FASTEST** | Tiny GPU-only | ~30s–1 min | Quick questions, autocomplete |
| **FAST** | All 8–12B GPU | ~2–4 min | Daily coding and tasks |
| **BALANCED** | Mixed GPU/hybrid | ~4–7 min | **Best daily driver** |
| **SMART** | Big + reasoning | ~8–15 min | Deep analysis, hard problems |
| **NUCLEAR** | Everything maxed | ~20–40 min | Maximum intelligence |
| **GEMMA12B** | Single model all roles | ~3–6 min | Most stable, great starting point |

### Adding Your Own Models to a Preset

Edit **`config/my_models.yaml`** — uncomment any line and swap the model ID:

```yaml
board:
  ceo: google/gemma-4-31b    # ← change this to any model you have downloaded
# cto: qwen/qwen3-coder-30b  # ← commented = uses preset default
```

One change here applies to **all presets** — no need to edit individual preset files.

---

## Hotkeys (while bridge is running)

| Key | Action |
|-----|--------|
| `SPACE` | Open preset switcher |
| `1`–`6` | Select preset (Fastest → Gemma12B) |
| `O` | Optimize — load model with best known settings + live performance report |
| `T` | **Tune** — benchmark config matrix, find fastest settings, save winner permanently |
| `X` | Abort current generation immediately |
| `S` | Show bridge status (loaded models, request count, active preset) |
| `ESC` / `Q` | Cancel menu, or stop T-tune early (keeps best result so far) |
| `Ctrl+C` | Stop the bridge |

---

## How Routing Works

When you use the `executive-swarm` model, the bridge reads your message and picks the right handler:

```
Message arrives
    │
    ├─ OpenWebUI background request? (auto-suggest, title gen)
    │   └─ Return empty instantly — no model used
    │
    ├─ Short / casual? (< 40 chars or matches casual patterns)
    │   └─ Route to COO (fast 14B, full GPU, near-instant)
    │
    ├─ Model explicitly selected? (ceo / cto / cfo / cpo / coo)
    │   └─ Route directly to that executive
    │
    └─ executive-swarm → detect intent from message content:
        ├─ Strategy / big picture  → full board (all 5 agents)
        ├─ Technical / code        → CTO only
        ├─ Financial / risk        → CFO only
        ├─ Product / UX            → CPO only
        └─ Tasks / planning        → COO only
```

You can also bypass routing entirely by selecting a specific model (CEO, CTO, etc.) in your client's model picker.

---

## Vision Analysis

The bridge includes a multimodal vision agent for analyzing images and screenshots:

```bash
python agents/vision.py screenshot.png "What errors do you see?"
python agents/vision.py chart.jpg "Summarize this data"
python agents/vision.py diagram.png "Explain this architecture"
```

Uses `qwen/qwen2.5-vl-7b-instruct` by default — fits fully on a 12GB GPU alongside other models.

---

## Client Integrations

Ready-to-use config files are in `docs/client-configs/`:

| File | Client |
|------|--------|
| `continue.yaml` | Continue VS Code extension (with slash commands) |
| `claude-code.json` | Claude Code CLI |
| `cline-kilocode.json` | Cline / Kilo Code VS Code extensions |
| `openwebui-setup.txt` | OpenWebUI Docker — step by step |
| `anythingllm.txt` | AnythingLLM desktop |

**OpenWebUI** — add as **Ollama** connection (not OpenAI):
```
Inside Docker: http://host.docker.internal:5555
Direct:        http://localhost:5555
```

**Continue (VS Code)** — add as OpenAI provider:
```yaml
apiBase: http://localhost:5555/v1
apiKey: local
model: executive-swarm
```

**Cline / Kilo Code** — set provider to OpenAI Compatible, base URL `http://localhost:5555/v1`

**Claude Code** — set `ANTHROPIC_BASE_URL=http://localhost:5555` in environment

---

## Bridge API

The bridge listens on `http://localhost:5555` and speaks both formats simultaneously.

### OpenAI Format
```
POST /v1/chat/completions    ← Main chat endpoint
GET  /v1/models              ← List all available models
GET  /v1/cache               ← Inspect response cache
DELETE /v1/cache             ← Clear response cache
GET  /v1/preset              ← Current preset + available presets
POST /v1/preset/<name>       ← Switch preset via API
GET  /v1/board               ← Full board status + timeouts
GET  /health                 ← Full system health check
```

### Ollama Format (for OpenWebUI)
```
GET  /api/tags               ← List models
GET  /api/ps                 ← Currently loaded model
POST /api/chat               ← Chat endpoint
POST /api/generate           ← Generate endpoint
POST /api/show               ← Model info
GET  /api/version            ← Reports Ollama 0.3.0
```

---

## Self-Learning Performance System

Every generation is tracked in `model_performance.json`. The system improves automatically over time:

**What gets tracked:**
- **TTFT** (Time To First Token) — running average with recency bias
- **Timeout count** — how often a model fails to respond in time
- **Speed** (tokens/second) — average generation speed
- **Success rate** — fraction of runs that completed

**What the system does with it:**
- **Learned timeout** — if your model consistently needs more time than the hardcoded limit, the bridge raises it automatically (`avg_ttft × 4`, needs 3+ samples)
- **Reliability score** — composite: `success_rate × tok_per_sec × (1 - timeout_penalty)`
- **Best model for role** — `get_best_model_for_role()` returns your most reliable model per role based on actual usage history

**View the report:**
```bash
python model_performance_log.py
```
Shows all models sorted by reliability, timeout rates color-coded green/yellow/red.

---

## Response Cache

Short and repeat queries are cached in memory for **5 minutes** (max 128 entries). Identical questions return instantly without touching the model.

- `full_board` meetings are **never** cached — always fresh
- Cache survives across requests in the same session
- Clear: `DELETE http://localhost:5555/v1/cache`
- Inspect: `GET http://localhost:5555/v1/cache`

---

## The O Key — Optimize

Pressing `O` ensures the current CEO model is loaded with the best known settings for your hardware:

1. Reads `learned_settings.json` — context size, KV cache type, Flash Attention, GPU layer count
2. If already loaded → reports ready, exits (no disruption)
3. If not loaded → loads with optimal settings

> *"Make sure the current model is loaded with the best config we know works on this machine."*

Run `O` after first launch, or any time you suspect a model loaded with suboptimal settings.

---

## The T Key — Auto-Tuner

Pressing `T` benchmarks a matrix of LM Studio load configs for the current model and saves the fastest one permanently.

**What gets tested:** context window (2K / 4K / 8K / 16K) × KV cache (q8_0 / f16) × Flash Attention (on/off)

**Each cycle:** unload → write config → reload → benchmark prompt → measure tok/s → record

**Output:** ranked results table with plain-English notes on why each config scored the way it did

**After tuning:** `O` key loads the model with the winner automatically. Run once per model — permanent in `learned_settings.json`.

Press `ESC` between configs to stop early and keep the best result found so far.

---

## What You Actually Built

### vs. Cloud AI Services

| | This System | GPT-4o API | Claude API | Gemini API |
|---|---|---|---|---|
| **Cost** | $0/month | ~$50–400/month | ~$80–500/month | ~$30–200/month |
| **Privacy** | 100% local | Sent to OpenAI | Sent to Anthropic | Sent to Google |
| **Offline** | Yes | No | No | No |
| **Customizable** | Full source | No | No | No |
| **Multi-agent** | 5 roles | Single model | Single model | Single model |
| **Quality** | GPT-4o to frontier | GPT-4o | Claude 3.5 | Gemini Pro |

The tradeoff is real: cloud is faster and simpler. Local is free, private, and runs indefinitely.

### vs. Other Local Solutions

| | This System | LM Studio (plain) | Ollama | AutoGen | CrewAI |
|---|---|---|---|---|---|
| **Setup** | 1 bat file | Manual | CLI | Python code | Python code |
| **Multi-agent** | Built-in (5 roles) | No | No | Yes (complex) | Yes (complex) |
| **OpenWebUI** | Native | Partial | Native | No | No |
| **Preset switching** | Hotkey / API | Manual reload | Manual | Code change | Code change |
| **Performance tracking** | Automatic | No | No | No | No |
| **Self-learning timeouts** | Yes | No | No | No | No |
| **VS Code integration** | All major extensions | None | Partial | No | No |

AutoGen and CrewAI are powerful but require writing Python orchestration code for every workflow. This ships with the orchestration built in — the board meeting is the default behavior.

---

## Key Technical Discoveries

Things that weren't obvious until actually building and running this:

### LM Studio API
- **No true parallel execution** — LM Studio runs one model at a time. "Parallel" CTO+CFO+CPO means they queue in Python threads while the orchestrator makes it look concurrent. Real win is that the next request processes while the current one streams.
- **Model switching is expensive** — unload + reload takes 15–60 seconds. Presets avoid this by committing to one lineup per session.
- **Context window size matters more than model size** — a 16K context 8B model can be slower than a 4K context 14B model. The T key finds the sweet spot.
- **Flash Attention isn't always faster** — on some CUDA versions and model architectures it slows things down. The tuner tests both.
- **`/api/v0/models` vs `/v1/models`** — `/api/v0/models` returns `state: loaded/not-loaded` per model. `/v1/models` returns all downloaded models regardless of state. The loader uses v0 to know what's actually in VRAM.

### Prompt Engineering
- **System prompts create real personality differences** — the same Gemma 3 12B gives measurably different answers depending on which executive system prompt it receives. The gemma12b preset runs one model through all five roles using this.
- **CEO-first ordering matters** — CEO opening sets a frame the other executives respond to. CEO closing (as final word) produces better synthesis because it can directly address conflicts between the middle responses.
- **Reasoning models generate `<think>` blocks** — DeepSeek R1 and Phi-4 think out loud before answering. These are visible in the terminal, stripped from client responses. They add 20–40% to generation time but significantly improve analytical quality.

### Performance
- **OpenWebUI sends 3–5 background requests per user message** — auto-suggest, title generation, follow-up questions. Without the short-circuit check, these would trigger partial board meetings for every real message.
- **TCP connection reuse saves meaningful time** — `requests.Session()` instead of per-request connections saves ~50ms per model call. Over a 5-agent board meeting that adds up.
- **Response caching cuts repeat queries to near-zero** — second identical question returns in milliseconds. Full board meetings are never cached.

---

## Honest Limitations

- **No memory between sessions** — each board meeting starts fresh. Executives don't remember yesterday's conversation.
- **No codebase awareness** — executives can't read your files unless you paste content into chat. Continue's `@codebase` helps but has limits.
- **No web search** — all knowledge is in the model weights. Nothing post-training-cutoff.
- **Sequential execution** — despite parallel Python threads, LM Studio runs one model at a time. A 5-agent meeting is 5 sequential calls.
- **Cold start latency** — first request waits for model to load (15–60s). Subsequent requests are fast.
- **No inter-agent awareness during a meeting** — CFO doesn't know what CTO said. Agents run independently; only the COO synthesis and CEO final see all previous responses.
- **12GB VRAM ceiling** — models over ~20GB need RAM offload. Speeds drop to 2–8 tok/s. Nuclear preset is for when quality beats speed.
- **Gemma 4 31B context limit on 12GB** — runs at ~2048 context on hybrid load. Around 1500 words before hitting the ceiling.

---

## Priority Roadmap

1. **Conversation memory** — compact session summaries injected into new conversations. Even 3–4 bullet points of prior context dramatically improves ongoing project quality.
2. **Project / codebase awareness** — lightweight indexer that builds a file/function map and injects relevant context automatically.
3. **Web search** — let CFO and CTO search before responding. Most useful for current pricing, library support, recent events.
4. **Context optimization** — trim conversation history intelligently to keep context windows small and generation fast.
5. **Multi-turn board meetings** — follow-up questions re-engage only the relevant executives instead of restarting the full meeting.
6. **Agent specialization memory** — let the CTO remember code patterns it generated, the CFO remember cost estimates from earlier in the project.
7. **Quantization testing in T-key** — add Q4 vs Q5 vs Q8 to the benchmark matrix to surface the quality/speed tradeoff per model.

---

## File Structure

```
ai-executive-team/
│
├── start.bat                   ← Main entry point (run this)
├── install.bat                 ← First-time setup
├── stop_openwebui.bat          ← Stop OpenWebUI Docker container
├── restart_openwebui.bat       ← Restart OpenWebUI
├── check_status.bat            ← Check ports 1234 and 5555
├── kill_bridge.bat             ← Free port 5555
│
├── config.yaml.example         ← Copy to config.yaml before first run
├── swarm_bridge_server.py      ← MAIN bridge server
├── executive_swarm.py          ← Board meeting orchestrator
│                                  (CEO → CTO+CFO+CPO parallel → COO → CEO final)
├── load_model.py               ← Smart model loader
├── model_performance_log.py    ← Self-learning performance tracker
├── preset_selector.py          ← Preset switcher (used by start.bat)
├── setup.py                    ← First-time config generator
│
├── agents/
│   ├── ceo.py                  ← Strategic direction
│   ├── cto.py                  ← Technical analysis + code
│   ├── cfo.py                  ← Financial + risk
│   ├── cpo.py                  ← Product + UX
│   ├── coo.py                  ← Execution planning
│   └── vision.py               ← Image / screenshot analysis
│
├── routers/
│   ├── local_router.py         ← LM Studio HTTP client (connection pooling)
│   ├── hybrid_router.py        ← Intent detection
│   └── cloud_router.py         ← Cloud API fallback (optional)
│
├── core/
│   └── config_loader.py        ← Loads config.yaml + active preset
│
├── config/
│   ├── my_models.yaml          ← Your model overrides (edit this to swap models)
│   └── routing_rules.yaml      ← Intent detection trigger words
│
├── presets/
│   ├── fastest.yaml
│   ├── fast.yaml
│   ├── balanced.yaml
│   ├── smart.yaml
│   ├── nuclear.yaml
│   └── gemma12b.yaml
│
├── docs/
│   └── client-configs/
│       ├── continue.yaml           ← Continue VS Code extension
│       ├── claude-code.json        ← Claude Code CLI
│       ├── cline-kilocode.json     ← Cline / Kilo Code
│       ├── openwebui-setup.txt     ← OpenWebUI Docker guide
│       └── anythingllm.txt         ← AnythingLLM desktop
│
├── tools/
│   ├── swap.bat                ← Switch presets
│   └── ...
│
├── learned_settings.json       ← Auto-generated (gitignored)
└── model_performance.json      ← Auto-generated (gitignored)
```

---

## Batch Scripts Reference

| Script | What It Does |
|--------|--------------|
| `start.bat` | **Main entry point** — preset picker, LM Studio check, bridge launch |
| `install.bat` | First-time setup — Python check, packages, config, verification |
| `stop_openwebui.bat` | Stop the OpenWebUI Docker container |
| `restart_openwebui.bat` | Restart OpenWebUI (creates container fresh if needed) |
| `check_status.bat` | Check if ports 1234 and 5555 are active |
| `kill_bridge.bat` | Kill process on port 5555 to free it |
| `tools/swap.bat` | Switch presets while bridge is stopped |

---

## Troubleshooting

**Bridge says LM Studio is not running**
→ Open LM Studio → Local Server tab → Start Server → wait for green "Running"

**Model times out**
→ Press `S` to see what's loaded
→ After a few timeouts the learned timeout raises automatically
→ Press `T` to tune — a smaller context window often eliminates timeouts

**OpenWebUI shows no models / connection error**
→ Make sure you added the bridge as **Ollama** (not OpenAI)
→ URL: `http://localhost:5555` (no `/v1` suffix)
→ Inside Docker: use `http://host.docker.internal:5555`

**Port 5555 already in use**
→ Run `kill_bridge.bat` then try again

**Wrong model responding / preset not applying**
→ Press `S` key or `GET /v1/board` to see active state
→ Run `python core/config_loader.py test` to verify model resolution

**All executives timed out**
→ Switch to FAST or FASTEST preset — current models may be too large for available VRAM
→ Press `T` to tune — smaller context window frees VRAM

**Performance slower than expected**
→ Press `T` to run the auto-tuner — Flash Attention and KV cache settings have major impact
→ Press `O` to check the live performance report

**Reset learned settings:**
```bash
del model_performance.json
del learned_settings.json
# Bridge rebuilds both from scratch on next run
```

---

## Files That Are Auto-Generated (Not Committed)

| File | What It Contains |
|------|-----------------|
| `learned_settings.json` | Best load config per model for your hardware |
| `model_performance.json` | Speed, TTFT, timeout history per model |
| `logs/` | Request logs |

These are in `.gitignore` — they're specific to your machine. Someone else's RTX 3090 will have completely different optimal settings than your RTX 3060.

---

## Support

If this saved you money on API bills or you just want to say thanks:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-support%20me-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/techfreq)
[![PayPal](https://img.shields.io/badge/PayPal-donate-00457C?logo=paypal&logoColor=white)](https://paypal.me/techfreq)

Not required — the project is free forever either way.

---

## License

MIT License — free to use, modify, and build upon.

---

## Acknowledgements

**Built for:** people who want frontier-quality AI reasoning without sending their data to the cloud.

**Powered by:**
- [LM Studio](https://lmstudio.ai) — local model server
- Qwen, Gemma, DeepSeek, Microsoft Phi, and the open model community
- Flask + Waitress bridge server
- Rich terminal UI
- OpenWebUI for ChatGPT-style interface

**A note on how the LM Studio integration was built:**

The loader and router were built by reverse-engineering LM Studio's undocumented internal API from actual server logs — not from official documentation. The `/api/v0/models` endpoint (which returns real loaded/unloaded state), the `instance_id` requirement for model operations, the warmup connection behavior, and the config path discovery were all found through log analysis. The official `/v1/models` endpoint turned out to be largely useless for this use case — it lists downloaded models, not loaded ones.

**AI assistance during development (May 16–18, 2026):**
Grok, ChatGPT, Claude Sonnet 4.6, LM Arena — each hitting API limits in turn, which is half the reason this project exists.

**The vision:** run a local AI swarm where specialized models handle specific tasks in parallel — bridging the gap between single-model chat and the kind of multi-perspective reasoning that actually helps you make better decisions.
