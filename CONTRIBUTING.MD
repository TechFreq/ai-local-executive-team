# Contributing to AI Local Executive Team

Thank you for considering contributing! This project is built for local AI enthusiasts
who want multi-agent reasoning without cloud dependencies.

---

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Make your changes
4. Test thoroughly — especially with different presets and hardware configs
5. Submit a Pull Request with a clear description of what changed and why

---

## Development Setup

```bash
# 1. Clone your fork
git clone https://github.com/your-username/ai-local-executive-team

# 2. Run first-time setup
install.bat

# 3. Start LM Studio → Local Server → Start Server (port 1234)

# 4. Launch the bridge
start.bat
```

---

## Areas That Need Help

| Area | What's Needed |
|------|--------------|
| **Conversation memory** | Summarize past sessions, inject context into new ones |
| **Codebase awareness** | Index project files, auto-inject relevant context |
| **Web search** | Let executives search before responding |
| **New backends** | Support for vLLM, Ollama, llama.cpp directly |
| **Better vision** | Improve image/screenshot analysis pipeline |
| **Linux/macOS** | Port the `.bat` scripts to shell scripts |
| **Documentation** | Examples, tutorials, video walkthroughs |
| **Testing** | Unit tests for the router, config loader, cache |

---

## Code Style

- Python: follow PEP 8, use `rich` for terminal output
- Keep agent system prompts in `agents/*.py` — one file per role
- Config changes go through `core/config_loader.py` — never read YAML directly in other files
- New presets go in `presets/` as YAML following the existing `board/utility/fallback/display` structure
- Batch scripts: test on Windows 10/11, use `setlocal enabledelayedexpansion`

---

## Adding a New Preset

1. Create `presets/yourname.yaml` following this structure:

```yaml
name: "YourPreset"
description: "What this preset is for"

board:
  ceo: model/id-here
  cto: model/id-here
  cfo: model/id-here
  cpo: model/id-here
  coo: model/id-here

utility:
  autocomplete: model/id-here
  vision: model/id-here
  embed: model/id-here

fallback:
  primary: model/id-here
  general: model/id-here
  fast_reasoning: model/id-here
  last_resort: model/id-here

display:
  vram: "~XGB"
  meeting_time: "X-Y min"
  quality: "★★★☆☆"
```

2. Test it with `python core/config_loader.py switch yourname`
3. Add it to the preset switcher in `preset_selector.py`

---

## Reporting Bugs

Open an issue with:
- Your GPU and VRAM
- Which preset you were using
- The exact error message or behavior
- LM Studio version

---

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Describe what hardware you tested on (GPU, VRAM, RAM)
- If you changed the bridge, test with both OpenWebUI and Continue
- Don't commit `learned_settings.json` or `model_performance.json` — these are personal to your hardware

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
See [LICENSE.txt](LICENSE.txt) for details.

Third-party models and tools (LM Studio, Gemma, Qwen, DeepSeek, etc.) retain their own
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
