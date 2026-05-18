# Third-Party Notices

This project (the bridge server, orchestration code, agents, and scripts in this
repository) is released under the MIT License — see [LICENSE.txt](LICENSE.txt).

**The MIT License covers only the code in this repository.**

This project connects to, and is designed to work with, a number of third-party
tools, services, and AI models that are developed and owned by their respective
organizations. The author of this project makes no claim of ownership, copyright,
or rights over any of the following. Each must be used in accordance with its own
license and terms of service.

---

## AI Models

The models referenced in the preset files are developed by their respective
organizations. This project does not distribute, modify, or claim any rights over
these models. Downloading and using these models is subject to each provider's
own license.

| Model Family | Developer | License / Terms |
|-------------|-----------|----------------|
| Gemma (3 12B, 4 31B, 4 26B, 4 E2B) | Google DeepMind | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| Qwen / Qwen2.5 / Qwen3 | Alibaba Group | [Qwen License](https://huggingface.co/Qwen) (Apache 2.0 or model-specific) |
| DeepSeek R1 / DeepSeek R1 Distill | DeepSeek AI | [DeepSeek License](https://huggingface.co/deepseek-ai) (MIT or model-specific) |
| Phi-4 / Phi-4 Reasoning Plus | Microsoft | [Microsoft Research License](https://huggingface.co/microsoft) (MIT or MSRLA) |
| Llama (if used) | Meta AI | [Llama Community License](https://llama.meta.com/llama-downloads/) |

> Model licenses vary and may include restrictions on commercial use, redistribution,
> or modification. Always check the specific model card on Hugging Face or the
> provider's website before using a model in a production or commercial context.

---

## LM Studio

This project interfaces with LM Studio's local HTTP API but is **not affiliated
with, endorsed by, or supported by LM Studio, Inc.**

LM Studio is proprietary software with its own Terms of Service:
https://lmstudio.ai/terms

This project uses LM Studio as an optional backend. Nothing in this repository
distributes or modifies LM Studio software.

---

## OpenWebUI

OpenWebUI is an open-source project licensed under the MIT License.
GitHub: https://github.com/open-webui/open-webui

This project is not affiliated with the OpenWebUI project.

---

## Python Packages

| Package | License | Source |
|---------|---------|--------|
| Flask | BSD-3-Clause | https://flask.palletsprojects.com |
| flask-cors | MIT | https://github.com/corydolphin/flask-cors |
| Waitress | ZPL 2.1 | https://github.com/Pylons/waitress |
| Requests | Apache 2.0 | https://requests.readthedocs.io |
| PyYAML | MIT | https://pyyaml.org |
| Rich | MIT | https://github.com/Textualize/rich |
| openai (SDK) | MIT | https://github.com/openai/openai-python |
| python-dotenv | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| keyboard | MIT | https://github.com/boppreh/keyboard |

---

## VS Code Extensions

The client configuration files in `docs/client-configs/` are provided as examples
for use with third-party VS Code extensions. These extensions are not affiliated
with this project and are governed by their own licenses:

- **Continue** — Apache 2.0 — https://github.com/continuedev/continue
- **Cline** — Apache 2.0 — https://github.com/cline/cline
- **Kilo Code** — see extension page
- **Claude Code** — Anthropic Terms of Service — https://anthropic.com

---

## Summary

To be clear:

- ✅ **This project's code** (bridge server, agents, scripts) → MIT License, copyright TechFreq 2026
- ❌ **AI models** → owned by Google, Alibaba, DeepSeek, Microsoft, Meta respectively
- ❌ **LM Studio** → proprietary, owned by LM Studio Inc.
- ❌ **OpenWebUI** → MIT, owned by OpenWebUI contributors
- ❌ **Python packages** → owned by their respective maintainers

Using this project does not grant any rights to the third-party software or models
it connects to. You are responsible for complying with the terms of each tool and
model you use.
