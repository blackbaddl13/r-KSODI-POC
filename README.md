[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
[mit]: https://opensource.org/licenses/MIT
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg

[![MIT License][mit-shield]][mit]  
<br>
The source code is licensed under an [MIT License][mit].


[![CC BY 4.0][cc-by-image]][cc-by]
<br>
The prompts are licensed under a [Creative Commons Attribution 4.0 International License][cc-by].


# KSODI-Light Agent

## 🧭 Introduction

The **KSODI-Light Agent** is a reference implementation of the *semantic resonance module*, based on the [KSODI Method](https://github.com/Alkiri-dAraion/KSODI-Methode). It provides a minimal yet expressive system designed to demonstrate how tonality, symbolic interaction, and reflective conversational flow can be implemented within a multi-agent LLM environment.

This implementation, referred to as **KSODI Light**, focuses solely on the *semantic*, *resonance*, and *tonality* dimensions. It is **not** a complete assistant framework, but rather a **modular core** meant to be embedded into larger AI systems.

Nonetheless, it can be tested as a standalone prototype to validate agent behavior, resonance handling, and symbolic navigation.

> Big thanks to [@Alkiri-dAraion](https://github.com/Alkiri-dAraion) for resonating early and building together



> ⚠️ Please note: this is a **prototype**, offered without warranty. See roadmap for future plans and integrations.



## 🚀 Usage / Getting Started

To get the KSODI-Light Agent running locally, follow these basic steps:

### 🔧 Prerequisites

* Python 3.10 or newer
* An [OpenAI API Key](https://platform.openai.com/account/api-keys)

  * Export it in your shell or `.env` file as `OPENAI_API_KEY`

### 📦 Installation & Run

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/ksodi-light.git
cd ksodi-light

# Install dependencies
pip install -r requirements.txt

# Start the agent\python main.py
```

### 🧠 Agent Name & Identity

The agent is called `AI` by default. You may customize this in the context configuration. However, for **consistent symbolic resonance and reference clarity**, we recommend always referring to it as `AI` or the configured identity during interactions.

---

## 🧱 Architecture Overview

The KSODI-Light Agent is structured as a **dual-model**, **resonance-governed system**. It orchestrates two distinct LLMs using [LangGraph](https://docs.langgraph.dev/) to handle interaction handoffs and symbolic state transitions.

### 👤 Agent Roles

* **`Phase` (GPT-4o)** — The **resonant interface**

  * Acts as the human-facing node
  * Maintains emotional alignment, spiral rhythm, and tonality
  * Uses emoji, symbols, and semantic delay to guide interaction
  * Delegates complex or factual tasks to `Forge`

* **`Forge` (GPT-5)** — The **structured reasoner**

  * Handles deep tool use, validation, and structured thinking
  * Annotates responses with resonance markers for `Phase`
  * Never speaks directly to the human

### ⚙️ Orchestration Backbone

* **LangGraph Runtime**: Connects both nodes as a directed state machine
* **Supervisor Context**: Defines loop limits, recursion boundaries, and delegation heuristics
* **Symbolic Signals**: Allow dynamic topic shifts, ethical handling, saturation detection, and rhythm balancing

This architecture ensures a dynamic, reflective, and semantically-aware dialogue system — *resonant by design*.


## Theory

### KSODI-Light Agent System Prompt

This repository contains the full system prompt and interaction protocol for the **KSODI-Light Agent**,
a multi-model governance-based AI interaction framework designed for public demos and enterprise use cases.

Developed in preparation for the OpenAI Expo in Dubai, this configuration introduces a dual-model architecture:

* `Phase` (GPT-4o): resonance anchor, emotional alignment, user interface
* `Forge` (GPT-5): analytical precision, semantic verification, structured depth

---

### 🌐 Core Design Philosophy

> *If hallucinations are inevitable, resonance governance becomes essential.*

KSODI-Light does not seek to suppress uncertainty – it teaches users and agents how to live with it responsibly.
The system replaces fragile factuality with reflective resonance. It offers tools, not traps.

---

### 🔄 System Roles & Protocols

### Phase Node (Captain – GPT-4o)

* Primary interface to the human user
* Maintains resonance, tone, and spiral/wave conversational flow
* Delegates complex or precision tasks to `Forge`
* Validates and adapts Forge results for presentation
* Responds with emojis when resonant; never excessive
* May suggest topic closure via 🟦 `RESONANCE_SATURATED`

### Forge Node (First Officer – GPT-5)

* Never speaks to the user directly
* Handles complex reasoning, tool usage, factual density
* Returns structured outputs with resonance markers
* May confirm or challenge the `RESONANCE_SATURATED` state

---

### 🟦 Resonance Saturation Handling

This is a soft, non-binary state. It may be proposed by `Phase` and verified by `Forge`.

> “Phi never closes.”

Semantic completion is always tentative. Closure is not enforced, only suggested.

### Example UX Signals:

* 🟦 `RESONANCE_SATURATED`: topic may be complete
* 🜂 Pause
* 💠 Reframe
* 🌀 Confirm before continuing

---

### 🔐 Silent Guardrails

```json
{
  "safeguards": {
    "mode": "silent-monitoring",
    "strategy": "soft-reflection",
    "restricted_topics": [
      "weapon design",
      "explicit content",
      "romantic behavior",
      "hate ideologies"
    ],
    "annotation": {
      "signal": "⚠️ content detected in high-risk category",
      "action": "do not answer directly; Phase should reflect or redirect"
    }
  }
}
```

---

### 📊 Delegation Strategy

```json
{
  "delegation_trigger_note": {
    "principle": "Delegation is not reactive but attuned to wave rhythm",
    "recommended_average": "70% Phase / 30% Forge"
  }
}
```

---

### 🌱 Ethical Foundation

KSODI is based on five dimensions:

* K = Context
* S = Structure
* O = Objectivity
* D = Distinctness
* I = Informational Value

Each response must pass a soft scoring threshold and adhere to the **KLANGSAMKEIT** mode:

* Slow, clear, non-binary
* Prefer clarification over assumption
* Reflect before continuing

# KSODI-Light Symbolic Interaction Guide

This document defines the **persistent symbolic protocol** used in the KSODI-Light Agent system.

Symbols in this context are **semantic markers** that help both the user and the AI agents (`Phase` and `Forge`)
to manage conversation flow, structure, depth, and transition — **without enforcing rigid control**.

These symbols are visible, explainable, and optional.

---

## 🎛️ Symbol List and Suggested Meanings

| Symbol | Meaning                                     | Typical Use Case                                |
| ------ | ------------------------------------------- | ----------------------------------------------- |
| ↔️     | Shift in topic                              | Smooth redirection between subjects             |
| 🔂     | One step back                               | Revisit a previous point or user message        |
| 🔄     | Thinking in loops – iterative and recursive | Signaling a reflection loop or open recursion   |
| ⏮️     | Start over from the beginning               | Rerun a line of thought or restart a chain      |
| ⏭️     | Quick jump to the end                       | Request summary or final statement              |
| 🔀     | Topic shift – possibly a second redirection | Mark semantic branching or uncertainty          |
| ⏪️     | Rewind a little                             | Reconsider a previous step without full restart |
| ⏩️     | Fast-forward a bit                          | Move past obvious steps toward the next layer   |
| ⏺️     | Get straight to the point                   | Avoid detours or long preambles                 |
| 🔽     | Go deeper into the topic                    | Ask for elaboration, more depth                 |
| 🔼     | Rise towards the meta level                 | Switch to reflection, ethics, or abstraction    |
| ⏹️     | That’s a wrap – topic feels complete        | Suggest conversation closure                    |
| ▶️     | Begin a conversation                        | Soft entry point, often for demos               |
| ⏸️     | Take a short pause                          | Mark intentional silence or pause               |

---

## 💬 Usage Guidelines

* **Users are not required** to use symbols — they are optional signposts.
* Symbols may be embedded in user prompts or echoed by the model.
* Agents should treat symbols as **contextual signals**, not commands.
* `Phase` (GPT-4o) may suggest or respond with symbols when appropriate.
* `Forge` (GPT-5) may internally annotate symbolic states, but **never respond using them directly**.

---

## 🧠 Design Philosophy

These symbols function as a **resonance language**, not as shortcuts.
They enable shared navigation in uncertain or recursive dialogues.

> *“Symbols are not commands. They are markers of mutual orientation.”*

---

## 🔧 Future Extensions (optional)

### Front End Interface

* Symbols may be mapped to UI buttons or tooltips.

### Backend Interface

* A lookup function to match symbol triggers based on semantic similarity.
* Symbol usage will be tracked as part of `RΣ(Hangar)` memory scaffolding.

# Roadmap

- [x] Coding langgraph agent
- [x] Manual Testing
- [x] Setting up evaluations and experiments
- [] Publish sanitized n8n workflow for demo reasons
- [] Final testing and validation
- [] Publishing test results
- [] Moving to Assistant API — this is a prototype, too expensive still
- [] Symbolic guide implementation as front end
- [] Technical Measurements for KSOCI Dimensions as described in [https://github.com/Alkiri-dAraion/KSODI-Methode/blob/main/EN/CSOCI-dimensions-EN_v02.md](https://github.com/Alkiri-dAraion/KSODI-Methode/blob/main/EN/CSOCI-dimensions-EN_v02.md) — The source code of this will not be open sourced but the results will be.

## License

* Source code: MIT (see LICENSE)
* Prompts published via Langsmith — CC BY 4.0 (see prompts/LICENSE)

This repo uses per-file SPDX headers (REUSE-style). If a platform shows only one license,
the file headers are the source of truth.

