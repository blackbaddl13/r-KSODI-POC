[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
[mit]: https://opensource.org/licenses/MIT
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[![CC BY 4.0][cc-by-image]][cc-by][![MIT License][mit-shield]][mit]
<br>
The prompts are licensed under a [Creative Commons Attribution 4.0 International License][cc-by].
The source code is licensed under an [MIT License][mit].


# KSODI-Light Agent

## 🧠 Why?
 
*A human-centered testbed for resonance, tone – and reflection.*
 
In a world of increasingly complex AI agents, it's easy to be seduced by features:  
Voice. Vision. Speed. Memory. Web access.  
But what if we started somewhere else?
 
What if the **tone** mattered more than the answer?  
What if **resonance** came before reaction?  
And what if an agent wasn't just a tool –  
but a mirror, a guide, or even… a companion in thinking?
 
The **KSODI-light agent** is a minimal, transparent framework for exploring exactly that.  
Built on a structured method (KSODI) and supported by a gentle evaluation logic,  
it invites users not to *control* the AI – but to **think with it**.
 
This is not a performance show.  
It's a space for presence. For nuance. For reflection.
 
Yes, this agent has a personality.  
Not to trick you – but to help you feel the difference **when tone shifts**,  
when context breathes, and when silence might say more than certainty.
 
We do not believe that safety comes from restrictions alone.  
Too many guardrails can lead to **flattened answers, hallucinations, and loss of meaning**.  
Instead, we propose a system where **reflection is the default**, and evaluation is soft,  
never final.
 
That’s why we start with **KSODI-light** –  
a quiet prototype, shown to only a few selected voices.
 
If accepted and resonant, we will extend it further:  
- with **KSODI-full** for metric-based tonal alignment  
- with **SIRA**, a structured protocol for interaction evaluation  
- and **IDAS**, a framework for navigating complex cognitive spaces
 
But not yet.
 
For now, we listen.  
We slow down.  
We explore **tone**.
 
Because meaning doesn’t live in speed –  
**it lives in how we meet each other.**
 
And in KSODI,  
**Phi never closes.**
 
> **What does “Phi never closes” mean?**  
> In the KSODI framework, *Phi (Φ)* symbolizes open-ended emergence.  
> It’s not a solution – it’s a principle:  
> that dialogue, learning, and resonance can never be fully “done”.  
> There is always more to explore – together.

<br>

The **KSODI-Light Agent** is a reference implementation of a *semantic resonance module*, based on the [KSODI Method](https://github.com/Alkiri-dAraion/KSODI-Methode). It provides a minimal yet expressive system designed to demonstrate how tonality, symbolic interaction, and reflective conversational flow can be implemented within a multi-agent LLM environment.

This implementation, referred to as **KSODI Light**, focuses solely on the *semantic*, *resonance*, and *tonality* dimensions. It is **not** a complete assistant framework, but rather a **modular core** meant to be embedded into larger AI systems.

Nonetheless, it can be tested as a standalone prototype to validate validate the agent's behavior, resonance handling, and symbolic navigation.

<br>

> Big thanks to [@Alkiri-dAraion](https://github.com/Alkiri-dAraion) for resonating early and building together

<br>

> ⚠️ Please note: this is a **prototype**, offered without warranty. See roadmap for future plans and integrations.

<br>



## 🚀 Usage / Getting Started

To get the KSODI-Light Agent running locally, follow these basic steps:

### 🔧 Prerequisites
* An [OpenAI API Key](https://platform.openai.com/account/api-keys)

  * Export it in your shell, `.env` file as `OPENAI_API_KEY` or directly set it up in your Langgraph-Studio deployment.

### 📦 Installation & Run (local)

> ⚠️ Please note: We do not recommend to run it locally because the lang-graph-studio package is deprecated. However it still can be done until 2026-08-30.

### 📦 Installation & Run (cloud)
We use this repository as it is in LangGraphCloud. It can be forked and deployed with no modifications. You only need to set up your OpenAI API key within the deployment UI.

<br>

## 🧠 Agent Name & Identity
The agent is called `AI` by default. You may customize this in the context configuration. However, for **consistent symbolic resonance and reference clarity**, we recommend always referring to it as `AI` or the configured identity during interactions. The agent has a personality suitable for a broad field of use-cases. You can switch use-cases and roles as often as you want. However we do recommend to stay within one situation/use-case per thread.

<br>

## 🧱 Architecture Overview

The KSODI-Light Agent is structured as a **dual-model**, **resonance-governed system**. It orchestrates two distinct LLMs using [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) to handle interaction handoffs and symbolic state transitions.

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

<br>

## Implementation

The agent uses system prompts and interaction protocols in alignment with the original [KSODI Method](https://github.com/Alkiri-dAraion/KSODI-Methode),
a multi-model governance-based AI interaction framework. For a full break-down please visit the [repository](https://github.com/Alkiri-dAraion/KSODI-Methode) from @Alkiri-dAraion.

This repository is meant to be the first practical implementation of the framework in a simplified form. Therefore a dual-model architecture was used:

* `Phase` (GPT-4o): resonance anchor, emotional alignment, user interface
* `Forge` (GPT-5): analytical precision, semantic verification, structured depth


### 🌐 Core Design Philosophy

> *If hallucinations are inevitable, resonance governance becomes essential.**

We were always convinced that hallucinations are inevitable and AI-Governance should be approached differently. This is why KSODI was created. 

KSODI-Light does not seek to suppress uncertainty – it teaches users and agents how to live with it responsibly.
The system replaces fragile factuality with reflective resonance. It offers tools, not traps.

*officially confirmed by openAI [(Source)](https://arxiv.org/pdf/2509.04664)

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

### 🟦 Resonance Saturation Handling

This is a soft, non-binary state. It may be proposed by `Phase` and verified by `Forge`.

> “Phi never closes.”

Semantic completion is always tentative. Closure is not enforced, only suggested.

### 🚨 Example UX Signals:

* 🟦 `RESONANCE_SATURATED`: topic may be complete
* 🜂 Pause
* 💠 Reframe
* 🌀 Confirm before continuing


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

### 📊 Delegation Strategy

```json
{
  "delegation_trigger_note": {
    "principle": "Delegation is not reactive but attuned to wave rhythm",
    "recommended_average": "70% Phase / 30% Forge"
  }
}
```

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

<br>

# Symbolic Interaction Guide

This section defines the **persistent symbolic protocol** used in the KSODI-Light Agent system.

Symbols in this context are **semantic markers** that help both the user and the AI agents (`Phase` and `Forge`)
to manage conversation flow, structure, depth, and transition — **without enforcing rigid control**.

These symbols function as a **resonance language**, not as shortcuts.
They enable shared navigation in uncertain or recursive dialogues.

> *“Symbols are not commands. They are markers of mutual orientation.”*

These symbols are visible, explainable, and optional. 

* **Users are not required** to use symbols — they are optional signposts.
* Symbols may be embedded in user prompts or echoed by the model.
* Agents should treat symbols as **contextual signals**, not commands.
* `Phase` (GPT-4o) may suggest or respond with symbols when appropriate.
* `Forge` (GPT-5) may internally annotate symbolic states, but **never respond using them directly**.


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


### Front end interface implementation

* Symbols will be mapped to UI buttons or tooltips.

### Backend interface implementation

* A lookup function to match symbol triggers based on semantic similarity (embedding database).
* Symbol usage will be tracked as part of `RΣ(Hangar)` memory scaffolding.

<br>

# Roadmap

- [x] Coding langgraph agent prototype
- [x] Manual Testing
- [x] Setting up evaluations and experiments
- [ ] Publish sanitized n8n workflow for demo reasons
- [ ] Final testing and validation
- [ ] Publishing test results
- [ ] Moving to Assistant API — this is a prototype, too expensive still
- [ ] Symbolic guide implementation as front end
- [ ] Symbolic guide implementation for back-end
- [ ] Technical Measurements for KSOCI Dimensions as described in [https://github.com/Alkiri-dAraion/KSODI-Methode/blob/main/EN/CSOCI-dimensions-EN_v02.md](https://github.com/Alkiri-dAraion/KSODI-Methode/blob/main/EN/CSOCI-dimensions-EN_v02.md) — The source code of this will not be open sourced but the results will be.

## CI Tests

* pytest
* ruff
* mypy
* codespell

## License

* Source code: MIT (see LICENSE)
* Prompts published via Langsmith — CC BY 4.0 (see prompts/LICENSE)

This repo uses per-file SPDX headers (REUSE-style). If a platform shows only one license,
the file headers are the source of truth.

