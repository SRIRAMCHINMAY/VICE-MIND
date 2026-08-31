# VICE-MIND

### Autonomous Tommy Vercetti — Instinct-Driven Character AI

VICE-MIND is an experimental AI agent designed to make **Tommy Vercetti autonomously perceive, decide, and act inside Grand Theft Auto: Vice City**.

The long-term goal is not to make Tommy simply follow a scripted set of commands, but to build a character whose actions emerge from a learned **instinct and decision-making model**.

The project is being developed incrementally, starting with a controlled simulation before connecting the agent to the actual game.

---

## Current Status

**Phase 1 — Complete**

We have successfully trained and integrated a local language model into an autonomous `perceive → decide → act` loop.

The current system:

- Uses **Qwen2.5-0.5B-Instruct** as the base language model
- Uses **LoRA** for parameter-efficient fine-tuning
- Trains entirely **locally on a GTX 1650**
- Uses a programmatically generated dataset of **5,000 simulated situations**
- Produces structured decisions in JSON
- Runs inference locally without OpenRouter or an external LLM API
- Includes a simulated Vice City world for testing
- Supports multiple possible Tommy actions

The first trained model has successfully completed training and has made decisions inside the sandbox.

---

# Architecture

The core architecture is:

```text
                  WORLD STATE
                       │
                       ▼
                  perceive()
                       │
                       ▼
              Situation Description
                       │
                       ▼
              ┌─────────────────┐
              │   TOMMY MODEL   │
              │                 │
              │ Qwen + LoRA     │
              └────────┬────────┘
                       │
                       ▼
                    decide()
                       │
                       ▼
             Structured JSON Decision
                       │
                       ▼
                     act()
                       │
                       ▼
                 WORLD CHANGES
```

The agent is intentionally separated into three responsibilities:

### `perceive()`

Converts the current world state into a textual description of what Tommy can perceive.

### `decide()`

The only component containing the learned decision-making model.

The model receives the current situation and chooses an action.

### `act()`

Applies the decision to the simulated world.

This separation is important because it allows the decision model to eventually be swapped into the real game without redesigning the entire agent.

---

# Phase 1 — Learning a Baseline Tommy

Phase 1 is intentionally **not yet based on Machiavelli's *The Prince***.

Instead, it establishes the complete technical pipeline first.

The goal is:

> Train a local model to map a situation to a Tommy-like action, then use that trained model inside the autonomous agent loop.

The initial action space is:

```text
steal_car
ignore
wait
flee
fight
kill
threaten
```

The model produces decisions in the following format:

```json
{
  "action": "steal_car",
  "target_id": "car_2",
  "justification": "The unattended car is an easy opportunity."
}
```

---

# Training Data

Phase 1 uses a **programmatically generated dataset** rather than an external LLM teacher.

`generate_dataset.py` creates randomized Vice City situations involving:

- cars
- pedestrians
- police
- weapons
- hostility
- wanted level
- health
- distance
- risk
- opportunities

Each situation is paired with a baseline expected decision.

Example:

```text
Tommy is at Ocean Beach.
wanted_level=0
health=100

Nearby:
- car_1: occupied
- car_2: unattended
```

Target output:

```json
{
  "action": "steal_car",
  "target_id": "car_2",
  "justification": "The unattended car is an easy opportunity with little immediate risk."
}
```

The resulting dataset is stored as JSONL.

---

# Model

## Qwen

The base model is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Qwen is a pretrained language model. It already understands natural language and instruction-style prompts.

We use Hugging Face to download the model, but the actual training and inference are performed locally.

---

# LoRA

Instead of modifying every parameter in Qwen, Phase 1 uses **LoRA (Low-Rank Adaptation)**.

The model contains approximately:

```text
498 million total parameters
```

while only approximately:

```text
4.4 million parameters
```

are trainable through LoRA.

This allows the model to be fine-tuned on consumer hardware with limited VRAM.

The current training configuration uses **FP16 (half precision)**.

The model is therefore:

```text
Qwen 0.5B
    +
LoRA
    +
FP16 training
```

The resulting model is saved locally and used during inference.

---

# Hardware

Phase 1 was trained on:

```text
GPU: NVIDIA GeForce GTX 1650
VRAM: 4 GB
```

The training successfully ran on the GPU, reaching approximately 99% GPU utilization.

Training the 5,000-example dataset for 3 epochs took approximately:

```text
6 hours 55 minutes
```

This gives us a working baseline that can later be optimized with techniques such as QLoRA/4-bit quantization if necessary.

---

# Runtime

After training, the model is loaded locally.

There is **no OpenRouter or Claude API call during inference**.

The runtime flow is:

```text
WorldState
    ↓
perceive()
    ↓
text description
    ↓
local Tommy model
    ↓
JSON decision
    ↓
act()
```

The model can therefore make decisions without an internet connection once the required model files have been downloaded.

---

# Current World Simulation

`world.py` provides a lightweight simulated environment.

It contains:

### Tommy state

```text
location
cash
wanted level
health
weapon
armed state
current vehicle
```

### Nearby entities

Entities currently support things such as:

```text
cars
pedestrians
police
```

with attributes such as:

```text
distance
health
weapon
hostility
whether a vehicle is occupied
whether a vehicle is locked
```

This simulated representation is intentionally designed as an abstraction layer.

Later, the same `WorldState` structure can be populated from the actual Vice City game state.

---

# Testing

`run_sandbox.py` runs a complete autonomous cycle against a predefined situation.

For example:

```text
Initial World
      ↓
perceive
      ↓
Tommy model
      ↓
steal_car
      ↓
act
      ↓
Tommy enters the selected vehicle
```

`test_scenarios.py` runs multiple different situations to check whether the model can generalize beyond the exact examples it was trained on.

Example scenarios include:

```text
Easy car opportunity
Police nearby
Nothing happening
Hostile attacker
Armed threat
Low health
Valuable car + police
Occupied car + police
```

These tests are intended as qualitative checks rather than a formal benchmark.

---

# Project Structure

```text
VICE_MIND/
│
├── agent.py
├── world.py
├── generate.py
├── train.py
├── run_sandbox.py
├── test_scenarios.py
├── requirements.txt
│
├── dataset/
│   └── phase1.jsonl
│
└── models/
    └── tommy-phase1/
```

The trained model files are intentionally excluded from Git because they are large binary artifacts.

---

# Environment

The project is developed using a Mamba environment.

Example:

```bash
mamba create -n Tommy-sandbox python=3.11
mamba activate Tommy-sandbox
```

Python dependencies are installed through `pip` inside the environment.

This allows Mamba to manage the Python environment while PyTorch, Transformers, PEFT, LangGraph and related packages are installed through Python's package ecosystem.

---

# Phase 2 — Machiavelli

The current model is only a **baseline decision model**.

The next major stage is to extract a set of decision principles from Niccolò Machiavelli's *The Prince*.

The goal is not to simply give the model the book and ask it to quote from it.

Instead, we want to extract an underlying structure resembling:

```text
situation
    ↓
threat / opportunity
    ↓
risk assessment
    ↓
power / control considerations
    ↓
consequences
    ↓
Tommy's instinctive stance
    ↓
action
```

This becomes the basis for new training data.

---

# Future Training Strategy

The planned progression is:

```text
Qwen base model
       ↓
Phase 1 fine-tuning
       ↓
Tommy baseline model
       ↓
Machiavelli principles
       ↓
new decision dataset
       ↓
fine-tune again
       ↓
Machiavellian Tommy model
```

The Phase 1 model will be preserved so that later versions can be compared against it.

---

# Future Game Integration

The final system is intended to connect to **reVC (reverse-engineered Vice City)**.

The planned architecture is:

```text
                   VICE CITY
                       │
                       ▼
              Game State Reader
                       │
                       ▼
                   WorldState
                       │
                       ▼
                  perceive()
                       │
                       ▼
                Tommy Model
                       │
                       ▼
                    decide()
                       │
                       ▼
                High-Level Action
                       │
                       ▼
                 Action Layer
                       │
                       ▼
                   VICE CITY
```

The game integration will therefore consist of two major bridges:

### Game → AI

Read the actual game state and convert it into the project's `WorldState`.

### AI → Game

Convert Tommy's high-level decision into actual game actions.

For example:

```text
Model:
steal_car
    ↓
Action layer:
move to vehicle
enter vehicle
take control
drive away
```

The model therefore focuses on **what Tommy wants to do**, while the action layer handles **how that intention is executed in the game**.

---

# Long-Term Goal

The eventual objective is an autonomous Tommy who does not simply follow a predefined script.

Instead:

```text
observe
   ↓
interpret
   ↓
evaluate
   ↓
choose
   ↓
act
   ↓
observe consequences
   ↓
repeat
```

The ambition is for Tommy's behavior to emerge from a learned instinct model rather than from manually scripted responses.

Phase 1 establishes the foundation for that system.

---

# Current Milestone

Phase 1 has demonstrated that:

- A small local language model can be fine-tuned on consumer hardware.
- LoRA can specialize the model without training all of its parameters.
- The trained model can run locally.
- The model can produce structured decisions.
- The decision model can be integrated directly into a `perceive → decide → act` agent loop.

**Next milestone: extract the instinct model from *The Prince* and use it to create a richer Tommy training dataset.**