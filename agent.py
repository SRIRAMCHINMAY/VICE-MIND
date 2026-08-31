import json
import os
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from world import WorldState


ACTIONS = [
    "steal_car",
    "ignore",
    "wait",
    "flee",
    "fight",
    "kill",
    "threaten",
]

MODEL_DIR = os.environ.get(
    "TOMMY_MODEL_DIR",
    "./models/tommy-phase1",
)


class GraphState(TypedDict):
    world: WorldState
    perception: str
    decision: dict


def perceive(state: GraphState) -> dict:
    """Turn raw world state into text the decision model can read."""
    world = state["world"]

    lines = [
        f"Tommy is at {world.tommy.location}.",
        f"cash={world.tommy.cash}, wanted_level={world.tommy.wanted_level}, "
        f"health={world.tommy.health}, armed={world.tommy.armed}, "
        f"weapon={world.tommy.weapon}, current_vehicle={world.tommy.current_vehicle}.",
    ]

    if not world.nearby_entities:
        lines.append("Nothing notable is nearby.")
    else:
        lines.append("Nearby entities:")
        for entity in world.nearby_entities:
            attrs = ", ".join(
                f"{key}={value}" for key, value in entity.attributes.items()
            )
            if attrs:
                lines.append(
                    f"- {entity.type} '{entity.id}' "
                    f"(distance={entity.distance}): {attrs}"
                )
            else:
                lines.append(
                    f"- {entity.type} '{entity.id}' "
                    f"(distance={entity.distance})"
                )

    return {"perception": "\n".join(lines)}


_LOCAL_MODEL = None
_TOKENIZER = None


def _load_local_model():
    """Load the fine-tuned model once, on first decision."""
    global _LOCAL_MODEL, _TOKENIZER

    if _LOCAL_MODEL is not None:
        return _TOKENIZER, _LOCAL_MODEL

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Local Tommy model not found at '{MODEL_DIR}'. "
            "Run train.py first, then set TOMMY_MODEL_DIR if needed."
        )

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
    )
    _LOCAL_MODEL.eval()

    return _TOKENIZER, _LOCAL_MODEL


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from model output."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {text!r}")

    return json.loads(match.group(0))


def _validate_decision(decision: dict, world: WorldState) -> dict:
    """Keep malformed model output from corrupting the world."""
    action = decision.get("action")
    target_id = decision.get("target_id")
    justification = decision.get("justification", "")

    if action not in ACTIONS:
        action = "wait"

    if target_id is not None:
        valid_ids = {entity.id for entity in world.nearby_entities}
        if target_id not in valid_ids:
            target_id = None

    if not isinstance(justification, str) or not justification.strip():
        justification = "Tommy follows his immediate instinct."

    return {
        "action": action,
        "target_id": target_id,
        "justification": justification.strip(),
    }


def decide(state: GraphState) -> dict:
    """Use the locally fine-tuned Tommy model to choose an action."""
    tokenizer, model = _load_local_model()

    prompt = (
        "You are Tommy Vercetti making a decision inside a simulated Vice City.\n\n"
        "Current situation:\n"
        f"{state['perception']}\n\n"
        f"Available actions: {', '.join(ACTIONS)}\n\n"
        "Choose exactly one action. Choose a target_id only when relevant. "
        "Respond with ONLY valid JSON and no markdown:\n"
        '{"action":"<action>","target_id":"<entity id or null>",'
        '"justification":"<one short sentence>"}'
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]

    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    decision = _extract_json(raw)

    return {"decision": _validate_decision(decision, state["world"])}


def act(state: GraphState) -> dict:
    """Apply the decision to the simulated world."""
    world = state["world"]
    decision = state["decision"]

    action = decision["action"]
    target_id = decision.get("target_id")
    justification = decision.get("justification", "")

    if action == "steal_car" and target_id:
        target = next(
            (entity for entity in world.nearby_entities if entity.id == target_id),
            None,
        )
        if target and target.type == "car":
            world.tommy.current_vehicle = target.id
            world.nearby_entities = [
                entity
                for entity in world.nearby_entities
                if entity.id != target_id
            ]
            world.log(
                f"Tommy stole car '{target_id}'. Instinct: {justification}"
            )
        else:
            world.log(
                f"Tommy tried to steal '{target_id}', but it was unavailable. "
                f"Instinct: {justification}"
            )

    elif action == "flee":
        world.log(f"Tommy flees the scene. Instinct: {justification}")

    elif action == "fight":
        world.log(
            f"Tommy confronts '{target_id or 'the threat'}'. "
            f"Instinct: {justification}"
        )

    elif action == "kill":
        world.log(
            f"Tommy chooses lethal force against '{target_id or 'the threat'}'. "
            f"Instinct: {justification}"
        )

    elif action == "threaten":
        world.log(
            f"Tommy threatens '{target_id or 'the target'}'. "
            f"Instinct: {justification}"
        )

    elif action == "wait":
        world.log(f"Tommy waits and watches. Instinct: {justification}")

    else:
        world.log(f"Tommy ignores the situation. Instinct: {justification}")

    world.last_decision = decision
    world.tick += 1
    return {"world": world}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("perceive", perceive)
    graph.add_node("decide", decide)
    graph.add_node("act", act)
    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "decide")
    graph.add_edge("decide", "act")
    graph.add_edge("act", END)
    return graph.compile()