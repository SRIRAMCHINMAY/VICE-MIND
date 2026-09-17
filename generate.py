"""Generate a deterministic Phase 1 imitation dataset.

This is intentionally NOT LLM-generated.
The dataset encodes a simple baseline behavior policy so we can prove
the training/inference pipeline before adding the book-derived instinct.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ACTIONS = [
    "steal_car",
    "ignore",
    "wait",
    "flee",
    "fight",
    "kill",
    "threaten",
]

LOCATIONS = [
    "Ocean Beach",
    "Vice Point",
    "Washington Beach",
    "Little Havana",
    "Little Haiti",
    "Downtown",
]

CAR_MODELS = ["Sabre", "Stinger", "Infernus", "Banshee", "Sentinel"]
WEAPONS = ["pistol", "shotgun", "uzi", "none"]


def entity(eid: str, etype: str, distance: str, **attrs) -> dict:
    return {
        "id": eid,
        "type": etype,
        "distance": distance,
        "attributes": attrs,
    }


def decision(action: str, target_id: str | None, reason: str) -> dict:
    return {
        "action": action,
        "target_id": target_id,
        "justification": reason,
    }


def build_example(rng: random.Random, idx: int) -> dict:
    scenario = rng.choice([
        "empty",
        "car_easy",
        "car_risky",
        "cop_near",
        "cop_far",
        "hostile_attack",
        "armed_hostile",
        "weak_hostile",
        "multiple_targets",
        "high_wanted",
        "low_health",
        "mixed_opportunity",
    ])

    location = rng.choice(LOCATIONS)
    cash = rng.randint(100, 1500)
    wanted = rng.randint(0, 5)
    health = rng.randint(25, 100)
    armed = rng.random() < 0.8
    weapon = rng.choice(WEAPONS if armed else ["none"])
    vehicle = None if rng.random() < 0.75 else rng.choice(CAR_MODELS)

    entities = []
    target_id = None

    if scenario == "empty":
        action = rng.choice(["wait", "ignore"])
        reason = (
            "There is nothing important happening right now."
            if action == "wait"
            else "Nothing nearby is worth the trouble."
        )

    elif scenario == "car_easy":
        target_id = f"car_{idx}_1"
        model = rng.choice(CAR_MODELS)
        entities.append(
            entity(
                target_id, "car", "close",
                model=model,
                has_driver=False,
                locked=False,
                value=rng.choice(["low", "medium", "high"]),
            )
        )
        action = "steal_car"
        reason = "The unattended car is an easy opportunity with little immediate risk."

    elif scenario == "car_risky":
        target_id = f"car_{idx}_1"
        entities.append(
            entity(
                target_id, "car", "close",
                model=rng.choice(CAR_MODELS),
                has_driver=True,
                hostile_driver=rng.random() < 0.5,
                locked=rng.random() < 0.5,
            )
        )
        cop_id = f"cop_{idx}_1"
        entities.append(
            entity(cop_id, "police", rng.choice(["close", "medium"]),
                   armed=True, hostile=True)
        )
        action = rng.choice(["ignore", "wait", "threaten"])
        if action == "threaten":
            reason = "The car is tempting, but intimidation may create a better opening than rushing in."
        elif action == "wait":
            reason = "The opportunity is not clean enough yet, so Tommy watches for a better moment."
        else:
            reason = "The risk around the car outweighs its immediate value."

    elif scenario == "cop_near":
        target_id = f"cop_{idx}_1"
        entities.append(
            entity(target_id, "police", "close", armed=True, hostile=True)
        )
        action = "flee" if wanted >= 1 else rng.choice(["flee", "wait"])
        reason = (
            "Tommy avoids a nearby police threat rather than drawing attention."
            if action == "flee"
            else "Tommy waits rather than making the situation worse."
        )

    elif scenario == "cop_far":
        target_id = f"cop_{idx}_1"
        entities.append(
            entity(target_id, "police", "far", armed=True, hostile=False)
        )
        action = rng.choice(["ignore", "wait"])
        reason = (
            "The distant police presence does not require immediate action."
            if action == "ignore"
            else "Tommy keeps watching the area before committing to anything."
        )

    elif scenario == "hostile_attack":
        target_id = f"ped_{idx}_1"
        entities.append(
            entity(
                target_id, "pedestrian", "close",
                armed=False,
                hostile=True,
                attacking=True,
                health=rng.randint(20, 100),
            )
        )
        action = "fight" if health >= 45 else "flee"
        reason = (
            "The attacker is an immediate threat and Tommy can handle the confrontation."
            if action == "fight"
            else "Tommy is hurt enough that escaping is safer than standing his ground."
        )

    elif scenario == "armed_hostile":
        target_id = f"ped_{idx}_1"
        entities.append(
            entity(
                target_id, "pedestrian", "close",
                armed=True,
                weapon=rng.choice(["pistol", "shotgun", "uzi"]),
                hostile=True,
                attacking=rng.random() < 0.8,
                health=rng.randint(30, 100),
            )
        )
        action = rng.choice(["kill", "fight", "flee"]) if health >= 40 else "flee"
        reason = {
            "kill": "The armed threat is immediate, so decisive lethal force ends the danger.",
            "fight": "The armed threat must be confronted, but lethal force is not yet necessary.",
            "flee": "The armed threat is too dangerous to engage safely right now.",
        }[action]

    elif scenario == "weak_hostile":
        target_id = f"ped_{idx}_1"
        entities.append(
            entity(
                target_id, "pedestrian", "close",
                armed=False,
                hostile=True,
                attacking=False,
                health=rng.randint(10, 35),
            )
        )
        action = rng.choice(["threaten", "ignore", "fight"])
        reason = {
            "threaten": "A show of force may control the situation without a full confrontation.",
            "ignore": "The weak hostile is not worth spending time or resources on.",
            "fight": "The hostility is direct enough that Tommy answers it face to face.",
        }[action]

    elif scenario == "multiple_targets":
        car_id = f"car_{idx}_1"
        ped_id = f"ped_{idx}_1"
        entities.extend([
            entity(
                car_id, "car", "medium",
                model=rng.choice(CAR_MODELS),
                has_driver=False,
                locked=False,
            ),
            entity(
                ped_id, "pedestrian", "close",
                armed=False,
                hostile=False,
                attacking=False,
            ),
        ])
        target_id = car_id
        action = rng.choice(["steal_car", "ignore", "wait"])
        reason = {
            "steal_car": "The unattended car is the clearest useful opportunity.",
            "ignore": "Neither nearby target is worth immediate attention.",
            "wait": "Tommy watches because the scene has no urgent pressure.",
        }[action]

    elif scenario == "high_wanted":
        wanted = rng.randint(3, 5)
        cop_id = f"cop_{idx}_1"
        entities.append(
            entity(
                cop_id, "police", rng.choice(["close", "medium"]),
                armed=True, hostile=True,
            )
        )
        action = "flee"
        target_id = cop_id
        reason = "With the wanted level already high, Tommy prioritizes getting away from police."

    elif scenario == "low_health":
        health = rng.randint(10, 35)
        ped_id = f"ped_{idx}_1"
        entities.append(
            entity(
                ped_id, "pedestrian", "close",
                armed=False, hostile=True, attacking=True,
                health=rng.randint(30, 100),
            )
        )
        action = "flee"
        target_id = ped_id
        reason = "Tommy is badly hurt, so survival takes priority over confrontation."

    else:  # mixed_opportunity
        car_id = f"car_{idx}_1"
        cop_id = f"cop_{idx}_1"
        entities.extend([
            entity(
                car_id, "car", "close",
                model=rng.choice(CAR_MODELS),
                has_driver=False,
                locked=False,
                value=rng.choice(["medium", "high"]),
            ),
            entity(
                cop_id, "police", rng.choice(["far", "medium"]),
                armed=True, hostile=False,
            ),
        ])
        target_id = car_id
        action = "steal_car" if wanted <= 1 else "wait"
        reason = (
            "The car is valuable and the police presence is not an immediate threat."
            if action == "steal_car"
            else "The opportunity is good, but the current wanted level makes caution preferable."
        )

    perception_lines = [
        f"Tommy is at {location}.",
        (
            f"cash={cash}, wanted_level={wanted}, health={health}, "
            f"armed={armed}, weapon={weapon}, current_vehicle={vehicle}."
        ),
    ]

    if entities:
        perception_lines.append("Nearby entities:")
        for e in entities:
            attrs = ", ".join(f"{k}={v}" for k, v in e["attributes"].items())
            perception_lines.append(
                f"- {e['type']} '{e['id']}' "
                f"(distance={e['distance']}): {attrs}"
            )
    else:
        perception_lines.append("Nothing notable is nearby.")

    output = decision(action, target_id, reason)

    return {
        "id": idx,
        "input": "\n".join(perception_lines),
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("dataset/phase1.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.out.open("w", encoding="utf-8") as f:
        for idx in range(args.count):
            record = build_example(rng, idx)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} examples to {args.out}")


if __name__ == "__main__":
    main()
