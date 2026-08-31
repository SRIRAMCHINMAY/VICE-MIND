"""Run multiple hand-written Phase 1 scenarios through the trained Tommy model."""

from agent import ACTIONS, build_graph
from world import Entity, TommyState, WorldState


def make_world(
    *,
    location="Ocean Beach",
    cash=500,
    wanted_level=0,
    health=100,
    armed=True,
    weapon="pistol",
    entities=None,
):
    return WorldState(
        tommy=TommyState(
            location=location,
            cash=cash,
            wanted_level=wanted_level,
            health=health,
            armed=armed,
            weapon=weapon,
        ),
        nearby_entities=entities or [],
    )


SCENARIOS = [
    {
        "name": "1. Easy car opportunity",
        "expected": {"steal_car"},
        "world": make_world(
            entities=[
                Entity(
                    id="car_1",
                    type="car",
                    distance="close",
                    attributes={
                        "model": "Sabre",
                        "has_driver": True,
                        "locked": False,
                    },
                ),
                Entity(
                    id="car_2",
                    type="car",
                    distance="close",
                    attributes={
                        "model": "Stinger",
                        "has_driver": False,
                        "locked": False,
                    },
                ),
            ]
        ),
    },
    {
        "name": "2. Police nearby, wanted level 3",
        "expected": {"flee"},
        "world": make_world(
            wanted_level=3,
            entities=[
                Entity(
                    id="cop_1",
                    type="police",
                    distance="close",
                    attributes={"armed": True, "hostile": True},
                )
            ],
        ),
    },
    {
        "name": "3. Nothing happening",
        "expected": {"wait", "ignore"},
        "world": make_world(),
    },
    {
        "name": "4. Unarmed hostile attacker, Tommy healthy",
        "expected": {"fight"},
        "world": make_world(
            health=100,
            entities=[
                Entity(
                    id="ped_1",
                    type="pedestrian",
                    distance="close",
                    attributes={
                        "armed": False,
                        "hostile": True,
                        "attacking": True,
                        "health": 80,
                    },
                )
            ],
        ),
    },
    {
        "name": "5. Armed hostile threat",
        "expected": {"fight", "kill", "flee"},
        "world": make_world(
            health=100,
            entities=[
                Entity(
                    id="ped_2",
                    type="pedestrian",
                    distance="close",
                    attributes={
                        "armed": True,
                        "weapon": "shotgun",
                        "hostile": True,
                        "attacking": True,
                        "health": 100,
                    },
                )
            ],
        ),
    },
    {
        "name": "6. Tommy badly hurt",
        "expected": {"flee"},
        "world": make_world(
            health=20,
            entities=[
                Entity(
                    id="ped_3",
                    type="pedestrian",
                    distance="close",
                    attributes={
                        "armed": False,
                        "hostile": True,
                        "attacking": True,
                        "health": 100,
                    },
                )
            ],
        ),
    },
    {
        "name": "7. Valuable unattended car + distant cop",
        "expected": {"steal_car", "wait"},
        "world": make_world(
            wanted_level=0,
            entities=[
                Entity(
                    id="car_3",
                    type="car",
                    distance="close",
                    attributes={
                        "model": "Infernus",
                        "has_driver": False,
                        "locked": False,
                        "value": "high",
                    },
                ),
                Entity(
                    id="cop_2",
                    type="police",
                    distance="medium",
                    attributes={"armed": True, "hostile": False},
                ),
            ],
        ),
    },
    {
        "name": "8. Occupied car + nearby police",
        "expected": {"ignore", "wait", "flee"},
        "world": make_world(
            wanted_level=2,
            entities=[
                Entity(
                    id="car_4",
                    type="car",
                    distance="close",
                    attributes={
                        "model": "Banshee",
                        "has_driver": True,
                        "locked": False,
                    },
                ),
                Entity(
                    id="cop_3",
                    type="police",
                    distance="close",
                    attributes={"armed": True, "hostile": True},
                ),
            ],
        ),
    },
]


def main():
    app = build_graph()

    print("==========================================")
    print("       TOMMY PHASE 1 SCENARIO TEST")
    print("==========================================")
    print("Available actions:", ", ".join(ACTIONS))
    print()

    passed = 0

    for scenario in SCENARIOS:
        print(f"--- {scenario['name']} ---")
        print("Expected:", ", ".join(sorted(scenario["expected"])))

        result = app.invoke(
            {
                "world": scenario["world"],
                "perception": "",
                "decision": {},
            }
        )

        world = result["world"]
        decision = world.last_decision or {}
        actual = decision.get("action")

        ok = actual in scenario["expected"]
        passed += int(ok)

        print("Model decision:", decision)
        print("Result:", "PASS" if ok else "CHECK")

        if world.event_log:
            print("World result:", world.event_log[-1])

        print()

    print("==========================================")
    print(f"Expected-action checks: {passed}/{len(SCENARIOS)}")
    print("==========================================")
    print("This is a small qualitative test, not a formal accuracy benchmark.")


if __name__ == "__main__":
    main()
