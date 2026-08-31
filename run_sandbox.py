"""Run one Phase 1 inference cycle with the trained local model."""

from agent import build_graph
from world import Entity, TommyState, WorldState


def scenario() -> WorldState:
    return WorldState(
        tommy=TommyState(
            location="Ocean Beach",
            cash=500,
            wanted_level=0,
            health=100,
            armed=True,
            weapon="pistol",
        ),
        nearby_entities=[
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
        ],
    )


def main() -> None:
    app = build_graph()
    world = scenario()

    print("=== Initial situation ===")
    for entity in world.nearby_entities:
        print(
            f"- {entity.type} {entity.id}: "
            f"distance={entity.distance}, attributes={entity.attributes}"
        )

    result = app.invoke(
        {
            "world": world,
            "perception": "",
            "decision": {},
        }
    )
    world = result["world"]

    print("\n=== Trained Tommy decision ===")
    print(world.last_decision)

    print("\n=== Event log ===")
    for line in world.event_log:
        print(line)

    print("\n=== Final Tommy state ===")
    print(world.tommy)


if __name__ == "__main__":
    main()
