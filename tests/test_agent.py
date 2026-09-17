from agent import _extract_json, _validate_decision, act, perceive
from world import Entity, WorldState


def test_perceive_includes_tommy_and_nearby_entity_details():
    world = WorldState(
        nearby_entities=[
            Entity(
                id="car_1",
                type="car",
                distance="close",
                attributes={"has_driver": False},
            )
        ]
    )

    result = perceive({"world": world, "perception": "", "decision": {}})

    assert "Tommy is at Ocean Beach." in result["perception"]
    assert "car 'car_1'" in result["perception"]
    assert "has_driver=False" in result["perception"]


def test_validate_decision_falls_back_for_invalid_values():
    world = WorldState()

    result = _validate_decision(
        {
            "action": "invented_action",
            "target_id": "missing",
            "justification": "",
        },
        world,
    )

    assert result == {
        "action": "wait",
        "target_id": None,
        "justification": "Tommy follows his immediate instinct.",
    }


def test_extract_json_accepts_json_embedded_in_model_text():
    result = _extract_json(
        'Here is the decision: {"action":"wait","target_id":null,'
        '"justification":"Watch."}'
    )

    assert result["action"] == "wait"
    assert result["target_id"] is None


def test_act_steal_car_updates_vehicle_and_removes_target():
    world = WorldState(
        nearby_entities=[
            Entity(
                id="car_1",
                type="car",
                distance="close",
                attributes={"has_driver": False, "locked": False},
            )
        ]
    )

    result = act(
        {
            "world": world,
            "perception": "",
            "decision": {
                "action": "steal_car",
                "target_id": "car_1",
                "justification": "It is unattended.",
            },
        }
    )

    assert result["world"].tommy.current_vehicle == "car_1"
    assert result["world"].nearby_entities == []
    assert result["world"].tick == 1
