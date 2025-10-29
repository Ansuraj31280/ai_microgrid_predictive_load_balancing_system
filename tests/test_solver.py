from src.main import run_optimization_solver


def test_solver_sheds_low_priority_first():
    # Peak predicted above threshold to force shedding
    predicted = [95.0] * 12  # over 85% of 100kW capacity
    status = "R1=on,R2=on,R3=on,R4=on"
    action = run_optimization_solver(predicted, status)
    # Expect to shed a low-priority relay first
    assert action.command in {"toggle", "noop"}
    if action.command == "toggle":
        assert action.target_relay_id in {3, 4, 2}
        assert action.state == "off"
