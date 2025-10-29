import pytest
from src.main import run_optimization_solver


def test_solver_infeasible_raises():
    # Very high predicted load, but no relays can be shed (2,3,4 already off)
    predicted = [99.0] * 12
    status = "R1=on,R2=off,R3=off,R4=off"
    with pytest.raises(RuntimeError):
        run_optimization_solver(predicted, status)
