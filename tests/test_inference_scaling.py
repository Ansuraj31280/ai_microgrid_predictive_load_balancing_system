import asyncio
import numpy as np
from src.main import load_models, run_inference


def test_run_inference_basic():
    asyncio.get_event_loop().run_until_complete(load_models())
    seq = [float(100 + i) for i in range(12)]
    preds = run_inference(seq)
    assert isinstance(preds, list)
    assert len(preds) == 12
    assert all(isinstance(x, (float, int)) for x in preds)
