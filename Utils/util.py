import asyncio
import numpy as np
import time
from typing import *


def run(*awaitables: Awaitable, timeout = None):
    loop = asyncio.get_event_loop_policy().get_event_loop()
    if len(awaitables) == 1:
        future = awaitables[0]
    else:
        future = asyncio.gather(*awaitables)
    if timeout:
        future = asyncio.wait_for(future, timeout)
    task = asyncio.ensure_future(future)
    result = loop.run_until_complete(task)
    return result

def round(x: float, stepSize: float = 1):
    assert stepSize > 0
    multiplier = x / stepSize
    sign = np.sign(multiplier)
    value = np.abs(multiplier)
    value = (np.ceil(value) if value % 1 >= 0.5 else np.floor(value))
    return sign * np.round(value * stepSize, decimalDigits(stepSize))

def decimalDigits(x: float):
    decimalIndex = str(x).find('.')
    if decimalIndex == -1:
        return 0
    return len(str(x)[decimalIndex + 1:])

def timer(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time