#!/usr/bin/env cs_python
"""Host program of the tutorial: writes a value to each of the two outer
PEs, runs compute(), and checks that the middle PE holds their sum."""

import argparse

import numpy as np

# pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the compile output directory")
parser.add_argument("--cmaddr", help="IP:port of a CS system")
args = parser.parse_args()

west = np.array([1.5], dtype=np.float32)
east = np.array([2.25], dtype=np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
symbol_x = runner.get_id("x")
runner.load()
runner.run()


def write(px, value):
    """Writes x of the PE in column px."""
    runner.memcpy_h2d(symbol_x, value, px, 0, 1, 1, 1, streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


write(0, west)
write(2, east)
runner.launch("compute", nonblock=False)
result = np.zeros(1, dtype=np.float32)
runner.memcpy_d2h(result, symbol_x, 1, 0, 1, 1, 1, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.stop()

print(f"{west[0]} + {east[0]} = {result[0]}")
np.testing.assert_allclose(result, west + east)
print("SUCCESS!")
