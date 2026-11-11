#!/usr/bin/env cs_python
"""Host program of the resources example: runs compute() and checks `result`."""

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

expected = np.array([1, 1, 1, 1], dtype=np.uint32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
result_symbol = runner.get_id("result")
runner.load()
runner.run()
runner.launch("compute", nonblock=False)
result = np.zeros(len(expected), dtype=np.uint32)
runner.memcpy_d2h(result, result_symbol, 0, 0, 1, 1, len(expected),
                  streaming=False, order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.stop()

np.testing.assert_array_equal(result, expected)
print("SUCCESS!")
