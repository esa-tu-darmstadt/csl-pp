#!/usr/bin/env cs_python
"""Host program of the quickstart: writes x, runs compute(), and checks
that x has been doubled."""

import argparse
import json

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

with open(f"{args.name}/out.json", encoding="utf-8") as f:
    params = json.load(f)["params"]
N = int(params["N"])

x = np.arange(N, dtype=np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
symbol_x = runner.get_id("x")
runner.load()
runner.run()

runner.memcpy_h2d(symbol_x, x, 0, 0, 1, 1, N, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.launch("compute", nonblock=False)
result = np.zeros(N, dtype=np.float32)
runner.memcpy_d2h(result, symbol_x, 0, 0, 1, 1, N, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.stop()

np.testing.assert_allclose(result, 2 * x)
print("SUCCESS!")
