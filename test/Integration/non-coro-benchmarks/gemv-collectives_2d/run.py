#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Kernel rectangle and matrix dimensions
kernel_rows = int(compile_data['params']['kernel_rows'])
kernel_cols = int(compile_data['params']['kernel_cols'])
matrix_rows = int(compile_data['params']['matrix_rows'])
matrix_cols = int(compile_data['params']['matrix_cols'])

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(matrix_rows, matrix_cols).astype(np.float32)
X = np.random.rand(matrix_cols).astype(np.float32)
B = np.random.rand(matrix_rows).astype(np.float32)

y_expected = (A @ X) + B

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

symbol_A = runner.get_id("A")
symbol_x = runner.get_id("x")
symbol_b = runner.get_id("b")
symbol_y = runner.get_id("y")
symbol_time_buf_u16 = runner.get_id("time_buf_u16")

runner.load()
runner.run()

print("Copying data...")
per_pe_rows = matrix_rows // kernel_rows
per_pe_cols = matrix_cols // kernel_cols

data = np.stack(np.split(np.stack(np.split(A, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()

runner.memcpy_h2d(symbol_A, data, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)

runner.memcpy_h2d(symbol_x, X, 0, 0, 1, 1, matrix_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)
runner.memcpy_h2d(symbol_b, B, 0, 0, 1, 1, matrix_rows,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

print("Launching kernel...")
# Enable per-PE timestamp counter.
runner.launch("f_enable_timer", nonblock=False)
# Run the kernel.
runner.launch("main", nonblock=False)
# Marshal (start, end) into the export-visible buffer.
runner.launch("f_memcpy_timestamps", nonblock=False)

y = np.zeros(matrix_rows, dtype=np.float32)
runner.memcpy_d2h(y, symbol_y, kernel_cols-1, kernel_rows-1, 1, 1, matrix_rows,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)

# D2H the per-PE timestamps (6 u16 words: 3 start, 3 end).
time_buf_1d = np.zeros(kernel_rows * kernel_cols * 6, np.uint32)
runner.memcpy_d2h(time_buf_1d, symbol_time_buf_u16,
                  0, 0, kernel_cols, kernel_rows, 6,
                  streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                  order=memcpy_order, nonblock=False)

runner.stop()
print("Copied back result.")

print("y calculated: ", y)
print("y expected:   ", y_expected)
np.testing.assert_allclose(y, y_expected, atol=0.01, rtol=0)

# Reconstruct u48 timestamps from the 3 u16 words and compute total cycles.
def make_u48(words):
  return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)

time_hwl = time_buf_1d.astype(np.uint16).reshape(kernel_rows, kernel_cols, 6)
time_start = np.zeros((kernel_rows, kernel_cols), dtype=np.int64)
time_end   = np.zeros((kernel_rows, kernel_cols), dtype=np.int64)
for hi in range(kernel_rows):
  for wi in range(kernel_cols):
    time_start[hi, wi] = make_u48(time_hwl[hi, wi, 0:3])
    time_end[hi, wi]   = make_u48(time_hwl[hi, wi, 3:6])

cycles_send = int(time_end.max() - time_start.min())
time_send_us = cycles_send / 850.0
print(f"cycles_send = {cycles_send} cycles")
print(f"time_send   = {time_send_us:.3f} us")

print("SUCCESS")
