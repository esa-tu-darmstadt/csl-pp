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

# Create tensors for A, X, B.
# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(matrix_rows, matrix_cols).astype(np.float32)
X = np.random.rand(matrix_cols).astype(np.float32)
B = np.random.rand(matrix_rows).astype(np.float32)

# Compute expected result
y_expected = (A @ X) + B

# Specify path to ELF files, set up runner
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
# Compute number of rows and cols of A on each PE
per_pe_rows = matrix_rows // kernel_rows
per_pe_cols = matrix_cols // kernel_cols

# This transformation on A creates a flattened array so that the matrix can
# be mapped onto the PEs using MemcpyOrder.ROW_MAJOR copy ordering.
# The arrays holding A on each PE are 1D arrays that store the submatrices
# in row major order.

# As an example, consider A[4, 4], mapped onto a 2x2 grid of PEs:
#
#   Matrix A on host            2 x 2 PE grid, row major A submatrices
# +----+----+----+----+         +----------------+----------------+
# | 0  | 1  | 2  | 3  |         | PE (0, 0):     | PE (1, 0):     |
# +----+----+----+----+         |  0,  1,  4,  5 |  2,  3,  6,  7 |
# | 4  | 5  | 6  | 7  |         |                |                |
# +----+----+----+----+   --->  +----------------+----------------+
# | 8  | 9  | 10 | 11 |         | PE (0, 1):     | PE (1, 1):     |
# +----+----+----+----+         |  8,  9, 12, 13 | 10, 11, 14, 15 |
# | 12 | 13 | 14 | 15 |         |                |                |
# +----+----+----+----+         +----------------+----------------+
#
# MemcpyOrder.ROW_MAJOR copy ordering maps an input array to dimensions h x w x l,
# with l varying fastest, where:
#   - h is kernel_rows (i.e., height of program rectangle)
#   - w is kernel_cols (i.e., width of program rectangle)
#   - l is per_pe_rows * per_pe_cols (i.e., num elems copied to each PE)
#
# So our input array for memcpy_h2d must be ordered as follows:
# [ 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 ]

# The transformation here takes the matrix A and:
#   1. splits A into kernel_cols submatrices, along the vertical axis
#   2. stacks them into a 3D array, with kernel_col as 0th dimension
#   3. splits 3D array into kernel_rows subarrays, along the horizontal axis
#   4. stacks them into a 4D array, with kernel_row as 0th dimension
#   5. flattens into a 1D array
data = np.stack(np.split(np.stack(np.split(A, kernel_cols, axis=1)), kernel_rows, axis=1)).ravel()

# Copy flattened A array onto PEs
runner.memcpy_h2d(symbol_A, data, 0, 0, kernel_cols, kernel_rows, per_pe_rows * per_pe_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False,
                  order=memcpy_order)

# Place x and b on PE (0,0). They will be scattered with collective comms
runner.memcpy_h2d(symbol_x, X, 0, 0, 1, 1, matrix_cols,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)
runner.memcpy_h2d(symbol_b, B, 0, 0, 1, 1, matrix_rows,
                  streaming=False, data_type=memcpy_dtype, nonblock=False, order=memcpy_order)

print("Launching kernel...")
# Initialize fiber runtime (untimed).
runner.launch("f_init", nonblock=False)
# Enable per-PE timestamp counter.
runner.launch("f_enable_timer", nonblock=False)
# Run the kernel.
runner.launch("main", nonblock=False)
# Marshal (start, end) into the export-visible buffer.
runner.launch("f_memcpy_timestamps", nonblock=False)

# Collect the result y from PE (kernel_cols-1,kernel_rows-1) and compare to expected
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
