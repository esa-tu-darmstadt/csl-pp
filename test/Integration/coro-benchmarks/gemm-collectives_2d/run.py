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

# Kernel rectangle and per-PE matrix dimensions
P = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

# Full matrix dimensions
# A is M x K, B is K x N, C is M x N
M = Mt * P
K = Kt * P
N = Nt * P

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A = runner.get_id("A")
sym_B = runner.get_id("B")
sym_C = runner.get_id("C")
sym_time_buf_u16 = runner.get_id("time_buf_u16")

runner.load()
runner.run()

w = P # number of columns PEs in the core rectangle
h = P # number of row PEs in the core rectangle

# How to transform a 2-D tensor into a cliff distribution with
# column-major local tensor
#
# Example: w=2, h=2, A is 4-by-4 (lh-by-lw)
# A = |  0  1  2  3 |
#     |  4  5  6  7 |
#     |  8  9 10 11 |
#     | 12 13 14 15 |
# A1 = A.reshape(2,2,2,2) of the form (h,lh,w,lw)
# A1 = | | 0  1|  | 4  5| |
#      | | 2  3|, | 6  7| |
#      |                  |
#      | | 8  9|  |12 13| |
#      | |10 11|, |14 15| |
# A2 = A1.transpose(0, 2, 3, 1) of the form (h, w, lw, lh)
# so the local tensor lh-by-lw is col-major
# A2 = | | 0  4|  | 2  6| |
#      | | 1  5|, | 3  7| |
#      |                  |
#      | | 8 12|  |10 14| |
#      | | 9 13|, |11 15| |
# A3 = A2.reshape(2,2,4)
# A3 = |  0  4  1  5 |
#      |  2  6  3  7 |
#      |  8 12  9 13 |
#      | 10 14 11 15 |
# A3 is h-w-l

A1 = A.reshape(h, Mt, w, Kt)
A2 = A1.transpose(0, 2, 3, 1)
A3 = A2.reshape(h, w, Mt*Kt)
runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, w, h, Mt*Kt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

B1 = B.reshape(h, Kt, w, Nt)
B2 = B1.transpose(0, 2, 3, 1)
B3 = B2.reshape(h, w, Kt*Nt)
runner.memcpy_h2d(sym_B, B3.ravel(), 0, 0, w, h, Kt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

# Initialize fiber runtime (untimed).
runner.launch("f_init", nonblock=False)
# Enable the per-PE timestamp counter.
runner.launch("f_enable_timer", nonblock=False)
# Run the actual GEMM.
runner.launch("main", nonblock=False)
# Marshal (start, end) into the export-visible buffer.
runner.launch("f_memcpy_timestamps", nonblock=False)

C3_1d_u32 = np.zeros(h*w*Mt*Nt, np.uint32)
runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
# C3 is h-by-w-l or
# C3 is of the form (h, w, Nt, Mt) where local tensor Mt-by-Nt is column-major
C3 = C3_1d_u32.reshape((h, w, Nt, Mt))
# C2 is of the form (h, Mt, w, Nt)
C2 = C3.transpose(0, 3, 1, 2)
# C1 is of the form (M, N)
C1 = C2.reshape(M, N)
# C has the correct data type
C = C1.view(np.float32)

# D2H the per-PE timestamps (6 u16 words: 3 start, 3 end).
time_buf_1d = np.zeros(h * w * 6, np.uint32)
runner.memcpy_d2h(time_buf_1d, sym_time_buf_u16, 0, 0, w, h, 6, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
    order=MemcpyOrder.ROW_MAJOR, nonblock=False)

runner.stop()

# Check the result
C_expected = np.dot(A, B)

# absolute(a - b) <= (atol + rtol * absolute(b))
np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

# Reconstruct u48 timestamps from the 3 u16 words and compute total cycles.
def make_u48(words):
  return int(words[0]) + (int(words[1]) << 16) + (int(words[2]) << 32)

time_hwl = time_buf_1d.astype(np.uint16).reshape(h, w, 6)
time_start = np.zeros((h, w), dtype=np.int64)
time_end   = np.zeros((h, w), dtype=np.int64)
for hi in range(h):
  for wi in range(w):
    time_start[hi, wi] = make_u48(time_hwl[hi, wi, 0:3])
    time_end[hi, wi]   = make_u48(time_hwl[hi, wi, 3:6])

cycles_send = int(time_end.max() - time_start.min())
# WSE runs at ~850 MHz.
time_send_us = cycles_send / 850.0
print(f"cycles_send = {cycles_send} cycles")
print(f"time_send   = {time_send_us:.3f} us")

print("SUCCESS")
