#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" Conjugate Gradient solver (coroutine port).

  Single-RPC on-device CG: f_cg(n, tol, max_ite) runs the entire solver
  inside one async task body composed via @await.
"""

import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from cg import conjugateGradient
from cmd_parser import parse_args
from scipy.sparse.linalg import eigs
from util import csr_7_pt_stencil, hwl_2_oned_colmajor, oned_to_hwl_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)


def csl_compile_core(
    cslc: str,
    width: int,
    height: int,
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  if not use_precompile:
    args = []
    args.append(cslc)
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")


def timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl):
  time_start = np.zeros((height, width)).astype(int)
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h, w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h, w)] = make_u48(word)

  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width + height - 2) - (px + py))

  time_start = time_start - time_ref
  time_end = time_end - time_ref

  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")


def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  print(f"cslc = {cslc}")

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  height = args.m
  width = args.n
  pe_length = args.k
  zDim = args.zDim
  blockSize = args.blockSize
  max_ite = args.max_ite

  print(
      f"width={width}, height={height}, pe_length={pe_length}, zDim={zDim}, blockSize={blockSize}"
  )
  print(f"max_ite = {max_ite}")
  assert pe_length >= 2, "the maximum size of z must be greater than 1"
  assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

  np.random.seed(2)
  x = (np.arange(height * width * zDim).reshape(height, width, zDim).astype(np.float32) + 100)

  x_1d = hwl_2_oned_colmajor(height, width, zDim, x, np.float32)
  nrm2_x = np.linalg.norm(x_1d.ravel(), 2)
  x_1d = x_1d / nrm2_x
  x = x / nrm2_x

  b = (np.arange(height * width * pe_length).reshape(height, width, pe_length).astype(np.float32) +
       1)
  b_1d = hwl_2_oned_colmajor(height, width, pe_length, b, np.float32)

  stencil_coeff = np.zeros((height, width, 7), dtype=np.float32)
  for i in range(height):
    for j in range(width):
      stencil_coeff[(i, j, 0)] = -1
      stencil_coeff[(i, j, 1)] = -1
      stencil_coeff[(i, j, 2)] = -1
      stencil_coeff[(i, j, 3)] = -1
      stencil_coeff[(i, j, 4)] = -1
      stencil_coeff[(i, j, 5)] = -1
      stencil_coeff[(i, j, 6)] = 6

  fabric_offset_x = 1
  fabric_offset_y = 1
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  print("store ELFs and log files in the folder ", dirname)

  code_csl = "src/layout.csl"

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
      blockSize,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      channels,
      width_west_buf,
      width_east_buf,
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  A_csr = csr_7_pt_stencil(stencil_coeff, height, width, zDim)

  A_csc = A_csr.tocsc(copy=True)
  A_csc = A_csc.sorted_indices().astype(np.float32)
  assert np.linalg.norm(A_csr.indptr - A_csc.indptr, np.inf) == 0, "A must be symmetric"
  assert np.linalg.norm(A_csr.indices - A_csc.indices, np.inf) == 0, "A must be symmetric"
  assert np.linalg.norm(A_csr.data - A_csc.data, np.inf) == 0, "A must be symmetric"

  nrm_b = np.linalg.norm(b_1d.ravel(), 2)
  eps = 1.0e-3
  tol = eps * nrm_b
  print(f"|b| = {nrm_b}")
  print(f"max_ite = {max_ite}")
  print(f"eps = {eps}")
  print(f"tol = {tol}")

  xf_1d, rho, k = conjugateGradient(A_csr, x_1d, b_1d, max_ite, tol)
  print(f"[host] after CG, rho = {rho}, k = {k}")

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_b = simulator.get_id("b")
  symbol_x = simulator.get_id("x")
  symbol_rho = simulator.get_id("rho")
  symbol_stencil_coeff = simulator.get_id("stencil_coeff")
  symbol_time_buf_u16 = simulator.get_id("time_buf_u16")
  symbol_time_ref = simulator.get_id("time_ref")

  simulator.load()
  simulator.run()

  print("copy vector b and x0")
  simulator.memcpy_h2d(
      symbol_b,
      b_1d,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  simulator.memcpy_h2d(
      symbol_x,
      x_1d,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("copy 7 stencil coefficients")
  stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, np.float32)
  simulator.memcpy_h2d(
      symbol_stencil_coeff,
      stencil_coeff_1d,
      0,
      0,
      width,
      height,
      7,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("step 0: init")
  simulator.launch("f_init", nonblock=False)

  print("step 0: enable timer")
  simulator.launch("f_enable_timer", nonblock=False)

  print("step 1: sync all PEs")
  simulator.launch("f_sync", nonblock=False)

  print("step 2: copy reference clock from reduce module")
  simulator.launch("f_reference_timestamps", nonblock=False)


  print(f"step 4: Conjugate Gradient with max_ite={max_ite}, zDim={zDim}, tol={tol}")
  simulator.launch(
      "f_cg",
      np.int16(zDim),
      np.float32(tol),
      np.int16(max_ite),
      nonblock=False,
  )


  rho_wse = np.zeros(1, np.float32)
  simulator.memcpy_d2h(
      rho_wse,
      symbol_rho,
      0,
      0,
      1,
      1,
      1,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  rho = rho_wse[0]
  print(f"[CG] rho = |b-A*x|^2 = {rho}")

  print("step 6: prepare (time_start, time_end)")
  simulator.launch("f_memcpy_timestamps", nonblock=False)

  print("step 7: D2H (time_start, time_end)")
  time_memcpy_hwl_1d = np.zeros(height * width * 6, np.uint32)
  simulator.memcpy_d2h(
      time_memcpy_hwl_1d,
      symbol_time_buf_u16,
      0,
      0,
      width,
      height,
      6,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

  print("step 8: D2H reference clock")
  time_ref_1d = np.zeros(height * width * 3, np.uint32)
  simulator.memcpy_d2h(
      time_ref_1d,
      symbol_time_ref,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

  print("step 9: D2H x[zDim]")
  xf_wse_1d = np.zeros(height * width * zDim, np.float32)
  simulator.memcpy_d2h(
      xf_wse_1d,
      symbol_x,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )

  simulator.stop()

  if args.cmaddr is None:
    dst_log = Path(f"{dirname}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{dirname}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

  timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl)

  nrm2_xf = np.linalg.norm(xf_wse_1d.ravel(), 2)
  print(f"|xf|_2 = {nrm2_xf}")

  z = xf_1d.ravel() - xf_wse_1d.ravel()
  nrm_z = np.linalg.norm(z, np.inf)
  print(f"|xf_ref - xf_wse| = {nrm_z}")
  np.testing.assert_allclose(xf_1d.ravel(), xf_wse_1d.ravel(), 1.0e-5)
  print("\nSUCCESS!")

  vals, _ = eigs(A_csr, k=1, which="SM")
  min_eig = abs(vals[0])
  vals, _ = eigs(A_csr, k=1, which="LM")
  max_eig = abs(vals[0])
  print(f"min(eig) = {min_eig}")
  print(f"max(eig) = {max_eig}")
  print(f"cond(A) = {max_eig/min_eig}")


if __name__ == "__main__":
  main()
