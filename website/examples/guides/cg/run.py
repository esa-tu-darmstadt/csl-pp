#!/usr/bin/env cs_python
"""Host program of the conjugate gradient tutorial: writes b, the initial
guess x and the stencil coefficients to the PEs, runs solve(), and compares
the result with the same algorithm run on the host."""

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
width = int(params["width"])
height = int(params["height"])
n = int(params["MAX_ZDIM"])
max_ite = 3

# The grid has height x width columns of n points, one column per PE. The
# matrix A is the 7-point stencil with the coefficients west, east, south,
# north, bottom, top, center on every point, which is symmetric and
# positive definite.
coeff = np.array([-1, -1, -1, -1, -1, -1, 6], dtype=np.float32)


def matvec(v):
    """A v for v of shape (height, width, n); rows are PE rows (y)."""
    y = coeff[6] * v
    y[:, 1:, :] += coeff[0] * v[:, :-1, :]  # west neighbor
    y[:, :-1, :] += coeff[1] * v[:, 1:, :]  # east neighbor
    y[:-1, :, :] += coeff[2] * v[1:, :, :]  # south neighbor
    y[1:, :, :] += coeff[3] * v[:-1, :, :]  # north neighbor
    y[:, :, 1:] += coeff[4] * v[:, :, :-1]  # bottom neighbor
    y[:, :, :-1] += coeff[5] * v[:, :, 1:]  # top neighbor
    return y


def conjugate_gradient(x, b, tol):
    """The algorithm of pe.csl, in the same order of operations."""
    x = x.copy()
    r = b - matvec(x)
    rho = np.float32(np.sum(r * r))
    k = 0
    rho_old = np.float32(0)
    while rho > tol * tol and k < max_ite:
        k += 1
        p = r.copy() if k == 1 else r + (rho / rho_old) * p
        w = matvec(p)
        eta = np.float32(np.sum(p * w))
        alpha = rho / eta
        x += alpha * p
        r -= alpha * w
        rho_old = rho
        rho = np.float32(np.sum(r * r))
    return x, rho, k


rng = np.random.default_rng(seed=0)
b = rng.random((height, width, n), dtype=np.float32)
x0 = np.zeros((height, width, n), dtype=np.float32)
tol = np.float32(1e-3 * np.linalg.norm(b))

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
symbols = {name: runner.get_id(name) for name in ("b", "x", "stencil_coeff", "rho")}
runner.load()
runner.run()


def write(name, values):
    elements = values.shape[2]
    runner.memcpy_h2d(symbols[name], values.ravel(), 0, 0, width, height, elements,
                      streaming=False, order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)


def read(name, elements):
    values = np.zeros(height * width * elements, dtype=np.float32)
    runner.memcpy_d2h(values, symbols[name], 0, 0, width, height, elements,
                      streaming=False, order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return values.reshape(height, width, elements)


write("b", b)
write("x", x0)
write("stencil_coeff", np.tile(coeff, (height, width, 1)))
runner.launch("solve", np.int16(n), np.float32(tol), np.int16(max_ite), nonblock=False)
x = read("x", n)
rho = read("rho", 1)[0, 0, 0]
runner.stop()

expected_x, expected_rho, k = conjugate_gradient(x0, b, tol)
print(f"{k} iterations, |r|^2 = {rho} on the device, {expected_rho} on the host")
np.testing.assert_allclose(x, expected_x, rtol=1e-4, atol=1e-5)
print("SUCCESS!")
