import numpy as np
from numpy import linalg as LA


def power_method(A_csr, x0, max_ite):
  prev_mu = 0
  nrm2_x = LA.norm(x0, 2)
  x = x0 / nrm2_x
  for i in range(max_ite):
    y = A_csr.dot(x)
    mu = np.dot(x, y)
    print(f"i = {i}, mu = {mu}, |prev_mu - mu| = {abs(mu - prev_mu)}")
    nrm2_x = LA.norm(y, 2)
    x = y / nrm2_x
    prev_mu = mu
  return x
