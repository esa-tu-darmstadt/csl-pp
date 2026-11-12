#!/usr/bin/env cs_python
"""Host program of the resources example: runs compute(). The PE program checks
its results with @assert, and a failed assertion stops the simulation."""

import argparse

# pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the compile output directory")
parser.add_argument("--cmaddr", help="IP:port of a CS system")
args = parser.parse_args()

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
runner.load()
runner.run()
runner.launch("compute", nonblock=False)
runner.stop()

print("SUCCESS!")
