#!/usr/bin/env python3
"""Stand-in for rpicam-still used by the unit tests (no camera needed).

Writes a minimal JPEG (SOI, SOF0 frame header with the requested --width/--height, EOI) to -o and a
metadata JSON to --metadata that echoes the requested manual controls, or plausible auto values.

Failure injection (environment variables):
  FAKE_RPICAM_FAIL=1       exit 1 without output
  FAKE_RPICAM_LUX=<x>      Lux value to report
  FAKE_RPICAM_NO_LUX=1     omit Lux from the metadata
  FAKE_RPICAM_NO_META=1    exit 0 but write no metadata file
  FAKE_RPICAM_BAD_META=1   exit 0 with a metadata file that is not JSON
  FAKE_RPICAM_NO_IMAGE=1   exit 0 but write no image
  FAKE_RPICAM_TRUNCATE=1   exit 0 with a JPEG missing its end-of-image marker (partial write)
  FAKE_RPICAM_SHUTTER=<us> report this ExposureTime instead of the requested one
  FAKE_RPICAM_CROP=<str>   report this ScalerCrop
  FAKE_RPICAM_SIZE=WxH     write a JPEG of this size instead of the requested one
  FAKE_RPICAM_IMAGE=<path> copy this file's bytes to -o instead of the minimal JPEG (lets tests feed
                           real, decodable JPEGs through pi_capture); FAKE_RPICAM_SIZE, if also set,
                           wins, and FAKE_RPICAM_TRUNCATE / FAKE_RPICAM_NO_IMAGE still apply
"""
import json
import os
import struct
import sys

args = sys.argv[1:]
env = os.environ.get
if "--version" in args:
    print("rpicam-apps build: fake")
    sys.exit(0)
if env("FAKE_RPICAM_FAIL") == "1":
    print("ERROR: *** no cameras available ***", file=sys.stderr)
    sys.exit(1)


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


md = {
    "ExposureTime": int(env("FAKE_RPICAM_SHUTTER") or opt("--shutter", 49995)),
    "AnalogueGain": float(opt("--gain", 2.6948)),
    "DigitalGain": 1.0,
    "ColourGains": [float(x) for x in opt("--awbgains", "1.2469,3.4177").split(",")],
    "LensPosition": float(opt("--lens-position", 1.6577)),
    "Lux": float(env("FAKE_RPICAM_LUX", "50.0")),
    "ScalerCrop": env("FAKE_RPICAM_CROP", "(0, 0)/4608x2592"),
    "FocusFoM": 300,
    "AfState": 2,
}
if env("FAKE_RPICAM_NO_LUX") == "1":
    del md["Lux"]
w, h = (int(v) for v in (env("FAKE_RPICAM_SIZE") or f"{opt('--width', 2304)}x{opt('--height', 1296)}").split("x"))
sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, h, w, 3) + bytes([1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1])
jpeg = b"\xff\xd8" + sof0 + b"\xff\xda\x00\x02" + " ".join(args).encode() + b"\xff\xd9"
if env("FAKE_RPICAM_IMAGE") and not env("FAKE_RPICAM_SIZE"):
    with open(env("FAKE_RPICAM_IMAGE"), "rb") as f:
        jpeg = f.read()
if env("FAKE_RPICAM_TRUNCATE") == "1":
    jpeg = jpeg[:-10]
if env("FAKE_RPICAM_NO_IMAGE") != "1":
    with open(opt("-o"), "wb") as f:
        f.write(jpeg)
if env("FAKE_RPICAM_NO_META") != "1":
    with open(opt("--metadata"), "w") as f:
        f.write("{not json" if env("FAKE_RPICAM_BAD_META") == "1" else json.dumps(md))
print("Still capture image received", file=sys.stderr)
