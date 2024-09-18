#!/usr/bin/env python
"""Demo script on using demosaicnet for inference."""

import os
from pkg_resources import resource_filename

import argparse
import numpy as np
import torch as th
import imageio.v2 as imageio

import demosaicnet



def main(args):
  # Load some ground-truth image
  print("INFO: Reading image...")
  mosaic = imageio.imread(args.input).astype(np.float32) / 255.0
  mosaic = np.array(mosaic)
  print("INFO: Done reading image.")

  

  # Network expects channel first
  print("INFO: Preparing image for network...")
  mosaic = np.transpose(mosaic, [2, 0, 1])
  print("INFO: Done preparing image for network.")
  
  
  # Run the model (expects batch as first dimension)
  print("INFO: Running the model...")
  mosaic = th.from_numpy(mosaic).unsqueeze(0)
  bayer = demosaicnet.BayerDemosaick()
  with th.no_grad():  # inference only
    out = bayer(mosaic).squeeze(0).cpu().numpy()
    out = np.clip(out, 0, 1)

  print("INFO: Done running the model.")


  # Write image
  print("INFO: Writing image...")
  out = (out*255.0).round().astype(np.uint8)
  imageio.imwrite(args.output, np.transpose(out, [1, 2, 0]))
  print("INFO: Done writing image.")


  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", help="output file (GRBG Color Mosaic)")
  parser.add_argument("--input", help="input file (demosaiced)")
  args = parser.parse_args()
  main(args)
  
