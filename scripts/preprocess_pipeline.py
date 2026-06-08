""" CLI runner for the SAR preprocessing pipeline.

Examples:
  python scripts/preprocess_pipeline.py --input_dir ./datasets/M4-SAR/raw_sar \
      --output_dir ./datasets/M4-SAR-preprocessed --max_items 100
  python scripts/preprocess_pipeline.py --config my_config.json --speckle_method bm3d
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import preprocessing as PP


def main():
    ap = argparse.ArgumentParser(description='SAR preprocessing pipeline')
    ap.add_argument('--config', type=str, default=None, help='JSON config path')
    ap.add_argument('--input_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--max_items', type=int, default=None)
    ap.add_argument('--speckle_method', type=str, default=None,
                    choices=PP.SPECKLE_METHODS)
    ap.add_argument('--histogram_mode', type=str, default=None,
                    choices=PP.HISTOGRAM_MODES)
    ap.add_argument('--optical_reference_dir', type=str, default=None)
    ap.add_argument('--image_size', type=int, default=None)
    args = ap.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = PP.default_config()

    io = cfg['io']
    if args.input_dir:
        io['input_dir'] = args.input_dir
    if args.output_dir:
        io['output_dir'] = args.output_dir
    if args.max_items is not None:
        io['max_items'] = args.max_items
    for s in cfg['pipeline']['steps']:
        if args.speckle_method and s['name'] == 'speckle_filter':
            s['params']['method'] = args.speckle_method
        if s['name'] == 'histogram_mapping':
            if args.histogram_mode:
                s['params']['mode'] = args.histogram_mode
            if args.optical_reference_dir:
                s['params']['optical_reference_dir'] = args.optical_reference_dir
        if args.image_size and s['name'] == 'resize_or_tile':
            s['params']['image_size'] = args.image_size

    for log, _ in PP.run_pipeline(cfg):
        pass
    print(log.splitlines()[-1])


if __name__ == '__main__':
    main()
