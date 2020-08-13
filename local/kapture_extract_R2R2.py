import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch

r2d2_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../thirdparty/r2d2/")
)
sys.path.append(r2d2_path)
from thirdparty.r2d2.extract_kapture import extract_kapture_keypoints

default_conf = {
    'model': 'models/r2d2_WASF_N8_big.pt',
    'top_k': 4000,  # number of kpts
    'scale_f': 2 ** 0.25,  # scale factor
    'min_size': 256,
    'max_size': 1024,
    'min_scale': 0,
    'max_scale': 1,
    'reliability_thr': 0.7,
    'repeatability_thr': 0.7,
}


def main(kapture_root, config):
    logging.info('Extracting R2D2 features with configuration:\n', config)
    config['gpu'] = [torch.cuda.is_available()]
    config['kapture_root'] = kapture_root
    config['model'] = r2d2_path + '/' + config['model']
    args = SimpleNamespace(**config)
    extract_kapture_keypoints(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kapture-root", type=str, required=True,
                        help='path to kapture root directory')
    args = parser.parse_args()
    main(args.kapture_root, default_conf)
