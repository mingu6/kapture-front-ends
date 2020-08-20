import argparse
import logging
import os
import os.path as path
from importlib import import_module

import torch


feature_dirs = {
    'r2d2': 'r2d2',
}


def main(kapture_root, output_dir, feature_name, config, overwrite=False):
    # update config with checkpoint and GPU information
    extractor_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f'../thirdparty/{feature_dirs[feature_name]}/')
    )
    config['checkpoint'] = extractor_path + '/' + config['checkpoint']
    # import feature extractor
    extract_kapture_keypoints = getattr(import_module(f'local.kapture_extract_{feature_name}'), 'extract_kapture_keypoints')
    # iterate over submaps and process in sequence
    submap_names = [f.name for f in os.scandir(kapture_root) if f.is_dir()]
    for submap_name in submap_names:
        logging.info(f'Extracting {feature_name} features for submap {submap_name} with configuration:\n{config}')
        submap_root = path.join(kapture_root, submap_name)
        submap_out = path.join(output_dir, submap_name)
        split_names = [f.name for f in os.scandir(submap_root) if f.is_dir()]
        for split in split_names:
            extract_kapture_keypoints(path.join(submap_root, split), config, output_dir=path.join(submap_out, split), overwrite=overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kapture-root", type=str, required=True,
                        help='path to root directory containing all submaps')
    parser.add_argument("--feature-name", type=str, help='name of local feature extractor to use',
                        choices=list(feature_dirs.keys()), required=True)
    parser.add_argument("--output-dir", type=str, help='optional output path, will replicate records_data'
                            'directory structure from underlying kapture but will base it from the specified'
                            'directory instead. If nothing is provided then extracts to kapture-root')
    parser.add_argument("--overwrite", action='store_true', help='overwrites already extracted features')
    args = parser.parse_args()
    default_conf = getattr(import_module(f'local.kapture_extract_{args.feature_name}'), 'default_conf')
    main(args.kapture_root, args.output_dir, args.feature_name, default_conf, overwrite=args.overwrite)
