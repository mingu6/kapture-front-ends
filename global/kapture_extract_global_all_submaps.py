import argparse
import logging
import os
import os.path as path
from importlib import import_module


feature_dirs = {
    'netvlad': 'netvlad_tf_open',
}


def main(kapture_root, output_dir, feature_name, config, overwrite=False):
    # import feature extractor
    extract_kapture_global = getattr(import_module(f'global.kapture_extract_{feature_name}'), 'extract_kapture_global')
    # iterate over submaps and process in sequence
    submap_names = [f.name for f in os.scandir(kapture_root) if f.is_dir()]
    for submap_name in submap_names:
        logging.info(f'Extracting {feature_name} features for submap {submap_name} with configuration:\n{config}')
        submap_root = path.join(kapture_root, submap_name)
        submap_out = path.join(output_dir, submap_name)
        split_names = [f.name for f in os.scandir(submap_root) if f.is_dir()]
        for split in split_names:
            extract_kapture_global(path.join(submap_root, split), config, output_dir=path.join(submap_out, split), overwrite=overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kapture-root", type=str, required=True,
                        help='path to kapture root directory')
    parser.add_argument("--feature-name", type=str, help='name of local feature extractor to use',
                        choices=list(feature_dirs.keys()), required=True)
    parser.add_argument("--output-dir", type=str, help='optional output path, will replicate records_data'
                            'directory structure from underlying kapture but will base it from the specified'
                            'directory instead. If nothing is provided then extracts to kapture-root')
    parser.add_argument("--overwrite", action='store_true', help='overwrites already extracted features')
    args = parser.parse_args()
    default_conf = getattr(import_module(f'global.kapture_extract_{args.feature_name}'), 'default_conf')
    main(args.kapture_root, args.output_dir, args.feature_name, default_conf, overwrite=args.overwrite)
