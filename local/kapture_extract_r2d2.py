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
# import R2D2 modules
from PIL import Image

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

from extract import load_network, NonMaxSuppression, extract_multiscale

# import Kapture modules
import kapture
from kapture.io.records import get_image_fullpath
from kapture.io.csv import kapture_from_dir, keypoints_from_dir, descriptors_from_dir
from kapture.io.csv import get_csv_fullpath, keypoints_to_file, descriptors_to_file
from kapture.io.features import get_keypoints_fullpath, keypoints_check_dir, image_keypoints_to_file
from kapture.io.features import get_descriptors_fullpath, descriptors_check_dir, image_descriptors_to_file
from kapture.io.structure import delete_existing_kapture_files


default_conf = {
    'checkpoint': 'models/r2d2_WASF_N8_big.pt',
    'top_k': 4000,  # number of kpts
    'scale_f': 2 ** 0.25,  # scale factor
    'min_size': 256,
    'max_size': 1024,
    'min_scale': 0,
    'max_scale': 1,
    'reliability_thr': 0.7,
    'repeatability_thr': 0.7,
}


def extract_kapture_keypoints(kapture_root,
                              config,
                              output_dir='',
                              overwrite=False):
    """
    Extract r2d2 keypoints and descritors to the kapture format directly
    """
    print('extract_kapture_keypoints...')
    kdata = kapture_from_dir(kapture_root, matches_pairsfile_path=None,
    skip_list= [kapture.GlobalFeatures,
                kapture.Matches,
                kapture.Points3d,
                kapture.Observations])
    export_dir = output_dir if output_dir else kapture_root  # root of output directory for features
    os.makedirs(export_dir, exist_ok=True)

    assert kdata.records_camera is not None
    image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]
    # resume extraction if some features exist
    try:
        # load existing features, if any
        kdata.keypoints = keypoints_from_dir(export_dir, None)
        kdata.descriptors = descriptors_from_dir(export_dir, None)
        if kdata.keypoints is not None and kdata.descriptors is not None and not overwrite:
            image_list = [name for name in image_list if name not in kdata.keypoints or name not in kdata.descriptors]
    except FileNotFoundError:
        pass
    except:
        logging.exception("Error with importing existing local features.")

    # clear features first if overwriting
    if overwrite: delete_existing_kapture_files(export_dir, True, only=[kapture.Descriptors, kapture.Keypoints])

    if len(image_list) == 0:
        print('All features were already extracted')
        return
    else:
        print(f'Extracting r2d2 features for {len(image_list)} images')

    iscuda = common.torch_set_gpu([torch.cuda.is_available()])

    # load the network...
    net = load_network(config['checkpoint'])
    if iscuda: net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = config['reliability_thr'],
        rep_thr = config['repeatability_thr'])

    keypoints_dtype = None if kdata.keypoints is None else kdata.keypoints.dtype
    descriptors_dtype = None if kdata.descriptors is None else kdata.descriptors.dtype

    keypoints_dsize = None if kdata.keypoints is None else kdata.keypoints.dsize
    descriptors_dsize = None if kdata.descriptors is None else kdata.descriptors.dsize

    for image_name in image_list:
        img_path = get_image_fullpath(kapture_root, image_name)

        if img_path.endswith('.txt'):
            images = open(img_path).read().splitlines() + images
            continue

        print(f"\nExtracting features for {img_path}")
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        img = norm_RGB(img)[None]
        if iscuda: img = img.cuda()

        # extract keypoints/descriptors for a single image
        xys, desc, scores = extract_multiscale(net, img, detector,
            scale_f   = config['scale_f'],
            min_scale = config['min_scale'],
            max_scale = config['max_scale'],
            min_size  = config['min_size'],
            max_size  = config['max_size'],
            verbose = True)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-config['top_k'] or None:]

        xys = xys[idxs]
        desc = desc[idxs]
        if keypoints_dtype is None or descriptors_dtype is None:
            keypoints_dtype = xys.dtype
            descriptors_dtype = desc.dtype

            keypoints_dsize = xys.shape[1]
            descriptors_dsize = desc.shape[1]

            kdata.keypoints = kapture.Keypoints('r2d2', keypoints_dtype, keypoints_dsize)
            kdata.descriptors = kapture.Descriptors('r2d2', descriptors_dtype, descriptors_dsize)

            keypoints_config_absolute_path = get_csv_fullpath(kapture.Keypoints, export_dir)
            descriptors_config_absolute_path = get_csv_fullpath(kapture.Descriptors, export_dir)

            keypoints_to_file(keypoints_config_absolute_path, kdata.keypoints)
            descriptors_to_file(descriptors_config_absolute_path, kdata.descriptors)
        else:
            assert kdata.keypoints.type_name == 'r2d2'
            assert kdata.descriptors.type_name == 'r2d2'
            assert kdata.keypoints.dtype == xys.dtype
            assert kdata.descriptors.dtype == desc.dtype
            assert kdata.keypoints.dsize == xys.shape[1]
            assert kdata.descriptors.dsize == desc.shape[1]

        keypoints_fullpath = get_keypoints_fullpath(export_dir, image_name)
        print(f"Saving {xys.shape[0]} keypoints to {keypoints_fullpath}")
        image_keypoints_to_file(keypoints_fullpath, xys)
        kdata.keypoints.add(image_name)


        descriptors_fullpath = get_descriptors_fullpath(export_dir, image_name)
        print(f"Saving {desc.shape[0]} descriptors to {descriptors_fullpath}")
        image_descriptors_to_file(descriptors_fullpath, desc)
        kdata.descriptors.add(image_name)

    if not keypoints_check_dir(kdata.keypoints, export_dir) or \
            not descriptors_check_dir(kdata.descriptors, export_dir):
        print('local feature extraction ended successfully but not all files were saved')


def main(kapture_root, output_dir, config, overwrite=False):
    logging.info('Extracting R2D2 features with configuration:\n', config)
    config['checkpoint'] = r2d2_path + '/' + config['checkpoint']
    extract_kapture_keypoints(kapture_root, config, output_dir=output_dir, overwrite=overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kapture-root", type=str, required=True,
                            help='path to kapture root directory')
    parser.add_argument("--output-dir", type=str, help='optional output path, will replicate records_data'
                            'directory structure from underlying kapture but will base it from the specified'
                            'directory instead. If nothing is provided then')
    parser.add_argument("--overwrite", action='store_true', help='overwrites already extracted features')
    args = parser.parse_args()
    main(args.kapture_root, args.output_dir, default_conf, overwrite=args.overwrite)
