import pdb

import argparse
import logging
import os
import os.path as path
import sys
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import cv2

import kapture
from kapture.io.features import image_global_features_to_file, get_global_features_fullpath, global_features_check_dir
from kapture.io.csv import get_csv_fullpath, kapture_from_dir, descriptors_to_file, global_features_from_dir
from kapture.io.records import get_image_fullpath
from kapture.io.structure import delete_existing_kapture_files

nv_path = path.abspath(
    path.join(path.dirname(__file__), "../thirdparty/netvlad_tf_open/python")
)
sys.path.append(nv_path)
chkpt_path = path.abspath(
    path.join(path.dirname(__file__), "../checkpoints/netvlad/")
)
import netvlad_tf.nets as nets


default_conf = {
    'checkpoint': 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white/vd16_pitts30k_conv5_3_vlad_preL2_intra_white',
    'grayscale': False,
    'pca_dim': 4096,
}


def extract_kapture_global(kapture_root,
                           config,
                           output_dir='',
                           overwrite=False):
    logging.info('Extracting NetVLAD features with configuration:\n', config)
    # use kapture io to identify image paths and loop
    kdata = kapture_from_dir(kapture_root, matches_pairsfile_path=None,
                             skip_list= [kapture.Matches,
                                         kapture.Points3d,
                                         kapture.Observations,
                                         kapture.Keypoints,
                                         kapture.Descriptors])
    assert kdata.records_camera is not None

    export_dir = output_dir if output_dir else kapture_root  # root of output directory for features
    os.makedirs(export_dir, exist_ok=True)

    image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]

    # resume extraction if some features exist
    try:
        # load features if there are any
        kdata.global_features = global_features_from_dir(export_dir, None)
        if kdata.global_features is not None and not overwrite:
            image_list = [name for name in image_list if name not in kdata.global_features]
    except FileNotFoundError:
        pass
    except:
        logging.exception("Error with importing existing global features.")

    # clear features first if overwriting
    if overwrite: delete_existing_kapture_files(export_dir, True, only=[kapture.GlobalFeatures])

    if len(image_list) == 0:
        print('All features were already extracted')
        return
    else:
        print(f'Extracting NetVLAD features for {len(image_list)} images')

    # for the global descriptor type specification
    global_dtype = None if kdata.global_features is None else kdata.global_features.dtype
    global_dsize = None if kdata.global_features is None else kdata.global_features.dsize

    # setup network
    tf.reset_default_graph()
    if config['grayscale']:
        tf_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 1])
    else:
        tf_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3])
    # load network and checkpoint
    net = nets.vgg16NetvladPca(tf_batch)
    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint = chkpt_path + '/' + config['checkpoint']
    saver.restore(sess, checkpoint)

    for image_name in image_list:
        img_path = get_image_fullpath(kapture_root, image_name)
        if img_path.endswith('.txt'):
            args.images = open(img_path).read().splitlines() + args.images
            continue

        print(f"\nExtracting features for {img_path}")

        if config['grayscale']:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(
                np.expand_dims(image, axis=0), axis=-1)
        else:
            image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
        descriptor = sess.run(net, feed_dict={tf_batch: image})[:, :config['pca_dim']]
        descriptor = np.squeeze(descriptor)


        # write global descriptor type specification
        if global_dtype is None:
            global_dtype = descriptor.dtype
            global_dsize = len(descriptor)

            kdata.global_features = kapture.GlobalFeatures('netvlad', global_dtype, global_dsize)

            global_descriptors_config_abs_path = get_csv_fullpath(kapture.GlobalFeatures, export_dir)
            descriptors_to_file(global_descriptors_config_abs_path, kdata.global_features)
        else:
            assert kdata.global_features.type_name == "netvlad"
            assert kdata.global_features.dtype == descriptor.dtype
            assert kdata.global_features.dsize == len(descriptor)
        # get output paths
        global_descriptors_abs_path = get_global_features_fullpath(export_dir, image_name)

        image_global_features_to_file(global_descriptors_abs_path, descriptor)
        kdata.global_features.add(image_name)
    # sess.close()  # close session before initializing again for next submap

    if not global_features_check_dir(kdata.global_features, export_dir):
        print('global feature extraction ended successfully but not all files were saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kapture-root", type=str, required=True,
                            help='path to root directory containing all submaps')
    parser.add_argument("--output-dir", type=str, help='optional output path, will replicate records_data'
                            'directory structure from underlying kapture but will base it from the specified'
                            'directory instead. If nothing is provided then')
    parser.add_argument("--overwrite", action='store_true', help='overwrites already extracted features')
    args = parser.parse_args()
    extract_kapture_global(args.kapture_root, default_conf, output_dir=args.output_dir, overwrite=args.overwrite)
