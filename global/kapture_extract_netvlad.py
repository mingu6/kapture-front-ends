import argparse
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import cv2

import kapture
from kapture.io.features import image_global_features_to_file, get_global_features_fullpath, global_features_check_dir
from kapture.io.csv import get_csv_fullpath, kapture_from_dir, descriptors_to_file
from kapture.io.records import get_image_fullpath

nv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../thirdparty/netvlad_tf_open/python")
)
sys.path.append(nv_path)
chkpt_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../checkpoints/netvlad/")
)
import netvlad_tf.nets as nets


default_conf = {
    'checkpoint': 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white/vd16_pitts30k_conv5_3_vlad_preL2_intra_white',
    'grayscale': False,
    'pca_dim': 1024,
}


def main(kapture_root, config):
    logging.info('Extracting NetVLAD features with configuration:\n', config)
    # setup input placeholder
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
    # use kapture io to identify image paths and loop
    kdata = kapture_from_dir(kapture_root, matches_pairsfile_path=None,
                             skip_list= [kapture.Matches,
                                         kapture.Points3d,
                                         kapture.Observations,
                                         kapture.Keypoints,
                                         kapture.Descriptors, kapture.GlobalFeatures])
    assert kdata.records_camera is not None
    image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]

    # for the global descriptor type specification
    global_dtype = None if kdata.global_features is None else kdata.global_features.dtype
    global_dsize = None if kdata.global_features is None else kdata.global_features.dsize

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

            global_descriptors_config_abs_path = get_csv_fullpath(kapture.GlobalFeatures, kapture_root)
            descriptors_to_file(global_descriptors_config_abs_path, kdata.global_features)
        else:
            assert kdata.global_features.type_name == "netvlad"
            assert kdata.global_features.dtype == "float32"
            assert kdata.global_features.dsize == len(descriptor)
        # get output paths
        global_descriptors_abs_path = get_global_features_fullpath(kapture_root, image_name)

        image_global_features_to_file(global_descriptors_abs_path, descriptor)
        kdata.global_features.add(image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kapture-root", type=str, required=True,
                        help='path to kapture root directory')
    args = parser.parse_args()
    main(args.kapture_root, default_conf)
