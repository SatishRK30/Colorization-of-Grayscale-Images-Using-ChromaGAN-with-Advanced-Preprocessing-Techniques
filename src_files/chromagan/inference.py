"""
Inference utilities for ChromaGAN-like model.
- colorize_image: given a grayscale image path and optional checkpoint path, returns BGR uint8 image.
- This code assumes the model defined in model.py and saved checkpoints compatible with the saver.
"""

import os
import numpy as np
import tensorflow as tf
from .model import build_model
from .utils import preprocess_L_from_gray, postprocess_ab_output, lab_scaled_to_bgr, load_image_grayscale

tf.compat.v1.disable_eager_execution()

def colorize_image(gray_path, ckpt_path=None, output_size=(224,224)):
    # Load grayscale image
    gray = load_image_grayscale(gray_path, target_size=output_size)
    L = preprocess_L_from_gray(gray, target_size=output_size)  # shape H,W,1
    L_batch = np.expand_dims(L, axis=0)  # [1,H,W,1]

    # Build model
    m = build_model(batch_size=1, height=output_size[0], width=output_size[1])
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=sess_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if ckpt_path is not None and os.path.exists(ckpt_path):
            saver = m['saver']
            saver.restore(sess, ckpt_path)
        else:
            # If no checkpoint, warning: results will be random/untrained.
            print("Warning: No valid checkpoint provided. Output will be from untrained network and may be poor.")

        gen_ab = sess.run(m['gen_ab'], feed_dict={m['L_ph']: L_batch})  # [-1,1]
        gen_ab = np.squeeze(gen_ab, axis=0)  # H,W,2
        ab = postprocess_ab_output(gen_ab)  # scale to [-128,127] approx
        bgr = lab_scaled_to_bgr(L, ab)
        return bgr

def colorize_and_save(gray_path, ckpt_path, out_path):
    bgr = colorize_image(gray_path, ckpt_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import cv2
    cv2.imwrite(out_path, bgr)
    return out_path
