"""
ChromaGAN-like generator & discriminator (simplified)
TensorFlow 1.x compatible (uses tf.compat.v1).
This implementation provides:
- generator: U-Net style encoder-decoder that predicts 'ab' channels from L input
- discriminator: PatchGAN-style discriminator
- build_model: wires up generator + discriminator and returns relevant tensors

Notes:
- This is a reference implementation for inference/training experiments.
- You will need a trained checkpoint to perform high-quality colorization.
- Report PDF (for project reference): /mnt/data/Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques (1).pdf
"""

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def conv_block(x, filters, kernel=4, stride=2, name='conv', use_bn=True, activation=tf.nn.leaky_relu):
    with tf.compat.v1.variable_scope(name):
        x = tf.compat.v1.layers.conv2d(
            x, filters, kernel, stride, padding='same',
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        if use_bn:
            x = tf.compat.v1.layers.batch_normalization(x, training=False)
        if activation is not None:
            x = activation(x)
        return x

def deconv_block(x, skip, filters, kernel=4, stride=2, name='deconv', use_dropout=False):
    with tf.compat.v1.variable_scope(name):
        x = tf.compat.v1.layers.conv2d_transpose(
            x, filters, kernel, stride, padding='same',
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        x = tf.compat.v1.layers.batch_normalization(x, training=False)
        if use_dropout:
            x = tf.compat.v1.layers.dropout(x, rate=0.5, training=False)
        x = tf.nn.relu(x)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        return x

def build_generator(input_L, name='generator', base_filters=64):
    """
    input_L: placeholder or tensor [batch, H, W, 1] in range [0, 100] (L channel of LAB)
    returns: predicted ab channels tensor [batch, H, W, 2]
    """
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        # Encoder
        e1 = conv_block(input_L, base_filters, use_bn=False, name='enc1')   # 112x112
        e2 = conv_block(e1, base_filters*2, name='enc2')                    # 56x56
        e3 = conv_block(e2, base_filters*4, name='enc3')                    # 28x28
        e4 = conv_block(e3, base_filters*8, name='enc4')                    # 14x14
        e5 = conv_block(e4, base_filters*8, name='enc5')                    # 7x7

        # Bottleneck
        b = conv_block(e5, base_filters*8, name='bottleneck')

        # Decoder (U-Net style)
        d1 = deconv_block(b, e5, base_filters*8, name='dec1', use_dropout=True)  # 14x14
        d2 = deconv_block(d1, e4, base_filters*8, name='dec2', use_dropout=True) # 28x28
        d3 = deconv_block(d2, e3, base_filters*4, name='dec3')                  # 56x56
        d4 = deconv_block(d3, e2, base_filters*2, name='dec4')                  # 112x112
        d5 = deconv_block(d4, e1, base_filters, name='dec5')                    # 224x224

        with tf.compat.v1.variable_scope('out'):
            ab = tf.compat.v1.layers.conv2d(d5, 2, kernel_size=3, strides=1, padding='same',
                                           activation=tf.nn.tanh,
                                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
            # output in range [-1,1], we will scale to typical ab range ~[-128,127] externally
        return ab

def build_discriminator(input_L, input_ab, name='discriminator'):
    """
    PatchGAN discriminator that takes the concatenation of L and ab and predicts real/fake patches.
    input_L: [batch, H, W, 1], input_ab: [batch, H, W, 2]
    returns: logits tensor
    """
    x = tf.concat([input_L, input_ab], axis=-1)
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        h1 = conv_block(x, 64, use_bn=False, name='d_conv1')   # 112x112
        h2 = conv_block(h1, 128, name='d_conv2')               # 56x56
        h3 = conv_block(h2, 256, name='d_conv3')               # 28x28
        h4 = conv_block(h3, 512, name='d_conv4')               # 14x14
        logits = tf.compat.v1.layers.conv2d(h4, 1, kernel_size=4, strides=1, padding='same',
                                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        return logits

def build_model(batch_size=1, height=224, width=224):
    """
    Build placeholders and model ops for convenience.
    Returns dict with placeholders and outputs.
    """
    L_ph = tf.compat.v1.placeholder(tf.float32, [None, height, width, 1], name='L_input')
    ab_ph = tf.compat.v1.placeholder(tf.float32, [None, height, width, 2], name='ab_input')  # for training/disc

    gen_ab = build_generator(L_ph)
    disc_fake = build_discriminator(L_ph, gen_ab)
    disc_real = build_discriminator(L_ph, ab_ph)

    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    return {
        'L_ph': L_ph,
        'ab_ph': ab_ph,
        'gen_ab': gen_ab,
        'disc_fake': disc_fake,
        'disc_real': disc_real,
        'saver': saver
    }
