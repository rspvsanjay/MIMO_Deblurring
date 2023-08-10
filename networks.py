import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.python.ops.gen_nn_ops import relu

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, str1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.str1 = str1

    def build(self, input_shape):
        self.conv1_layers = [layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoRb1_{self.str1}_{i}') for i in range(0, 8)]
        self.conv2_layers = [layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoRb2_{self.str1}_{i}') for i in range(0, 8)]
        self.relu = layers.ReLU()

    def call(self, input_x):
        x = input_x
        for i in range(8):  # Since the loop range is (1, 8), it runs for 7 iterations.
            x = self.conv1_layers[i](x)
            x = self.conv2_layers[i](x)
            x = self.relu(x)
        return x

class AsymmetricFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, num_filters, str1, **kwargs):
        super(AsymmetricFeatureFusion, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.str1 = str1

    def build(self, input_shape):
        self.convo1aff = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoAFF1_{self.str1}')
        self.relu1aff = layers.ReLU()
        self.convo2aff = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoAFF2_{self.str1}')

    def call(self, inputs):
        a, b, c = inputs
        x = tf.concat([a, b, c], -1)
        x = self.convo1aff(x)
        x = self.relu1aff(x)
        x = self.convo2aff(x)
        return x

class FeatureAttentionModule(tf.keras.layers.Layer):
    def __init__(self, num_filters, str1, **kwargs):
        super(FeatureAttentionModule, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.str1 = str1

    def build(self, input_shape):
        self.conv_layer = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoFAM1_{self.str1}')
        
    def call(self, inputs):
        x, scm_out = inputs
        xx = tf.math.multiply(scm_out, x, name=None)
        xx = self.conv_layer(xx)
        x = tf.math.add(xx, x, name=None)
        return x

class ShallowConvolutionModule(tf.keras.layers.Layer):
    def __init__(self, num_filters, str1, **kwargs):
        super(ShallowConvolutionModule, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.str1 = str1

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoSCM1_{self.str1}')
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters=self.num_filters, kernel_size=1, strides=(1, 1), padding='same', use_bias=True, name=f'convoSCM2_{self.str1}')
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoSCM3_{self.str1}')
        self.relu3 = layers.ReLU()
        self.conv4 = layers.Conv2D(filters=self.num_filters, kernel_size=1, strides=(1, 1), padding='same', use_bias=True, name=f'convoSCM4_{self.str1}')
        self.relu4 = layers.ReLU()
        self.conv5 = layers.Conv2D(filters=self.num_filters, kernel_size=3, strides=(1, 1), padding='same', use_bias=True, name=f'convoSCM5_{self.str1}')
        self.relu5 = layers.ReLU()

    def call(self, x):
        original_x = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = tf.concat([original_x, x], -1)
        x = self.conv5(x)
        x = self.relu5(x)
        return x

class MIMO_Deblur(Model):
    def __init__(self):
        super(MIMO_Deblur, self).__init__()

        # Downsampling Level1
        self.convo1k3s1f32 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name='convo1k3s1f32')
        self.residual_block1 = ResidualBlock(32, 'res_block1')

        # Downsampling Level2
        self.convo1k3s2f64 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name='convo1k3s2f64')
        self.scm_block2 = ShallowConvolutionModule(64, 'scm_block2')
        self.fam_block2 = FeatureAttentionModule(64, 'fam_block2')
        self.residual_block2 = ResidualBlock(64, 'res_block2')

        # Downsampling Level3
        self.convo1k3s2f128 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name='convo1k3s2f128')
        self.scm_block3 = ShallowConvolutionModule(128, 'scm_block3')
        self.fam_block3 = FeatureAttentionModule(128, 'fam_block3')
        self.residual_block3 = ResidualBlock(128, 'res_block3')

        # Asymmetric Feature Fusion
        self.aff1 = AsymmetricFeatureFusion(32, 'aff1')
        self.aff2 = AsymmetricFeatureFusion(64, 'aff2')

        # Upsampling Level3
        self.residual_block4 = ResidualBlock(128, 'res_block4')
        self.trnspconvo1k3s2f128 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name='trnspconvo1k3s2f128')
        self.convo1k3s1f3l3 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name='convo1k3s1f3l3')

        # Upsampling Level2
        self.convo2k3s1f64 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name='convo2k3s1f64')
        self.residual_block5 = ResidualBlock(64, 'res_block5')
        self.trnspconvo1k3s2f64 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name='trnspconvo1k3s2f64')
        self.convo1k3s1f3l2 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name='convo1k3s1f3l2')

        # Upsampling Level1
        self.convo2k3s1f32 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name='convo2k3s1f32')
        self.residual_block6 = ResidualBlock(32, 'res_block6')
        self.convo1k3s1f3l1 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name='convo1k3s1f3l1')
        
    def call(self, inputs):
        output3, output2, output1 = self.call_layers(inputs)
        return output3, output2, output1

    def call_layers(self, inputs):
        b1 = inputs
        b2 = tf.image.resize(b1, [tf.shape(b1)[1]//2, tf.shape(b1)[2]//2])
        b3 = tf.image.resize(b2, [tf.shape(b2)[1]//2, tf.shape(b2)[2]//2])

        # Downsampling Level1
        e1 = self.convo1k3s1f32(b1)
        e1 = self.residual_block1(e1)

        # Downsampling Level2
        e2 = self.convo1k3s2f64(e1)
        scm_out = self.scm_block2(b2)
        e2 = self.fam_block2([e2, scm_out])
        e2 = self.residual_block2(e2)

        # Downsampling Level3
        e3 = self.convo1k3s2f128(e2)
        scm_out = self.scm_block3(b3)
        e3 = self.fam_block3([e3, scm_out])
        e3 = self.residual_block3(e3)

        # Asymmetric Feature Fusion
        ee2 = tf.image.resize(e2, [tf.shape(b1)[1], tf.shape(b1)[2]])
        ee3 = tf.image.resize(e3, [tf.shape(b1)[1], tf.shape(b1)[2]])
        aaff1 = self.aff1([e1, ee2, ee3])
        ee1 = tf.image.resize(e1, [tf.shape(b1)[1]//2, tf.shape(b1)[2]//2])
        ee3 = tf.image.resize(e3, [tf.shape(b1)[1]//2, tf.shape(b1)[2]//2])
        aaff2 = self.aff2([ee1, e2, ee3])

        # Upsampling Level3
        d3 = self.residual_block4(e3)
        S3 = self.convo1k3s1f3l3(d3)
        d3 = self.trnspconvo1k3s2f128(d3)

        # Upsampling Level2
        d2 = tf.concat([d3, aaff2], -1)
        d2 = self.convo2k3s1f64(d2)
        d2 = self.residual_block5(d2)
        S2 = self.convo1k3s1f3l2(d2)
        d2 = self.trnspconvo1k3s2f64(d2)

        # Upsampling Level1
        d1 = tf.concat([d2, aaff1], -1)
        d1 = self.convo2k3s1f32(d1)
        d1 = self.residual_block6(d1)
        S1 = self.convo1k3s1f3l1(d1)

        output3 = tf.identity(S3, name='output3')
        output2 = tf.identity(S2, name='output2')
        output1 = tf.identity(S1, name='output1')

        return output3, output2, output1

    def compute_output_shape(self, input_shape):
          return [(input_shape[0], None, None, 3), (input_shape[0], None, None, 3), (input_shape[0], None, None, 3)]
