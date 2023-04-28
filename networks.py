import tensorflow as tf

class Asymetric_Feature_Fusion(tf.keras.layers.Layer): # Used extract the feature from downsampled image so that it can be merge with deeper layers
    def __init__(self, num_filters, trainable, **kwargs):
        super(Asymetric_Feature_Fusion, self).__init__(**kwargs)
        self.convoAFF1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoAFF1')
        self.convoAFF2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoAFF2')
        self.reluAFF = tf.keras.layers.ReLU()

    def call(self, a, b, c):
        x = tf.concat([a,b,c],-1)
        x = self.convoAFF1(x)
        x = self.reluAFF(x)
        x = self.convoAFF2(x)
        return x

class Feature_Attention_Module(tf.keras.layers.Layer): # Used extract the feature from downsampled image so that it can be merge with deeper layers
    def __init__(self, num_filters, trainable, **kwargs):
        super(Feature_Attention_Module, self).__init__(**kwargs)
        self.convoFAM1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoFAM1') 

    def call(self, x, scm_out):
        xx = tf.math.multiply(scm_out, x, name = None)
        xx = self.convoFAM1(xx)
        x = tf.math.add(xx, x, name = None)
        return x

class Shallow_Convolution_Module(tf.keras.layers.Layer): # Used extract the feature from downsampled image so that it can be merge with deeper layers
    def __init__(self, num_filters, trainable, **kwargs):
        super(Shallow_Convolution_Module, self).__init__(**kwargs)
        self.convoSCM1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoSCM1')
        self.reluSCM1 = tf.keras.layers.ReLU()
        self.convoSCM2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoSCM2')
        self.reluSCM2 = tf.keras.layers.ReLU()
        self.convoSCM3 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoSCM3')
        self.reluSCM3 = tf.keras.layers.ReLU()
        self.convoSCM4 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoSCM4')
        self.reluSCM4 = tf.keras.layers.ReLU()
        self.convoSCM5 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoSCM5')
        self.reluSCM5 = tf.keras.layers.ReLU()

    def call(self, x):
        orignal_x = x
        x = self.convoSCM1(x)
        x = self.reluSCM1(x)

        x = self.convoSCM2(x)
        x = self.reluSCM2(x)

        x = self.convoSCM3(x)
        x = self.reluSCM3(x)

        x = self.convoSCM4(x)
        x = self.reluSCM4(x)

        x = tf.concat([orignal_x, x], -1)

        x = self.convoSCM5(x)
        x = self.reluSCM5(x)
        return x

class Residual_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, trainable, **kwargs):
        super(Residual_Block, self).__init__(**kwargs)

        self.convoRb1_1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_1')
        self.convoRb2_1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_1')
        self.reluRb1_1 = tf.keras.layers.ReLU()

        self.convoRb1_2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_2')
        self.convoRb2_2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_2')
        self.reluRb1_2 = tf.keras.layers.ReLU()

        self.convoRb1_3 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_3')
        self.convoRb2_3 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_3')
        self.reluRb1_3 = tf.keras.layers.ReLU()

        self.convoRb1_4 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_4')
        self.convoRb2_4 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_4')
        self.reluRb1_4 = tf.keras.layers.ReLU()

        self.convoRb1_5 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_5')
        self.convoRb2_5 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_5')
        self.reluRb1_5 = tf.keras.layers.ReLU()

        self.convoRb1_6 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_6')
        self.convoRb2_6 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_6')
        self.reluRb1_6 = tf.keras.layers.ReLU()

        self.convoRb1_7 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_7')
        self.convoRb2_7 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_7')
        self.reluRb1_7 = tf.keras.layers.ReLU()

        self.convoRb1_8 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb1_8')
        self.convoRb2_8 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convoRb2_8')
        self.reluRb1_8 = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.convoRb1_1(x)
        x = self.convoRb2_1(x)
        x = self.reluRb1_1(x)

        x = self.convoRb1_2(x)
        x = self.convoRb2_2(x)
        x = self.reluRb1_3(x)
        
        x = self.convoRb1_3(x)
        x = self.convoRb2_3(x)
        x = self.reluRb1_3(x)

        x = self.convoRb1_4(x)
        x = self.convoRb2_4(x)
        x = self.reluRb1_4(x)

        x = self.convoRb1_5(x)
        x = self.convoRb2_5(x)
        x = self.reluRb1_5(x)

        x = self.convoRb1_6(x)
        x = self.convoRb2_6(x)
        x = self.reluRb1_6(x)
        
        x = self.convoRb1_7(x)
        x = self.convoRb2_7(x)
        x = self.reluRb1_7(x)

        x = self.convoRb1_8(x)
        x = self.convoRb2_8(x)
        x = self.reluRb1_8(x)

        return x

# define the model using subclassing
class MIMO_Network(tf.keras.Model):
    def __init__(self):
        super(MIMO_Network, self).__init__()
        self.num_filters = 32
        self.trainable = True

        # Downsampling Level1
        self.convo1k3s1f32 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convo1k3s1f32')
        self.residual_block1 = Residual_Block(self.num_filters, self.trainable, name = 'residual_block1')

        # Downsampling Level2
        self.convo1k3s2f64 = tf.keras.layers.Conv2D(filters = self.num_filters*2, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name = 'convo1k3s2f64')
        self.scm_block2 = Shallow_Convolution_Module(self.num_filters*2, self.trainable, name = 'scm_block2')
        self.fam_block2 = Feature_Attention_Module(self.num_filters*2, self.trainable, name = 'fam_block2')
        self.residual_block2 = Residual_Block(self.num_filters*2, self.trainable, name = 'residual_block2')

        # Downsampling Level3
        self.convo1k3s2f128 = tf.keras.layers.Conv2D(filters = self.num_filters*4, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name = 'convo1k3s2f128')
        self.scm_block3 = Shallow_Convolution_Module(self.num_filters*4, self.trainable, name = 'scm_block3')
        self.fam_block3 = Feature_Attention_Module(self.num_filters*4, self.trainable, name = 'fam_block3')
        self.residual_block3 = Residual_Block(self.num_filters*4, self.trainable, name = 'residual_block3')

        # Asymetric Feature Fusion
        self.aff1 = Asymetric_Feature_Fusion(self.num_filters, self.trainable, name = 'scm_block3')
        self.aff2 = Asymetric_Feature_Fusion(self.num_filters*2, self.trainable, name = 'scm_block3')

        # Upsampling Level3
        self.residual_block4 = Residual_Block(self.num_filters*4, self.trainable, name = 'residual_block4')
        self.trnspconvo1k3s2f128 = tf.keras.layers.Conv2DTranspose(filters = self.num_filters*4, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name = 'trnspconvo1k3s2f128')
        self.convo1k3s1f3l3 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convo1k3s1f3l3')

        # Upsampling Level2
        self.convo2k3s1f64 = tf.keras.layers.Conv2D(filters = self.num_filters*2, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convo2k3s1f64')
        self.residual_block5 = Residual_Block(self.num_filters*2, self.trainable, name = 'residual_block5')
        self.trnspconvo1k3s2f64 = tf.keras.layers.Conv2DTranspose(filters = self.num_filters*2, kernel_size=3, strides=(2,2), padding='same', use_bias=True, name = 'trnspconvo1k3s2f64')
        self.convo1k3s1f3l2 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convo1k3s1f3l2')

        # Upsampling Level1
        self.convo2k3s1f32 = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convo2k3s1f32')
        self.residual_block6 = Residual_Block(self.num_filters, self.trainable, name = 'residual_block6')
        self.convo1k3s1f3l1 = tf.keras.layers.Conv2D(3, kernel_size=3, strides=(1,1), padding='same', use_bias=True, name = 'convo1k3s1f3l1')

    
    def call(self, inputs):
        b1 = inputs
        b2 = tf.image.resize(b1, [tf.shape(b1)[1]//2, tf.shape(b1)[2]//2])
        b3 = tf.image.resize(b2, [tf.shape(b2)[1]//2, tf.shape(b2)[2]//2])
        # Downsampling Level1
        e1 = self.convo1k3s1f32(b1)
        e1 = self.residual_block1(e1)

        # Downsampling Level2
        e2 = self.convo1k3s2f64(e1)
        scm_out = self.scm_block2(b2)
        e2 = self.fam_block2(e2, scm_out)
        e2 = self.residual_block2(e2)

        # Downsampling Level3
        e3 = self.convo1k3s2f128(e2)
        scm_out = self.scm_block3(b3)
        e3 = self.fam_block3(e3, scm_out)
        e3 = self.residual_block3(e3)

        # Asymetric Feature Fusion
        ee2 = tf.image.resize(e2, [256, 256]) #self.rezise_two_times1(e2)
        ee3 = tf.image.resize(e3, [256, 256]) #self.rezise_four_times1(e3)
        aaff1 = self.aff1(e1, ee2, ee3)
        ee1 = tf.image.resize(e1, [128, 128])  #self.rezise_one_half2(e1)
        ee3 = tf.image.resize(e3, [128, 128])  #self.rezise_two_times2(e3)
        aaff2 = self.aff2(ee1, e2, ee3)

        # Upsampling Level3
        d3 = self.residual_block4(e3)
        S3 = self.convo1k3s1f3l3(d3)
        d3 = self.trnspconvo1k3s2f128(d3)

        # Upsampling Level2
        d2 = tf.concat([d3, aaff2],-1)
        d2 = self.convo2k3s1f64(d3)
        d2 = self.residual_block5(d2)
        S2 = self.convo1k3s1f3l2(d2)
        d2 = self.trnspconvo1k3s2f64(d2)

        # Upsampling Level1
        d1 = tf.concat([d2, aaff1],-1)
        d1 = self.convo2k3s1f32(d1)
        d1 = self.residual_block6(d1)
        S1 = self.convo1k3s1f3l1(d2)

        output3 = tf.identity(S3, name = 'output3')
        output2 = tf.identity(S2, name = 'output2')
        output1 = tf.identity(S1, name = 'output1')

        return output3, output2, output1 





