import tensorflow as tf

Network=lambda x: tf.ones(tf.shape(x[:,:,:]),dtype=x.dtype)#tf.nn.elu(x)+1