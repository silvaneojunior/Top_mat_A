import tensorflow as tf

Network=lambda x: tf.ones(tf.shape(x[:,2:,:]),dtype=x.dtype)#tf.nn.elu(x)+1