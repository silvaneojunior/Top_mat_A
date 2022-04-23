from tensorflow import *

sum=math.reduce_sum
any=math.reduce_any
max=math.reduce_max
min=math.reduce_min
sqrt=math.sqrt
flip=reverse

pretty_print=print

# class Map_fn(keras.layers.Layer):    
#     def __init__(self,data,n):
#         super(Map_fn, self).__init__(name='Map_fn_layer',dtype=data.dtype) # Chamando o inicializador da superclasse
#         self.data=data
#         self.size=data.shape[-1]-n+1
#         self.n=n
#     def helper(self,index):
#         return self.data[...,index:index+self.n]
#     def call(self):
#         map_fn(self.helper,arange(self.size),fn_output_signature=self.data.dytpe)

# def slicer(data,n):
#     slicer_layer=Map_fn(data,n)
#     l=max(arange(slicer_layer.size))
#     print(data[...,l:l+n].shape)
#     return slicer_layer()