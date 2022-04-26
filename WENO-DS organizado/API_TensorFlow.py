from tensorflow import *

sum=math.reduce_sum
any=math.reduce_any
max=math.reduce_max
min=math.reduce_min
sqrt=math.sqrt
flip=reverse

pretty_print=print

def closed_range(a,b,Δx):
    values=range(0,1,Δx)
    values/=max(values)
    values*=b-a
    values+=a
    return values