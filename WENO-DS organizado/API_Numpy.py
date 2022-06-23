from numpy import *

constant = asarray
concat   = concatenate
range    = arange
gather   = take

def map_fn(fn, elems, dtype=None, parallel_iterations=None, back_prop=True, swap_memory=False, infer_shape=True, name=None, fn_output_signature=None):
    outs=[]
    for i in elems:
        outs.append(fn(i))
    return stack(outs, axis=0)

cast = lambda x, dtype: asarray(x).astype(dtype)
max  = amax
min  = amin
abs  = absolute

def unstack(a, axis=0):
    return moveaxis(a, axis, 0)

function = lambda x:x
reverse  = lambda x, axis: flip(x, axis=axis)

pretty_print = print
