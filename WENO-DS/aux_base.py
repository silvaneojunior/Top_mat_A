import numpy as np

import API_Numpy
import API_TensorFlow

dtype='float64'

def const(x, API):
    return API.constant(x, dtype = dtype)

ε_default = np.asarray(10, dtype=dtype)**np.asarray(-40, dtype=dtype)

B = np.asarray([[1,0,0],[0,6,0],[0,0,3]], dtype=dtype)/10                # Matriz B
C = np.asarray([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]], dtype=dtype)/6 # Matriz C
C = np.transpose(C)

# γ = np.asarray(14, dtype=dtype)/10