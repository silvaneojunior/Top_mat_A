import numpy as np

import API_Numpy
import API_TensorFlow

dtype='float64'

def const(x, API):
    return API.constant(x, dtype = dtype)

ε_default = np.asarray(10, dtype=dtype)**np.asarray(-40, dtype=dtype)

B1 = np.asarray([[ 1,0,0],[0, 6,0],[0,0, 3]], dtype=dtype)/10 # Matriz B1
B2 = np.asarray([[-2,0,0],[0,19,0],[0,0,-2]], dtype=dtype)/15 # Matriz B2
B3 = np.asarray([[ 7,0,0],[0,15,0],[0,0, 8]], dtype=dtype)/30 # Matriz B3

C1 = np.asarray([[ 2,-7,11,0,0],[0,-1, 5, 2,0],[0,0, 2,5,-1]], dtype=dtype)/6  # Matriz C1
C2 = np.asarray([[-1, 2,11,0,0],[0,-1,14,-1,0],[0,0,11,2,-1]], dtype=dtype)/12 # Matriz C2
C3 = np.asarray([[ 1,-4, 7,0,0],[0,-1, 4, 1,0],[0,0, 1,4,-1]], dtype=dtype)/4  # Matriz C3

C1 = np.transpose(C1)
C2 = np.transpose(C2)
C3 = np.transpose(C3)

# γ = np.asarray(14, dtype=dtype)/10