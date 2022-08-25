from aux_base import dtype
from aux_base import B as B_default
import numpy as np

"""
Obtendo matrizes de constantes convenintes para executar o WENO-Z
utilizando operações tensoriais, uma vez que permite a integração
com o tensorflow
"""

null_mapping = lambda λ, API, map_function, B=B_default: API.matmul(λ, B)
    
def post_mapping(λ, API, map_function, B=B_default):

    α    = API.matmul(λ, B)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)

    return α

def pre_mapping(λ, API, map_function, B=B_default):
    
    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, B)

    return α

def post_inv_mapping(λ, API, map_function, B=B_default):

    α    = API.matmul(λ, B)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)
    α    = α * soma

    return α

def pre_inv_mapping(λ, API, map_function, B=B_default):

    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, B)
    α    = α * soma

    return α

Henrick_function = lambda ω, d: ω*(d + d**2 - 3*d*ω + ω**2)/(d**2 + ω*(1-2*d))

def Henrick_mapping(ω, API):
        
    d = [0.1, 0.6, 0.3]

    ω0 = ω[...,0:1]
    ω1 = ω[...,1:2]
    ω2 = ω[...,2:]

    ω0 = Henrick_function(ω0, d[0])
    ω1 = Henrick_function(ω1, d[1])
    ω2 = Henrick_function(ω2, d[2])
    
    α = API.concat([ω0, ω1, ω2], axis = -1)
        
    return α

def Hong_mapping(ω, API):
    
    α = Henrick_function(ω, 1/3)
    
    return α

def function_BI(x, k):
    
    if x < 1/10:
        return Henrick_function(x, 1/3)
    elif x < 7/16:
        return 1/3
    elif x < 9/10:
        if k == 0:
            return 2/5
        if k == 1:
            return 1/5
        if k == 2:
            return 2/5
    else:
        return Henrick_function(x, 1/3)

resolution = 10000
    
def discrete_map(function):
    
    vetor = list(range(0, resolution))
    for i in range(len(vetor)):
        vetor[i] = function(vetor[i]/vetor[-1])
        
    vetor = np.asarray(vetor, dtype=dtype)
    
    return vetor

vetor_BI_0 = discrete_map(lambda x: function_BI(x, k=0))
vetor_BI_1 = discrete_map(lambda x: function_BI(x, k=1))
vetor_BI_2 = discrete_map(lambda x: function_BI(x, k=2))

def BI_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BI_0, index[...,0:1])
    ω1 = API.gather(vetor_BI_1, index[...,1:2])
    ω2 = API.gather(vetor_BI_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α