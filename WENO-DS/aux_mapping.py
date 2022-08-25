from aux_base import dtype,B
import numpy as np

"""
Obtendo matrizes de constantes convenintes para executar o WENO-Z
utilizando operações tensoriais, uma vez que permite a integração
com o tensorflow
"""

null_mapping = lambda λ, API, map_function: API.matmul(λ, B)

def post_mapping(λ, API, map_function):

    α    = API.matmul(λ, B1)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)

    return α

def pre_mapping(λ, API, map_function):
    
    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, B1)

    return α

def post_inv_mapping(λ, API, map_function):

    α    = API.matmul(λ, B1)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)
    α    = α * soma

    return α

def pre_inv_mapping(λ, API, map_function):

    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, B1)
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

resolution = 10000
    
def discrete_map(function):
    
    vetor = list(range(0, resolution))
    for i in range(len(vetor)):
        vetor[i] = function(vetor[i]/vetor[-1])
        
    vetor = np.asarray(vetor, dtype=dtype)
    
    return vetor

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

vetor_BI_0 = discrete_map(lambda x: function_BI(x, k=0))
vetor_BI_1 = discrete_map(lambda x: function_BI(x, k=1))
vetor_BI_2 = discrete_map(lambda x: function_BI(x, k=2))

def ENO_mapping(ω, API):
    
    aux_c1 = 1/5
    aux_c2 = aux_c1/2
    aux_c3 = 1-aux_c2
    
    c1 = API.maximum(aux_c1-(ω[...,0:1]+ω[...,2:]), 0)
    c2 = API.maximum(ω[...,0:1]-aux_c3, 0)
    c3 = API.maximum(ω[...,2:] -aux_c3, 0)
    
    c4  = API.maximum(ω[...,0:1]-aux_c2, 0)
    c4 *= API.maximum(ω[...,2:] -aux_c2, 0)
    
    c5  = API.maximum(ω[...,0:1]+ω[...,2:]-aux_c1, 0)
    c5 *= API.maximum(aux_c3-ω[...,0:1], 0)
    c5 *= API.maximum(aux_c3-ω[...,2:] , 0)
    c5 *= API.maximum(aux_c3-ω[...,0:1], 0) + API.maximum(aux_c2-ω[...,2:], 0)
    
    c6 = c5*API.maximum(ω[...,2:] -ω[...,0:1], 0)
    c5 = c5*API.maximum(ω[...,0:1]-ω[...,2:] , 0)
    
    α  = c1*ω # API.constant([0,1,0], dtype = dtype)
    α += c2*ω # API.constant([1,0,0], dtype = dtype)
    α += c3*ω # API.constant([0,0,1], dtype = dtype)
    
    α += c5*API.constant([5/2,5/4,0], dtype = dtype)
    α += c6*API.constant([0,5/6,5/3], dtype = dtype)
    
    α += c4*API.constant([1/3,1/3,1/3], dtype = dtype)
    
    return α

def BI_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BI_0, index[...,0:1])
    ω1 = API.gather(vetor_BI_1, index[...,1:2])
    ω2 = API.gather(vetor_BI_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α

def function_BIP(x, k, p):
    
    # Valor recomendado de p entre -2 e 2
    
    f = lambda x: x**(10**p)
    
    corte1 = 1/10 #       corte1 < 1/3
    corte2 = 7/16 # 1/3 < corte2 < 1/2
    corte3 = 9/10 # 1/2 < corte3
    
    if x < corte1:
        return 1/3 - f(1-(x/corte1))/3
    elif x < corte2:
        return 1/3
    elif x < corte3:
        if k == 0:
            return 2/5
        if k == 1:
            return 1/5
        if k == 2:
            return 2/5
    else:
        if k == 0:
            a = 1-(2/5)
            b = 2/5
        if k == 1:
            a = 1-(1/5)
            b = 1/5
        if k == 2:
            a = 1-(2/5)
            b = 2/5
        return a*f((x-corte3)/(1-corte3)) + b

vetor_BIM_0 = discrete_map(lambda x: function_BIP(x, k=0, p=1))
vetor_BIM_1 = discrete_map(lambda x: function_BIP(x, k=1, p=1))
vetor_BIM_2 = discrete_map(lambda x: function_BIP(x, k=2, p=1))

vetor_BIm_0 = discrete_map(lambda x: function_BIP(x, k=0, p=-2))
vetor_BIm_1 = discrete_map(lambda x: function_BIP(x, k=1, p=-2))
vetor_BIm_2 = discrete_map(lambda x: function_BIP(x, k=2, p=-2))

def BIM_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BIM_0, index[...,0:1])
    ω1 = API.gather(vetor_BIM_1, index[...,1:2])
    ω2 = API.gather(vetor_BIM_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α

def BIm_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BIm_0, index[...,0:1])
    ω1 = API.gather(vetor_BIm_1, index[...,1:2])
    ω2 = API.gather(vetor_BIm_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α

def function_BI2(x, k):
    
    if x < 1/10:
        return x
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
        return x

vetor_BI2_0 = discrete_map(lambda x: function_BI2(x, k=0))
vetor_BI2_1 = discrete_map(lambda x: function_BI2(x, k=1))
vetor_BI2_2 = discrete_map(lambda x: function_BI2(x, k=2))

def BI2_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BI2_0, index[...,0:1])
    ω1 = API.gather(vetor_BI2_1, index[...,1:2])
    ω2 = API.gather(vetor_BI2_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α