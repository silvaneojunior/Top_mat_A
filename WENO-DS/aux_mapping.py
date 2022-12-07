from aux_base import dtype
import numpy as np

"""
Obtendo matrizes de constantes convenintes para executar o WENO-Z
utilizando operações tensoriais, uma vez que permite a integração
com o tensorflow
"""

null_mapping = lambda λ, API, map_function, d: API.matmul(λ, d)
    
def post_mapping(λ, API, map_function, d):

    α    = API.matmul(λ, d)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)

    return α

def pre_mapping(λ, API, map_function, d):
    
    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, d)

    return α

def post_inv_mapping(λ, API, map_function, d):

    α    = API.matmul(λ, d)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)
    α    = α * soma

    return α

def pre_inv_mapping(λ, API, map_function, d):

    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, d)
    α    = α * soma

    return α

# Mapeamentos para esquemas com RK explícito
#----------------------------------------------------
null_mapping_RK = lambda λ, API, map_function: λ

def pre_mapping_RK(λ, API, map_function):
    
    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)

    return α
#----------------------------------------------------

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
        return x # Henrick_function(x, 1/3)
    elif x < 7/16:
        return 1/3
    elif x < 9/10:
        if k == 0:
            return 2/5 # Valor original: 2/5
        if k == 1:
            return 1/5 # Valor original: 1/5
        if k == 2:
            return 2/5 # Valor original: 2/5
    else:
        return x # Henrick_function(x, 1/3)

vetor_BI_0 = discrete_map(lambda x: function_BI(x, k=0))
vetor_BI_1 = discrete_map(lambda x: function_BI(x, k=1))
vetor_BI_2 = discrete_map(lambda x: function_BI(x, k=2))

def ENO_mapping(ω, API):
    
#     aux = 1/11
    
#     c0_plus = API.maximum(ω[...,0:1] - aux + 10e-6, 0)
#     c1_plus = API.maximum(ω[...,1:2] - aux + 10e-6, 0)
#     c2_plus = API.maximum(ω[...,2:]  - aux + 10e-6, 0)
    
#     c0_minus = API.maximum(aux - ω[...,0:1], 0)
#     c1_minus = API.maximum(aux - ω[...,1:2], 0)
#     c2_minus = API.maximum(aux - ω[...,2:] , 0)
    
#     α  = c0_plus*c2_plus*API.constant([1,1,1], dtype = dtype)
    
#     α += c0_plus*c1_plus*c2_minus*API.constant([2,1,0], dtype = dtype)
#     α += c0_minus*c1_plus*c2_plus*API.constant([0,1,2], dtype = dtype)
    
#     α += c0_plus*c1_minus*c2_minus*ω # API.constant([1,0,0], dtype = dtype)
#     α += c0_minus*c1_plus*c2_minus*API.constant([1,1,1], dtype = dtype)
#     α += c0_minus*c1_minus*c2_plus*ω # API.constant([0,0,1], dtype = dtype)
    
    aux = 1/11
    
    c1_plus  = API.maximum(ω[...,1:2] - aux + 10e-6, 0)
    c1_minus = API.maximum(aux - ω[...,1:2], 0)
    
    α  = c1_plus*API.constant([1,1,1], dtype = dtype)
    α += c1_minus*ω
    
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
            return 2/5 # valor original: 2/5
        if k == 1:
            return 1/5 # valor original: 1/5
        if k == 2:
            return 2/5 # valor original: 2/5
    else:
        if k == 0:
            a = 1-(2/5) # valor original: 1-(2/5)
            b = 2/5     # valor original: 2/5
        if k == 1:
            a = 1-(1/5) # valor original: 1-(1/5)
            b = 1/5     # valor original: 1/5
        if k == 2:
            a = 1-(2/5) # valor original: 1-(2/5)
            b = 2/5     # valor original: 2/5
        return a*f((x-corte3)/(1-corte3)) + b
    
def function_BIm(x, k):
    
    if k == 0 or k == 2:
        
        if x < 1/10:
            return 0 # Henrick_function(x, 1/3)
        elif x < 7/16:
            return 1/3
        elif x < 9/10:
            return 2/5 # Valor original: 2/5
        else:
            return 1 # Henrick_function(x, 1/3)
        
    if k == 1:
    
        if x < 1/10:
            return 0 # Henrick_function(x, 1/3)
        elif x < 7/16:
            return 1/3
        elif x < 9/10:
            return 1/5 # Valor original: 2/5
        else:
            return 1 # Henrick_function(x, 1/3)

vetor_BIm_0 = discrete_map(lambda x: function_BIP(x, k=0, p=-2))
vetor_BIm_1 = discrete_map(lambda x: function_BIP(x, k=1, p=-2))
vetor_BIm_2 = discrete_map(lambda x: function_BIP(x, k=2, p=-2))

# vetor_BIm_0 = discrete_map(lambda x: function_BIm(x, k=0))
# vetor_BIm_1 = discrete_map(lambda x: function_BIm(x, k=1))
# vetor_BIm_2 = discrete_map(lambda x: function_BIm(x, k=2))

def BIm_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BIm_0, index[...,0:1])
    ω1 = API.gather(vetor_BIm_1, index[...,1:2])
    ω2 = API.gather(vetor_BIm_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α

function_IM = lambda ω, d, k, A: d + ((ω-d)**(k+1)*A)/((ω-d)**k*A + ω*(1-ω))

def IM_mapping(ω, API):

    α = function_IM(ω, 1/3, 2, 0.1)
    
    return α

# def function_BI1(x, k, p):
    
#     # Valor recomendado de p >= 1
    
#     def f(x, c1, c2, c3):
#         # Cruva de Bèzier com:
#         # P0 = ( 0,  0)
#         # P1 = (c1, c3)
#         # P2 = (c2, c3)
#         t = (corte1 - np.sqrt(c1**2-(2*c1-c2)*x))/(2*c1-c2)
#         y = (2*(1-t)*t + t**2)*c3
#         return y
    
#     fator     = 9/40
#     interior1 = 1/3*fator # valor original: 1/3
#     interior2 = 1/3*fator # valor original: 1/5
    
#     if k == 0 or k == 2:
    
#         corte1 = 3/40  #       corte1 < 1/3
#         corte2 = 19/40 # 1/3 < corte2 < 1/2
#         corte3 = 1/10  # corte1 < corte3 < corte2

#         if x < corte1:
#             return f(x, corte1, corte3, interior1)
#         elif x < corte2:
#             return interior1
#         else:
#             return 2*interior2
# #         else:
# #             a = 2*interior2-1
# #             b = 1
# #             return a*f((1-x)/(1-corte3)) + 1
# #         else:
# #             a = 1-2*interior2
# #             b = 2*interior2
# #             return a*f((x-corte3)/(1-corte3)) + b
        
#     if k == 1:
        
#         corte1 = 3/40  #       corte1 < 1/3
#         corte2 = 19/40 # 1/3 < corte2 < 1/2
#         corte3 = 1/10  # corte1 < corte3 < corte2

#         if x < corte1:
#             return f(x, corte1, corte3, interior1)
#         elif x < corte2:
#             return interior1
#         else:
#             return interior2 
# #         else:
# #             a = interior2-1
# #             b = 1
# #             return a*f((1-x)/(1-corte3)) + 1
# #         else:
# #             a = 1-interior2
# #             b = interior2
# #             return a*f((x-corte3)/(1-corte3)) + b
        
def function_BI2(x, k):
    
    # Valor recomendado de p >= 1
    
#     def f(x, c1, c2, c3):
#         # Cruva de Bèzier com:
#         # P0 = ( 0,  0)
#         # P1 = (c1, c3)
#         # P2 = (c2, c3)
#         t = (corte1 - np.sqrt(c1**2-(2*c1-c2)*x))/(2*c1-c2)
#         y = (2*(1-t)*t + t**2)*c3
#         return y
    
    fator     = 9/40
    interior1 = (1/3)*fator # valor original: 1/3
    interior2 = (1/3)*fator # valor original: 1/5
    
    if k == 0 or k == 2:
    
        corte1 =  3/40 #          corte1 < corte2
        corte2 =  1/10 # corte1 < corte2 < 1/3
        corte3 = 19/40 # corte2 < corte3 < 1/2
        corte4 = 37/40 # 1/2    < corte4
        
        if x < corte2:
            return x # f(x, corte1, corte2, interior1)
        elif x < corte3:
            return interior1
        elif x <= corte4:
            return 2*interior2
        else:
            return x
#         else:
#             a = 2*interior2-1
#             b = 1
#             return a*f((1-x)/(1-corte3)) + 1
#         else:
#             a = 1-2*interior2
#             b = 2*interior2
#             return a*f((x-corte3)/(1-corte3)) + b
        
    if k == 1:
        
        corte1 =  3/40 #          corte1 < corte2
        corte2 =  1/10 # corte1 < corte2 < 1/3
        corte3 = 19/40 # corte2 < corte3 < 1/2
        corte4 = 37/40 # 1/2    < corte4
        
        if x < corte2:
            return x # f(x, corte1, corte2, interior1)
        elif x < corte3:
            return interior1
        elif x <= corte4:
            return interior2
        else:
            return x
#         else:
#             a = interior2-1
#             b = 1
#             return a*f((1-x)/(1-corte3)) + 1
#         else:
#             a = 1-interior2
#             b = interior2
#             return a*f((x-corte3)/(1-corte3)) + b
        
# def function_BI3(x, k, p):
    
#     # Valor recomendado de p >= 1
    
#     f = lambda x: x**p
    
#     fator     = 9/40
#     interior1 = 1/3*fator # valor original: 1/3
#     interior2 = 1/3*fator # valor original: 1/5
    
#     if k == 0 or k == 2:
    
#         corte1 = 3/40  #       corte1 < 1/3
#         corte2 = 19/40 # 1/3 < corte2 < 1/2
#         corte3 = 1     # 1/2 < corte3

#         if x < corte1:
#             return f(x/corte1)*interior1
#         elif x < corte2:
#             return interior1
#         else:
#             return 2*interior2
# #         else:
# #             a = 2*interior2-1
# #             b = 1
# #             return a*f((1-x)/(1-corte3)) + 1
# #         else:
# #             a = 1-2*interior2
# #             b = 2*interior2
# #             return a*f((x-corte3)/(1-corte3)) + b
        
#     if k == 1:
        
#         corte1 = 3/40  #       corte1 < 1/3
#         corte2 = 19/40 # 1/3 < corte2 < 1/2
#         corte3 = 1     # 1/2 < corte3

#         if x < corte1:
#             return f(x/corte1)*interior1
#         elif x < corte2:
#             return interior1
#         else:
#             return interior2 
# #         else:
# #             a = interior2-1
# #             b = 1
# #             return a*f((1-x)/(1-corte3)) + 1
# #         else:
# #             a = 1-interior2
# #             b = interior2
# #             return a*f((x-corte3)/(1-corte3)) + b

# def function_BI4(x, k, p):
    
#     # Valor recomendado de p >= 1
    
#     f = lambda x: x**p
    
#     fator     = 9/40
#     interior1 = 1/3*fator # valor original: 1/3
#     interior2 = 1/3*fator # valor original: 1/5
    
#     if k == 0 or k == 2:
    
#         corte1 = 3/40  #       corte1 < 1/3
#         corte2 = 19/40 # 1/3 < corte2 < 1/2
#         corte3 = 1     # 1/2 < corte3

#         if x < corte1:
#             return f(x/corte1)*interior1
#         elif x < corte2:
#             return interior1
#         else:
#             return 2*interior2
# #         else:
# #             a = 2*interior2-1
# #             b = 1
# #             return a*f((1-x)/(1-corte3)) + 1
# #         else:
# #             a = 1-2*interior2
# #             b = 2*interior2
# #             return a*f((x-corte3)/(1-corte3)) + b
        
#     if k == 1:
        
#         corte1 = 3/40  #       corte1 < 1/3
#         corte2 = 19/40 # 1/3 < corte2 < 1/2
#         corte3 = 1     # 1/2 < corte3

#         if x < corte1:
#             return f(x/corte1)*interior1
#         elif x < corte2:
#             return interior1
#         else:
#             return interior2 
# #         else:
# #             a = interior2-1
# #             b = 1
# #             return a*f((1-x)/(1-corte3)) + 1
# #         else:
# #             a = 1-interior2
# #             b = interior2
# #             return a*f((x-corte3)/(1-corte3)) + b

# vetor_BI1_0 = discrete_map(lambda x: function_BI1(x, k=0, p=1))
# vetor_BI1_1 = discrete_map(lambda x: function_BI1(x, k=1, p=1))
# vetor_BI1_2 = discrete_map(lambda x: function_BI1(x, k=2, p=1))

vetor_BI2_0 = discrete_map(lambda x: function_BI2(x, k=0))
vetor_BI2_1 = discrete_map(lambda x: function_BI2(x, k=1))
vetor_BI2_2 = discrete_map(lambda x: function_BI2(x, k=2))

# vetor_BI3_0 = discrete_map(lambda x: function_BI3(x, k=0, p=1))
# vetor_BI3_1 = discrete_map(lambda x: function_BI3(x, k=1, p=1))
# vetor_BI3_2 = discrete_map(lambda x: function_BI3(x, k=2, p=1))
        
# vetor_BI4_0 = discrete_map(lambda x: function_BI4(x, k=0, p=1))
# vetor_BI4_1 = discrete_map(lambda x: function_BI4(x, k=1, p=1))
# vetor_BI4_2 = discrete_map(lambda x: function_BI4(x, k=2, p=1))

# def BI1_mapping(ω, API):
    
#     index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
#     ω0 = API.gather(vetor_BI1_0, index[...,0:1])
#     ω1 = API.gather(vetor_BI1_1, index[...,1:2])
#     ω2 = API.gather(vetor_BI1_2, index[...,2:])
    
#     α  = API.concat([ω0, ω1, ω2], axis = -1)
    
#     return α

def BI2_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BI2_0, index[...,0:1])
    ω1 = API.gather(vetor_BI2_1, index[...,1:2])
    ω2 = API.gather(vetor_BI2_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α

# def BI3_mapping(ω, API):
    
#     index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
#     ω0 = API.gather(vetor_BI3_0, index[...,0:1])
#     ω1 = API.gather(vetor_BI3_1, index[...,1:2])
#     ω2 = API.gather(vetor_BI3_2, index[...,2:])
    
#     α  = API.concat([ω0, ω1, ω2], axis = -1)
    
#     return α

# def BI4_mapping(ω, API):
    
#     index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
#     ω0 = API.gather(vetor_BI4_0, index[...,0:1])
#     ω1 = API.gather(vetor_BI4_1, index[...,1:2])
#     ω2 = API.gather(vetor_BI4_2, index[...,2:])
    
#     α  = API.concat([ω0, ω1, ω2], axis = -1)
    
#     return α