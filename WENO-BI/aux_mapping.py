null_mapping = lambda λ, API, map_function: API.matmul(λ, B)
    
def post_mapping(λ, API, map_function):

    α    = API.matmul(λ, B)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)

    return α

def pre_mapping(λ, API, map_function):

    soma = API.sum(λ, axis=-1, keepdims=True)
    ω    = λ / soma
    α    = map_function(ω, API)
    α    = API.matmul(α, B)

    return α

def post_inv_mapping(λ, API, map_function):

    α    = API.matmul(λ, B)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    α    = map_function(ω, API)
    α    = α * soma

    return α

def pre_inv_mapping(λ, API, map_function):

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
    
#     xi  = [0, 1/3, 1/2, 1]
#     aux1 =  tf.math.tan(pi*4/10)
#     aux2 =  tf.math.tan(pi*2/6)
    
#     if k == 0:
#         yi = [[0, aux1, 0], [1/3, 0, 0], [(2/5)*(5/3), 0, 0], [1, aux2, 0]]
#     if k == 1:
#         yi = [[0, aux1, 0], [1/3, 0, 0], [(1/5)*(5/3), 0, 0], [1, aux2, 0]]
#     if k == 2:
#         yi = [[0, aux1, 0], [1/3, 0, 0], [(2/5)*(5/3), 0, 0], [1, aux2, 0]]
        
#     f = BPoly.from_derivatives(
#         xi = xi,
#         yi = yi
#     )
    
#     return f(x)
    
    if x < 1/10:
#         return 0
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
#         return 1
        return Henrick_function(x, 1/3)

#     if x > 1/10 and x < 9/10:
#         return 1/3
#     else:
#         return Henrick_function(x, 1/3)

resolution = 10000
    
def discrete_map(function):
    
    vetor = list(range(0, resolution))
    for i in range(len(vetor)):
        vetor[i] = function(vetor[i]/vetor[-1])
        
    vetor = tf.constant(vetor, dtype=float_pres)
    
    return vetor

vetor_BI_0 = discrete_map(lambda x: function_BI(x, k=0))
vetor_BI_1 = discrete_map(lambda x: function_BI(x, k=1))
vetor_BI_2 = discrete_map(lambda x: function_BI(x, k=2))

# vetor_BI_0 = function_BI(tf.range(0.0, resolution)/resolution, k=0)
# vetor_BI_1 = function_BI(tf.range(0.0, resolution)/resolution, k=1)
# vetor_BI_2 = function_BI(tf.range(0.0, resolution)/resolution, k=2)

def BI_mapping(ω, API):
    
    index = API.cast(API.floor(ω*(resolution-1)), dtype='int32')
    
    ω0 = API.gather(vetor_BI_0, index[...,0:1])
    ω1 = API.gather(vetor_BI_1, index[...,1:2])
    ω2 = API.gather(vetor_BI_2, index[...,2:])
    
    α  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return α