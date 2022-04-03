# Código para implementar o WENO-Z utilizando o tensorflow
# Importando os módulos que serão utilizados

import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from numpy import pi

import API_Numpy
import API_TensorFlow

float_pres='float64' # Definindo a precisão padrão para as análises

"""
Obtendo matrizes de constantes convenintes para executar o WENO-Z
utilizando operações tensoriais, uma vez que permite a integração
com o tensorflow
"""
ɛ = 10.0**(-40)

a = np.asarray([[0.5],[-2],[3/2],[0],[0]])
b = np.asarray([[1],[-2],[1],[0],[0]])
c = 13/12

A0 = np.dot(a,a.T)+c*np.dot(b,b.T) # Primiera matriz A

a = np.asarray([[0],[-0.5],[0],[0.5],[0]])
b = np.asarray([[0],[1],[-2],[1],[0]])
c = 13/12

A1 = np.dot(a,a.T)+c*np.dot(b,b.T) # Segunda matriz A

a = np.asarray([[0],[0],[-3/2],[2],[-1/2]])
b = np.asarray([[0],[0],[1],[-2],[1]])
c = 13/12

A2 = np.dot(a,a.T)+c*np.dot(b,b.T) # Terceira matriz A

# Empilhando as matrizes A em um único tensor
A = np.stack([A0,A1,A2], axis=0).astype(dtype=float_pres)
A = np.expand_dims(A, axis=0)

B = np.asarray([[1,0,0],[0,6,0],[0,0,3]], dtype=float_pres)/10                # Matriz B
C = np.asarray([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]], dtype=float_pres)/6 # Matriz C
C = np.transpose(C)



def FronteiraFixa(U, API):
    """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    fixa, repetindo os valores nos extremos da malha
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = API.concat([
        U[...,0:1],
        U[...,0:1],
        U[...,0:1],
        U,
        U[...,-1:],
        U[...,-1:],
        U[...,-1:]],
        axis=-1)
    return U

def FronteiraPeriodica(U, API):
    """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    periódica, continuado os valores de acordo com os extremos opostos
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = API.concat([U[...,-3:], U, U[...,:3]], axis=-1)
    return U



def transp_equation(U_full, API):
    # Setup para equação do transporte
    M = 1                           # Valor utilizado para realizar a separação de fluxo
    f_plus  = (U_full + M*U_full)/2 # Fluxo positivo
    f_minus = (U_full - M*U_full)/2 # Fluxo negativo
    return f_plus, f_minus
    
def burgers_equation(U_full, API):
    M = API.max(API.abs(U_full), axis=-1, keepdims=True) # Valor utilizado para realizar a separação de fluxo
    f_plus  = (U_full**2/2 + M*U_full) / 2 # Fluxo positivo
    f_minus = (U_full**2/2 - M*U_full) / 2 # Fluxo negativo
    return f_plus, f_minus

def diff_equation(U_full, API):
    f_plus  = U_full / 2 # Fluxo positivo
    f_minus = U_full / 2 # Fluxo negativo
    return f_plus, f_minus



def WENO_JS(β, δ, API, mapa):
    β = β*(δ+0.1)
    # Calcula os pesos do WENO-JS
    α    = ((1/(β + ɛ))**2) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    
    return α, ω

def WENO_Z(β, δ, API, mapa):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    β = β*(δ+0.1)

    # Calcula os pesos do WENO-Z
    α    = (1 + (τ/(β + ɛ))**2) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    
    return α, ω

def WENO_Z_plus(β, δ, API, mapa):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    gamma=(τ + ɛ)/(β + ɛ)
    α    = (1 + gamma**2+δ/gamma) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    
    return α, ω

def WENO_JS_M(β, δ, API, mapa):
    β = β*(δ+0.1)
    # Calcula os pesos do WENO-JS
    α    = ((1/(β + ɛ))**2) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    ω     = mapa(ω, API)
    soma  = API.sum(ω, axis=-1, keepdims=True)
    ω     = ω / soma
    
    return α, ω

def WENO_Z_M(β, δ, API, mapa):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    β = β*(δ+0.1)

    # Calcula os pesos do WENO-Z
    α    = (1 + (τ/(β + ɛ))**2) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    ω    = mapa(ω, API)
    soma  = API.sum(ω, axis=-1, keepdims=True)
    ω     = ω / soma
    
    return α, ω

def WENO_Z_plus_M(β, δ, API, mapa):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    gamma = (τ + ɛ)/(β + ɛ)
    α     = (1 + gamma**2 + δ/gamma) @ B
    soma  = API.sum(α, axis=-1, keepdims=True)
    ω     = α / soma
    ω     = mapa(ω, API)
    soma  = API.sum(ω, axis=-1, keepdims=True)
    ω     = ω / soma
    
    return α, ω

def WENO_JS_MS(β, δ, API, mapa):
    β = β*(δ+0.1)
    # Calcula os pesos do WENO-JS
    α    = ((1/(β + ɛ))**2)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω     = α / soma
    ω     = mapa(ω, API) @ B
    soma  = API.sum(ω, axis=-1, keepdims=True)
    ω     = ω / soma
    
    return α, ω

def WENO_Z_MS(β, δ, API, mapa):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    β = β*(δ+0.1)

    # Calcula os pesos do WENO-Z
    α    = (1 + (τ/(β + ɛ))**2)
    soma = API.sum(α, axis=-1, keepdims=True)
    ω     = α / soma
    ω     = mapa(ω, API) @ B
    soma  = API.sum(ω, axis=-1, keepdims=True)
    ω     = ω / soma
    
    return α, ω

def WENO_Z_plus_MS(β, δ, API, mapa):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    gamma = (τ + ɛ)/(β + ɛ)
    α     = (1 + gamma**2 + δ/gamma)
    soma  = API.sum(α, axis=-1, keepdims=True)
    ω     = α / soma
    ω     = mapa(ω, API) @ B
    soma  = API.sum(ω, axis=-1, keepdims=True)
    ω     = ω / soma
    
    return α, ω

def mapa_Henrick(ω, API):
    
    d = [0.1, 0.6, 0.3]
    
    ω0 = ω[...,0:1]
    ω1 = ω[...,1:2]
    ω2 = ω[...,2:]
    
    ω0 = ω0*(d[0] + d[0]**2 - 3*d[0]*ω0 + ω0**2)/(d[0]**2 + ω0*(1-2*d[0]))
    ω1 = ω1*(d[1] + d[1]**2 - 3*d[1]*ω1 + ω1**2)/(d[1]**2 + ω1*(1-2*d[1]))
    ω2 = ω2*(d[2] + d[2]**2 - 3*d[2]*ω2 + ω2**2)/(d[2]**2 + ω2*(1-2*d[2]))
    
    ω = API.concat([ω0, ω1, ω2], axis = -1)
    
    return ω

def mapa_Hong(ω, API):
    d = 1/3
    ω = ω*(d + d**2 - 3*d*ω + ω**2)/(d**2 + ω*(1-2*d))
    return ω

def function_BI(x, k):
    if x < 1/4:
        return x
    elif x < 5/12:
        return 1/3
    elif x < 2/3:
        if k == 0:
            return 5/2
        if k == 1:
            return 5/4
        if k == 2:
            return 5/2
    else:
        return x

resolution = 10000
    
def discrete_map(function):
    
    vetor = list(range(0, resolution))
    for i in range(len(vetor)):
        vetor[i] = function(vetor[i]/vetor[-1])
        
    vetor = tf.constant(vetor, dtype=float_pres)
    
    return vetor

vetor_BI_0 = discrete_map(lambda x: function_BI(x, k=0))
vetor_BI_1 = discrete_map(lambda x: function_BI(x, k=0))
vetor_BI_2 = discrete_map(lambda x: function_BI(x, k=0))

def mapa_BI(ω, API):
    
    index  = API.floor(ω*(resolution+1))
    
    ω0 = API.gather(vetor_BI_0, index[...,0:1])
    ω1 = API.gather(vetor_BI_1, index[...,1:2])
    ω2 = API.gather(vetor_BI_2, index[...,2:])
    
    ω  = API.concat([ω0, ω1, ω2], axis = -1)
    
    return ω

def create_simulation(API, flux_calc, WENO, mapa=lambda x:x, network=None, compile_flag=True):
    
    def Sim(u, t_final, Δx, CFL, fronteira):    
        t = API.max(API.abs(u), axis=-1, keepdims=True)*0 # Instante de tempo incial para a computação

        while API.any(t < t_final):
            Λ  = API.max(API.abs(u), axis=-1, keepdims=True)

            Δt = Δx*CFL/Λ  
            Δt = API.where(t + Δt > t_final, t_final - t, Δt)

            u = Sim_step(u, Δt, Δx, fronteira)
            
            t = t + Δt # Avançando no tempo
            
        return u
    
    def Sim_step(u, Δt, Δx, fronteira):
        u1 = u - Δt*DerivadaEspacial(u, Δx, fronteira)
        u2 = (3*u +   u1 -   Δt*DerivadaEspacial(u1, Δx, fronteira)) / 4.0
        u  = (  u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, fronteira)) / 3.0
        return u

    def DerivadaEspacial(U, Δx, AdicionaGhostPoints):
        """
        Calcula a derivada espacial numa malha de pontos utilizando o WENO-Z
        ---------------------------------------------------------------------------------
        U                     (tensor): valores da função para o cálcula da derivada
        Δx                     (float): distância espacial dos pontos na malha utilizada
        AdicionaGhostPoints (function): função que adicionada pontos na malha de acordo 
                                        com a condição de fronteira
        network             (function): função que computa o valor de saída da rede 
                                        neural a partir do valor de entrada
        ---------------------------------------------------------------------------------
        Fdif                  (tensor): derivada espacial
        ---------------------------------------------------------------------------------
        """
        U = AdicionaGhostPoints(U,API) # Estende a malha de pontos de acordo com as condições de fronteira
        U = slicer(U,6,API)

        f_plus,f_minus = flux_calc(U,API)

        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = ReconstructionMinus(f_plus[...,:-1], network) 
        f_half_plus  = ReconstructionPlus( f_minus[...,1:], network)
        Fhat         = (f_half_minus + f_half_plus)

        # Calculando uma estimava da derivada a partir de diferenças finitas
        Fhat = (Fhat[...,1:] - Fhat[...,:-1]) / Δx

        return Fhat


    def ReconstructionMinus(u0, network):
        """
        Calcula o WENO modificado para obter uma estimativa da função utilizando um 
        subestêncil de cinco pontos
        --------------------------------------------------------------------------------
        u0          (tensor): tensor com o substêncil de 5 pontos
        beta_weight (tensor): modificadores dos indicadores de suavidade
        --------------------------------------------------------------------------------
        fhat        (tensor): estimativa da função
        --------------------------------------------------------------------------------
        """
        
        if network is not None:
            δ = network(API.concat([u0[...,0:1,0], u0[...,0:1,1], u0[...,2], u0[...,-1:,-2], u0[...,-1:,-1]], axis=-1))
            δ = slicer(δ, 3, API)
        else:
            δ = 1-0.1
        # Calcula os indicadores de suavidade locais
        u = API.repeat(API.expand_dims(u0, axis=-2), 3, axis=-2)
        u = API.expand_dims(u, axis=-2)

        β = API.sum((u * (u @ A))[...,0,:], axis=-1)
        ω = WENO(β, δ, API, mapa)[1]
        
        # Calcula os fhat em cada subestêncil
        fhat = u0 @ C
        
        # Calcula o fhat do estêncil todo
        fhat = API.sum(ω * fhat, axis=-1)

        return fhat

    def ReconstructionPlus(u0, network):
        """
        Calcula o WENO modificado para obter uma estimativa da função utilizando um 
        subestêncil de cinco pontos
        --------------------------------------------------------------------------------
        u0          (tensor): tensor com o substêncil de 5 pontos
        beta_weight (tensor): modificadores dos indicadores de suavidade
        --------------------------------------------------------------------------------
        fhat        (tensor): estimativa da função
        --------------------------------------------------------------------------------
        """
        # Reciclando a função WenoZ5ReconstructionMinus
        fhat = ReconstructionMinus(API.reverse(u0, axis=[-1]), network)
        return fhat
    

    def Get_weights(U, AdicionaGhostPoints):

        U = AdicionaGhostPoints(U, API)

        if network is not None:
            δ = network(U)
            δ = slicer(δ, 3, API)
        else:
            δ = 1-0.1
        
        U = slicer(U, 5, API)

        # Calcula os indicadores de suavidade locais
        u = API.repeat(API.expand_dims(U, axis=-2), 3, axis=-2)
        u = API.expand_dims(u, axis=-2)

        β = API.sum((u * (u @ A))[...,0,:], axis=-1)
        α, ω = WENO(β, δ, API, mapa)
        
        return ω, α, β, δ
    
    if compile_flag:
        return API.function(Sim), API.function(Sim_step), API.function(DerivadaEspacial), API.function(Get_weights)
    else:
        return Sim, Sim_step, DerivadaEspacial, Get_weights
    
class WENO(k.layers.Layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,flux_calc,WENO_method):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO, self).__init__(dtype=float_pres) # Chamando o inicializador da superclasse
        self.Sim, self.Sim_step, self.DerivadaEspacial, self.Get_weights=create_simulation(API_TensorFlow, flux_calc, WENO_method, network=self.network_graph, compile_flag=False)
        
    def build(self, input_shape):
        """
        Função para compor as camadas que constituem essa camada da rede neural
        ------------------------------------------------------------------------
        input_shape : não é utilizado por essa função, mas é um argumento obrigatório para camadas do Keras.
        ------------------------------------------------------------------------
        """
        self.layers = []
        wei_reg = k.regularizers.L2(0*10**-3)                                                                         # Regularização dos pesos da rede 
        self.layers.append(k.layers.ZeroPadding1D(padding=1))                                                         # Camada de padding de zeros em 1 dimensão
        self.layers.append(k.layers.Conv1D(5, 3, activation='elu',     dtype=float_pres, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(k.layers.ZeroPadding1D(padding=1))                                                         # Camada de padding de zeros em 1 dimensão
        self.layers.append(k.layers.Conv1D(3, 3, activation='elu',     dtype=float_pres, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(k.layers.Conv1D(1, 1, activation='sigmoid', dtype=float_pres, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(k.layers.Flatten(dtype=float_pres))
        
    def network_graph(self, x):
        """
        Função utilizado para executar sucessivamente as camadas dessa camada 
        da rede neural, passando o input de uma para a próxima
        ----------------------------------------------------------------------
        x (tensor): valor de entrada da rede
        ----------------------------------------------------------------------
        y (tensor): valor de saída da rede
        ----------------------------------------------------------------------
        """
        y = tf.stack([x[...,2:]-x[...,:-2], x[...,2:]-2*x[...,1:-1]+x[...,:-2]], axis=-1)
        # Percorrendo as camadas
        for layer in self.layers:
            
            # Atualizando o valor de entrada para a próxima camada
            y = layer(y)
        return y

class WENO_temporal(WENO):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,Δx,CFL,Δt,fronteira,flux_calc,WENO_method):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_temporal,self).__init__(flux_calc,WENO_method)
        self.Δx  = tf.Variable( Δx, dtype=float_pres, trainable=False)
        self.CFL = tf.Variable(CFL, dtype=float_pres, trainable=False)
        self.Δt  = tf.Variable( Δt, dtype=float_pres, trainable=False)
        self.fronteira = fronteira
        
    def call(self, inpt, mask=None):
        return self.Sim_step(inpt, self.Δt, self.Δx, FronteiraPeriodica)
    
class WENO_espacial(WENO):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self, Δx, fronteira, flux_calc, WENO_method):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_espacial,self).__init__(flux_calc, WENO_method)
        self.Δx = tf.Variable(Δx, dtype=float_pres, trainable=False)
        self.fronteira = fronteira
        
    def call(self, inpt, mask=None):
        return self.DerivadaEspacial(inpt, self.Δx, FronteiraPeriodica)
    
class MES_OF(k.losses.Loss):
    """Criando uma função de custo cuja superclasse é a de funções de
    custo do keras"""
    
    def call(self, y_true, y_pred):
        """
        Função que avalia o custo dado um valor de referência e um valor previsto
        --------------------------------------------------------------------------
        y_true (tensor): valor de referência
        y_pred (tensor): valor predito
        --------------------------------------------------------------------------
        loss   (tensor): custo associado
        --------------------------------------------------------------------------
        """
        y_true = tf.cast(y_true, y_pred.dtype) # Convertendo os tipos para evitar conflitos
        y_min  = tf.math.reduce_min(y_true, axis=1, keepdims=True)
        y_max  = tf.math.reduce_max(y_true, axis=1, keepdims=True)
        
        loss = tf.reduce_mean(
            tf.math.square(y_pred - y_true), axis=-1) + \
            tf.reduce_sum(
                tf.where(y_pred > y_max, y_pred - y_max,  0) + \
                tf.where(y_pred < y_min, y_min  - y_pred, 0),    
            axis=-1)
        
        return loss

class Only_smooth_loss(k.losses.Loss):
    """Criando uma função de custo cuja superclasse é a de funções de
    custo do keras"""
    
    def __init__(self, pre_loss, tol=100):
        super(Only_smooth_loss,self).__init__()
        self.pre_loss = pre_loss
        self.tol = tol

    
    def call(self, y_true, y_pred):
        """
        Função que avalia o custo dado um valor de referência e um valor previsto
        --------------------------------------------------------------------------
        y_true (tensor): valor de referência
        y_pred (tensor): valor predito
        --------------------------------------------------------------------------
        loss   (tensor): custo associado
        --------------------------------------------------------------------------
        """
        y_true = tf.cast(y_true, y_pred.dtype) # Convertendo os tipos para evitar conflitos
        valid_index = tf.where(tf.abs(y_true) < self.tol, 1.0, 0.0)
        valid_index = tf.cast(valid_index, y_pred.dtype) # Convertendo os tipos para evitar conflitos
        
        loss = tf.reduce_mean(valid_index*self.pre_loss(y_pred, y_true)/(tf.abs(y_true)+1), axis=-1)
        
        return loss

def slicer(data, n, API):
    helper = lambda i: data[...,i:i+n]

    data_sliced = API.einsum(
    'i...j -> ...ij',
        API.map_fn(
            helper,                             # Função a ser executada a cada iteração do loop
            API.range(API.shape(data)[-1]-n+1), # Índices utilizados no loop
            fn_output_signature=data.dtype      # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        )
    )

    return data_sliced