# Código para implementar o WENO-Z utilizando o tensorflow
# Importando os módulos que serão utilizados

import tensorflow as tf
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import Null_net

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

def FronteiraFixa(U,API):
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


def FronteiraPeriodica(U,API):
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

def transp_equation(U_full,API):
    # Setup para equação do transporte
    M = 1                           # Valor utilizado para realizar a separação de fluxo
    f_plus  = (U_full + M*U_full)/2 # Fluxo positivo
    f_minus = (U_full - M*U_full)/2 # Fluxo negativo
    return f_plus,f_minus
    
def burgers_equation(U_full,API):
    M = API.max(API.abs(U_full),axis=-1,keepdims=True) # Valor utilizado para realizar a separação de fluxo
    f_plus  = (U_full**2/2 + M*U_full) / 2                      # Fluxo positivo
    f_minus = (U_full**2/2 - M*U_full) / 2                      # Fluxo negativo
    return f_plus,f_minus

def diff_equation(U_full,API):
    f_plus  = U_full / 2                      # Fluxo positivo
    f_minus = U_full / 2                      # Fluxo negativo
    return f_plus,f_minus

def WENO_Z(β,API):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = (1 + (τ/(β + ɛ))**2) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    return ω

def WENO_JS(β,API):
    # Calcula os pesos do WENO-JS
    α    = ((1/(β + ɛ))**2) @ B
    soma = API.sum(α, axis=-1, keepdims=True)
    ω    = α / soma
    return ω

def create_simulation(API,pre_flux_calc,WENO,network=None,compile_flag=True):
    
    flux_calc=lambda x: pre_flux_calc(x,API)
    
    def Sim(u, t_final, Δx, CFL, fronteira):    
        t = API.max(API.abs(u), axis=-1, keepdims=True)*0 # Instante de tempo incial para a computação

        while API.any(t < t_final):
            Λ  = API.max(API.abs(u), axis=-1, keepdims=True)

            Δt = Δx*CFL/Λ  
            Δt = API.where(t + Δt > t_final, t_final - t, Δt)

            u=Sim_step(u, Δt, Δx, fronteira)
            
            t  = t + Δt # Avançando no tempo
        return u
    
    def Sim_step(u, Δt, Δx, fronteira):
        u1 = u - Δt*DerivadaEspacial(u, Δx, fronteira)
        u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, fronteira)) / 4.0
        u  = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, fronteira)) / 3.0
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

        # Funções auxiliares que serão utilizadas dentro de tf.map_fn
        helper1 = lambda i: U[...,i:i+6]

        # Utilizando um loop implícito (com tf.map_fn) para obter cada subestêncil de 
        # 5 pontos e depois concatenar em um único tensor (com tf.concat) na primeira
        # dimensão
        U_full = API.stack(
        API.unstack(
            API.map_fn(
                helper1,                    # Função a ser executada a cada iteração do loop
                API.range(tf.shape(U)[-1]-5),     # Índices utilizados no loop
                fn_output_signature=U.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
            ), axis=0),
            axis=-2
        )

        if network is not None:
            beta_weight = network(U)    # Passa U_diff como input para a rede salva o output
            helper2 = lambda i: beta_weight[...,i:i+3]
            beta_weight_full = API.stack(
            API.unstack(
                API.map_fn(
                    helper2,                              # Função a ser executada a cada iteração do loop
                    API.range(tf.shape(beta_weight)[-1]-2),     # Índices utilizados no loop
                    fn_output_signature=beta_weight.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
                ), axis=0),
                axis=-2
            )
        else:
            beta_weight_full=API.ones([1,2,1],dtype=float_pres)

        f_plus,f_minus=flux_calc(U_full)

        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = ReconstructionMinus(f_plus[...,:-1], beta_weight_full[...,:-1,:]) 
        f_half_plus  = ReconstructionPlus( f_minus[...,1:], beta_weight_full[...,1:,:])
        Fhat         = (f_half_minus + f_half_plus)

        # Calculando uma estimava da derivada a partir de diferenças finitas
        Fhat = (Fhat[...,1:] - Fhat[...,:-1]) / Δx

        return Fhat



    def ReconstructionMinus(u0, beta_weight):
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
        
        # Calcula os indicadores de suavidade locais
        u = API.repeat(API.expand_dims(u0,axis=-2),3, axis=-2)
        u = API.expand_dims(u,axis=-2)

        β = API.sum((u * (u @ A))[...,0,:], axis=-1)
        β = β*(beta_weight+0.1)
        ω = WENO(β,API)
        
        # Calcula os fhat em cada subestêncil
        fhat = u0 @ C
        
        # Calcula o fhat do estêncil todo
        fhat = API.sum(ω * fhat, axis=-1)

        return fhat

    def ReconstructionPlus(u0, beta_weight):
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
        fhat = ReconstructionMinus(API.reverse(u0,axis=[-1]), API.reverse(beta_weight,axis=[-2,-1]))
        return fhat
    
    def Get_weights(U,AdicionaGhostPoints):
        U = AdicionaGhostPoints(U,API)
        # Funções auxiliares que serão utilizadas dentro de tf.map_fn
        helper1 = lambda i: U[...,i:i+6]

        # Utilizando um loop implícito (com tf.map_fn) para obter cada subestêncil de 
        # 5 pontos e depois concatenar em um único tensor (com tf.concat) na primeira
        # dimensão
        U_full = API.stack(
        API.unstack(
            API.map_fn(
                helper1,                    # Função a ser executada a cada iteração do loop
                API.range(tf.shape(U)[-1]-5),     # Índices utilizados no loop
                fn_output_signature=U.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
            ), axis=0),
            axis=-2
        )

        if network is not None:
            beta_weight = network(U)    # Passa U_diff como input para a rede salva o output
            helper2 = lambda i: beta_weight[...,i:i+3]
            beta_weight_full = API.stack(
            API.unstack(
                API.map_fn(
                    helper2,                              # Função a ser executada a cada iteração do loop
                    API.range(tf.shape(beta_weight)[-1]-2),     # Índices utilizados no loop
                    fn_output_signature=beta_weight.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
                ), axis=0),
                axis=-2
            )
        else:
            beta_weight_full=API.ones([1,2,1],dtype=float_pres)
            
        # Calcula os indicadores de suavidade locais
        u = API.repeat(API.expand_dims(U_full[...,:-1],axis=-2),3, axis=-2)
        u = API.expand_dims(u,axis=-2)

        β = API.sum((u * (u @ A))[...,0,:], axis=-1)
        β = β*(beta_weight_full[...,:-1,:]+0.1)
        ω = WENO(β,API)

        return ω,β
    
    if compile_flag:
        return API.function(Sim), API.function(Sim_step), API.function(DerivadaEspacial), API.function(Get_weights)
    else:
        return Sim, Sim_step, DerivadaEspacial, Get_weights
    
class WENO_temporal(keras.layers.Layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,Δx,CFL,Δt,fronteira):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_layer, self).__init__(dtype='float64') # Chamando o inicializador da superclasse
        self.Δx=tf.Variable(Δx,dtype=float_pres,trainable=False)
        self.CFL=tf.Variable(CFL5,dtype=float_pres,trainable=False)
        self.Δt=tf.Variable(Δt,dtype=float_pres,trainable=False)
        self.fronteira=fronteira
        self.Sim, self.Sim_step, self.DerivadaEspacial, self.Get_weights=create_simulation(API_TensorFlow,burgers_equation,WENO_JS,network=self.network_graph,compile_flag=False)
        
    def build(self, input_shape):
        """
        Função para compor as camadas que constituem essa camada da rede neural
        ------------------------------------------------------------------------
        input_shape : não é utilizado por essa função, mas é um argumento obrigatório para camadas do Keras.
        ------------------------------------------------------------------------
        """
        self.layers = []
        wei_reg = tf.keras.regularizers.L2(0*10**-3)                                                                        # Regularização dos pesos da rede 
        self.layers.append(tf.keras.layers.ZeroPadding1D(padding=2))                                                        # Camada de padding de zeros em 1 dimensão
        self.layers.append(keras.layers.Conv1D(5, 5, activation='elu',     dtype=data_x.dtype, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(tf.keras.layers.ZeroPadding1D(padding=2))                                                        # Camada de padding de zeros em 1 dimensão
        self.layers.append(keras.layers.Conv1D(3, 5, activation='elu',     dtype=data_x.dtype, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(keras.layers.Conv1D(1, 1, activation='sigmoid', dtype=data_x.dtype, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(keras.layers.Flatten(dtype=data_x.dtype))
        
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
        
    def call(self, inpt, mask=None):
        return self.Sim_step(inpt,self.Δt, self.Δx, FronteiraPeriodica)
    
class WENO_espacial(keras.layers.Layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,Δx,fronteira):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_layer, self).__init__(dtype='float64') # Chamando o inicializador da superclasse
        self.Δx=tf.Variable(Δx,dtype=float_pres,trainable=False)
        self.fronteira=fronteira
        self.Sim, self.Sim_step, self.DerivadaEspacial, self.Get_weights=create_simulation(API_TensorFlow,burgers_equation,WENO_JS,network=self.network_graph,compile_flag=False)
        
    def build(self, input_shape):
        """
        Função para compor as camadas que constituem essa camada da rede neural
        ------------------------------------------------------------------------
        input_shape : não é utilizado por essa função, mas é um argumento obrigatório para camadas do Keras.
        ------------------------------------------------------------------------
        """
        self.layers = []
        wei_reg = tf.keras.regularizers.L2(0*10**-3)                                                                        # Regularização dos pesos da rede 
        self.layers.append(tf.keras.layers.ZeroPadding1D(padding=2))                                                        # Camada de padding de zeros em 1 dimensão
        self.layers.append(keras.layers.Conv1D(5, 5, activation='elu',     dtype=data_x.dtype, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(tf.keras.layers.ZeroPadding1D(padding=2))                                                        # Camada de padding de zeros em 1 dimensão
        self.layers.append(keras.layers.Conv1D(3, 5, activation='elu',     dtype=data_x.dtype, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(keras.layers.Conv1D(1, 1, activation='sigmoid', dtype=data_x.dtype, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        self.layers.append(keras.layers.Flatten(dtype=data_x.dtype))
        
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
        
    def call(self, inpt, mask=None):
        return self.DerivadaEspacial(inpt,self.Δx, FronteiraPeriodica)
    
class MES_OF(tf.keras.losses.Loss):
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
        y_min  = tf.math.reduce_min(y_true,axis=1,keepdims=True)
        y_max  = tf.math.reduce_max(y_true,axis=1,keepdims=True)
        
        loss = tf.reduce_mean(
            tf.math.square(y_pred - y_true), axis=-1) + \
            tf.reduce_sum(
                tf.where(y_pred > y_max, y_pred - y_max,  0) + \
                tf.where(y_pred < y_min, y_min  - y_pred, 0),    
            axis=-1)
        
        return loss