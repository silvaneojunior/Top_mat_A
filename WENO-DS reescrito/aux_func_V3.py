# Código para implementar o WENO-Z utilizando o tensorflow
# Importando os módulos que serão utilizados

from sympy import beta
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



def FronteiraFixa(U,API,n=3):
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
        API.repeat(U[...,:1],n,axis=-1),
        U,
        API.repeat(U[...,-1:],n,axis=-1)],
        axis=-1)
    return U

def FronteiraPeriodica(U,API,n=3):
    """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    periódica, continuado os valores de acordo com os extremos opostos
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = API.concat([U[...,-n:], U, U[...,:n]], axis=-1)
    return U

def FronteiraReflexiva(U,API,n=3):
    U0 = API.concat([
        API.flip(U[...,0,:n],axis=[-1]),
        U[...,0,:],
        API.flip(U[...,0,-n:],axis=[-1])],
        axis=-1)

    U1 = API.concat([
        -API.flip(U[...,1,:n],axis=[-1]),
        U[...,1,:],
        -API.flip(U[...,1,-n:],axis=[-1])],
        axis=-1)

    U2 = API.concat([
        API.flip(U[...,2,:n],axis=[-1]),
        U[...,2,:],
        API.flip(U[...,2,-n:],axis=[-1])],
        axis=-1)
    
    U=tf.stack([U0,U1,U2],axis=-2)
    
    return U

class equation:
    def __init__(self,API, WENO, network,p=2):
        self.API=API
        self.WENO=WENO
        self.network=network
        self.p=p
    def ReconstructionMinus(self,u0):
        
        if self.network is not None:
            δ = self.network(self.API.concat([u0[...,0:1,0],u0[...,0:1,1],u0[...,2],u0[...,-1:,-2],u0[...,-1:,-1]],axis=-1))
            δ = slicer(δ,3,self.API)
        else:
            δ=1-0.1
        # Calcula os indicadores de suavidade locais
        u = self.API.repeat(self.API.expand_dims(u0,axis=-2),3, axis=-2)
        u = self.API.expand_dims(u,axis=-2)

        β = self.API.sum((u * self.API.matmul(u, A))[...,0,:], axis=-1)
        α = self.WENO(β,δ,self.API,self.p)
        soma = self.API.sum(α, axis=-1, keepdims=True)
        ω    = α / soma
        
        # Calcula os fhat em cada subestêncil
        fhat = self.API.matmul(u0, C)
        
        # Calcula o fhat do estêncil todo
        fhat = self.API.sum(ω * fhat, axis=-1)

        return fhat

    def ReconstructionPlus(self,u0):
        fhat = self.ReconstructionMinus(self.API.reverse(u0,axis=[-1]))
        return fhat
        
    def flux_sep(self,U):
        pass

    def DerivadaEspacial(self,U, Δx, AdicionaGhostPoints):
        U = AdicionaGhostPoints(U,self.API) # Estende a malha de pontos de acordo com as condições de fronteira

        f_plus,f_minus=self.flux_sep(U)

        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = self.ReconstructionMinus(f_plus[...,:-1]) 
        f_half_plus  = self.ReconstructionPlus( f_minus[...,1:])
        Fhat         = (f_half_minus + f_half_plus)

        # Calculando uma estimava da derivada a partir de diferenças finitas
        Fhat = (Fhat[...,1:] - Fhat[...,:-1]) / Δx

        return Fhat

class transp_equation(equation):
    def maximum_speed(self,U):
        return self.API.cast(1,float_pres)
    def flux_sep(self,U):
        U = slicer(U,6,self.API)
        # Setup para equação do transporte
        M = self.maximum_speed(U)                          # Valor utilizado para realizar a separação de fluxo
        f_plus  = (U + M*U)/2 # Fluxo positivo
        f_minus = (U - M*U)/2 # Fluxo negativo
        return f_plus,f_minus

class burgers_equation(equation):
    def maximum_speed(self,U):
        return self.API.max(self.API.abs(U),axis=-1,keepdims=True)
    def flux_sep(self,U):
        U = slicer(U,6,self.API)
        # Setup para equação do transporte
        M = self.maximum_speed(U)                          # Valor utilizado para realizar a separação de fluxo
        f_plus  = (U**2/2 + M*U) / 2                      # Fluxo positivo
        f_minus = (U**2/2 - M*U) / 2                      # Fluxo negativo
        return f_plus,f_minus

class diff_equation(equation):
    def maximum_speed(self,U):
        return 1
    def flux_sep(self,U):
        U = slicer(U,6,self.API)
        f_plus  = U / 2                      # Fluxo positivo
        f_minus = U / 2                      # Fluxo negativo
        return f_plus,f_minus

# Dummy comment
γ=1.4

class euler_equation(equation):
    def Pressure(self,Q):
        Q0,Q1,Q2=self.API.unstack(Q,axis=-2)
        a=γ-1.0
        b=Q1**2
        c=2*Q0
        d=b/c
        e=Q2-d
        
        return a*e

    def Eigensystem(self,Q):
        Q0,Q1,Q2=self.API.unstack(Q,axis=-2)
        
        U = Q1/Q0
        P = self.Pressure(Q)
        A = self.API.sqrt(γ*P/Q0) # Sound speed
        H = (Q2 + P)/Q0    # Enthalpy
        h = 1/(2*H - U**2)
        
        ones_ref=self.API.ones(self.API.shape(U),dtype=U.dtype)
        R1c=self.API.stack([ones_ref,U-A,H - U*A],axis=-2)
        R2c=self.API.stack([ones_ref,U  ,U**2/2 ],axis=-2)
        R3c=self.API.stack([ones_ref,U+A,H + U*A],axis=-2)
        
        R = self.API.stack([R1c,R2c,R3c],axis=-2)
        
        L1c=self.API.stack([U/(2*A) + U**2*h/2,2 - (2*H)*h,U**2*h/2 - U/(2*A)],axis=-2)
        L2c=self.API.stack([-U*h - 1/(2*A)    ,(2*U)*h    ,1/(2*A) - U*h     ],axis=-2)
        L3c=self.API.stack([h                 ,-2*h       ,h                 ],axis=-2)
        
        L = self.API.stack([L1c,L2c,L3c],axis=-2)
        return R,L

    def Eigenvalues(self,Q):
        Q0,Q1,Q2=self.API.unstack(Q,axis=-2)
        
        U = Q1/Q0
        P = self.Pressure(Q)
        A = self.API.sqrt(γ*P/Q0)

        return self.API.stack([U-A, U, U+A],axis=-2)

    def maximum_speed(self,U):
        eig_val=self.API.abs(self.Eigenvalues(U))
        return self.API.max(self.API.max(eig_val,axis=-1,keepdims=True),axis=-2,keepdims=True)

    def ReconstructedFlux(self, F, Q, M):
        M=self.API.expand_dims(M,axis=-3)
        F_plus  = (F + M*Q)/2
        F_minus = (F - M*Q)/2

        F_plus=self.API.einsum('...ijk->...jik',F_plus)
        F_minus=self.API.einsum('...ijk->...jik',F_minus)
        
        F_half_plus  = self.ReconstructionMinus(F_plus[...,:-1])
        F_half_minus = self.ReconstructionPlus(F_minus[...,1:])

        return F_half_plus + F_half_minus

    def DerivadaEspacial(self,Q, Δx, AdicionaGhostPoints):
        Ord = 5 # The order of the scheme
        Q = AdicionaGhostPoints(Q,self.API)

        #N = Q.shape[1]

        M = self.maximum_speed(Q)
        Qi=slicer(Q,6,self.API)
        Qi=self.API.einsum('...ijk->...jik',Qi)
        
        Λ = self.Eigenvalues(Qi)
        
        r = 2
        Qa=(Qi[...,r] + Qi[...,r+1])/2
        Qa=self.API.einsum('...ij->...ji',Qa)
        R, L = self.Eigensystem(Qa)
            
        W = self.API.einsum('...nvc,...uvn -> ...nuc',Qi,L)       # Transforms into characteristic variables
        G = Λ*W       # The flux for the characteristic variables is Λ * L*Q
        #M = M[2:N-3]
        G_half=self.ReconstructedFlux(G, W, M)
        F_half = self.API.einsum('...vn,...uvn -> ...un',G_half,R)  # Brings back to conservative variables
        return (F_half[...,1:] - F_half[...,:-1])/Δx # Derivative of Flux


def WENO_Z(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    β=β*(δ+0.1)
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    return α

def WENO_Z_alpha(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    α=α*(δ+0.1)
    return α

def WENO_Z_plus(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ =(τ + ɛ)/(β + ɛ)
    α    = API.matmul((1 + γ**p+δ/γ), B)
    return α

def WENO_JS(β,δ,API,p=2):
    β=β*(δ+0.1)
    # Calcula os pesos do WENO-JS
    α    = API.matmul(((1/(β + ɛ))**p), B)
    return α

def WENO_mix(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    #δ=δ[...,1:2]
    β=β*δ+(1-δ)*np.asarray([[1/10,6/10,3/10]])
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    return α

def Continuous_case(β,δ,API,p=2):
    β=β*0+(1-0)*np.asarray([[1/10,6/10,3/10]])
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    return α

class simulation:
    def __init__(self,API,equation_class,WENO,network=None,p=2):
        self.equation=equation_class(API, WENO, network, p)
        self.API=API
        self.WENO=WENO
        self.network=network
        self.p=p

        self.Sim=API.function(self.Sim_graph)
        self.Sim_step=API.function(self.Sim_step_graph)
        self.Get_weights=API.function(self.Get_weights_graph)

        self.DerivadaEspacial_graph=equation.DerivadaEspacial
        self.DerivadaEspacial=API.function(self.DerivadaEspacial_graph)
    def Sim_graph(self,u, t_final, Δx, CFL, fronteira):    
        t = 0.0*self.equation.maximum_speed(u) # Instante de tempo incial para a computação
        while self.API.any(t < t_final):
            Λ  = self.equation.maximum_speed(u)

            Δt = Δx*CFL/Λ  
            Δt = self.API.where(t + Δt > t_final, t_final - t, Δt)

            u=self.Sim_step_graph(u, Δt, Δx, fronteira)
            
            t  = t + Δt # Avançando no tempo
        return u
    def Sim_step_graph(self,u, Δt, Δx, fronteira):
        u1 = u - Δt*self.equation.DerivadaEspacial(u, Δx, fronteira)
        u2 = (3*u + u1 - Δt*self.equation.DerivadaEspacial(u1, Δx, fronteira)) / 4.0
        u  = (u + 2*u2 - 2*Δt*self.equation.DerivadaEspacial(u2, Δx, fronteira)) / 3.0
        return u
    def Get_weights_graph(self,U,AdicionaGhostPoints):

        U = AdicionaGhostPoints(U,self.API)

        if self.network is not None:
            δ = self.network(U)
            δ = slicer(δ,3,self.API)
        else:
            δ=1-0.1
        
        U = slicer(U,5,self.API)

        # Calcula os indicadores de suavidade locais
        u = self.API.repeat(API.expand_dims(U,axis=-2),3, axis=-2)
        u = self.API.expand_dims(u,axis=-2)

        β = self.API.sum((u * API.matmul(u, A))[...,0,:], axis=-1)
        α = self.WENO(β,δ,self.API)
        soma = self.API.sum(α, axis=-1, keepdims=True)
        ω    = α / soma

        return ω,α,β,δ
    
class WENO(k.layers.Layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,flux_calc,WENO_method,conv_size=5,regul_weight=0,p=2,ativ_func=tf.nn.sigmoid):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO, self).__init__(name='WENO_layer',dtype=float_pres) # Chamando o inicializador da superclasse
        self.simulation=simulation(API_TensorFlow,flux_calc,WENO_method,network=self.network_graph,p=p)

        self.Sim, self.Sim_step, self.DerivadaEspacial, self.Get_weights=self.simulation.Sim, self.simulation.Sim_step, self.simulation.DerivadaEspacial, self.simulation.Get_weights
        self.Sim_graph, self.Sim_step_graph, self.DerivadaEspacial_graph, self.Get_weights_graph=self.simulation.Sim_graph, self.simulation.Sim_step_graph, self.simulation.DerivadaEspacial_graph, self.simulation.Get_weights_graph

        self.regul_weight=regul_weight
        self.ativ_func=ativ_func
        if (conv_size-1)%2==0:
            self.conv_size=conv_size
        else:
            raise(ValueError('Invalid conv_size. Expected a odd number, got {}'.format(conv_size)))
    def build(self, input_shape):
        """
        Função para compor as camadas que constituem essa camada da rede neural
        ------------------------------------------------------------------------
        input_shape : não é utilizado por essa função, mas é um argumento obrigatório para camadas do Keras.
        ------------------------------------------------------------------------
        """
        self.layers = []                                                                   # Regularização dos pesos da rede 
        self.layers.append(Conv1D(10, self.conv_size, activation=tf.nn.elu,name='conv1')) # Camada de convolução em 1 dimensão                                                    # Camada de padding de zeros em 1 dimensão
        self.layers.append(Conv1D(10, self.conv_size, activation=tf.nn.elu,name='conv2')) # Camada de convolução em 1 dimensão
        self.layers.append(Conv1D(1, 1, activation=self.ativ_func,name='conv3')) # Camada de convolução em 1 dimensão
        
        # wei_reg = k.regularizers.L2(self.regul_weight)                                                                        # Regularização dos pesos da rede 
        # self.layers.append(k.layers.ZeroPadding1D(padding=(self.conv_size-1)//2))                                                        # Camada de padding de zeros em 1 dimensão
        # self.layers.append(k.layers.Conv1D(10, self.conv_size, activation='elu',     dtype=float_pres, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        # self.layers.append(k.layers.ZeroPadding1D(padding=(self.conv_size-1)//2))                                                        # Camada de padding de zeros em 1 dimensão
        # self.layers.append(k.layers.Conv1D(10, self.conv_size, activation='elu',     dtype=float_pres, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        # self.layers.append(k.layers.Conv1D(1, 1, activation='sigmoid', dtype=float_pres, kernel_regularizer=wei_reg)) # Camada de convolução em 1 dimensão
        
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
        return y[...,0]
    def compute_output_shape(self, input_shape):
        return input_shape


class WENO_temporal(WENO):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,Δx,CFL,Δt,fronteira,flux_calc,WENO_method,conv_size=5,regul_weight=0,p=2,ativ_func=tf.nn.sigmoid):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_temporal,self).__init__(flux_calc,WENO_method,conv_size,regul_weight,p=2,ativ_func=ativ_func)
        self.Δx=tf.constant(Δx,dtype=float_pres)
        self.CFL=tf.constant(CFL,dtype=float_pres)
        self.Δt=tf.constant(Δt,dtype=float_pres)
        self.fronteira=fronteira
        
    def call(self, inpt, mask=None):
        return self.Sim_step_graph(inpt,self.Δt, self.Δx, self.fronteira)
    
class WENO_espacial(WENO):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,Δx,fronteira,flux_calc,WENO_method,conv_size=5,regul_weight=0,p=2,ativ_func=tf.nn.sigmoid):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_espacial,self).__init__(flux_calc,WENO_method,conv_size,regul_weight,p=2,ativ_func=ativ_func)
        self.Δx=tf.Variable(Δx,dtype=float_pres,trainable=False)
        self.fronteira=fronteira
        
    def call(self, inpt, mask=None):
        return self.DerivadaEspacial_graph(inpt,self.Δx, self.fronteira)


class WENO_temporal_Z_plus(WENO_temporal):
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
        return self.Δx**y

class WENO_espacial_Z_plus(WENO_espacial):
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
        return self.Δx**y
    
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
        y_min  = tf.math.reduce_min(y_true,axis=1,keepdims=True)
        y_max  = tf.math.reduce_max(y_true,axis=1,keepdims=True)
        
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
    def __init__(self,pre_loss,tol=100):
        super(Only_smooth_loss,self).__init__()
        self.pre_loss=pre_loss
        self.tol=tol

    
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
        valid_index  = tf.where(tf.abs(y_true)<self.tol,1.0,0.0)
        valid_index = tf.cast(valid_index, y_pred.dtype) # Convertendo os tipos para evitar conflitos
        
        loss = tf.reduce_mean(valid_index*self.pre_loss(y_pred,y_true)/(tf.abs(y_true)+1), axis=-1)
        
        return loss

def slicer(data,n,API):
    helper = lambda i: data[...,i:i+n]

    data_sliced = API.einsum(
    'i...j -> ...ij',
        API.map_fn(
            helper,                    # Função a ser executada a cada iteração do loop
            API.range(API.shape(data)[-1]-n+1),     # Índices utilizados no loop
            fn_output_signature=data.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        )
    )

    return data_sliced

class Conv1D(k.layers.Layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,n_kernel,kernel_size,activation,name='conv_custom'):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(Conv1D, self).__init__(name=name,dtype=float_pres) # Chamando o inicializador da superclasse
        self.n_kernel=n_kernel
        self.kernel_size=kernel_size
        self.activation=activation
        self.pad=(kernel_size-1)//2
    def build(self, input_shape):
        """
        Função para compor as camadas que constituem essa camada da rede neural
        ------------------------------------------------------------------------
        input_shape : não é utilizado por essa função, mas é um argumento obrigatório para camadas do Keras.
        ------------------------------------------------------------------------
        """
        in_size=input_shape[-1]
        self.w = self.add_weight(self.name+'_'+"kernel",
                        initializer='glorot_uniform',
                        shape=[self.kernel_size,in_size,self.n_kernel],
                        trainable=True
                                     )
        self.b = self.add_weight(self.name+'_'+"bias",
        shape=[1,1,self.n_kernel],
        trainable=True)
    def call(self,inpt):
        return self.activation(
            tf.nn.conv1d(inpt, self.w, [1], padding='SAME')+self.b
        )
    def compute_output_shape(self, input_shape):
        tf.print(input_shape)
        return input_shape