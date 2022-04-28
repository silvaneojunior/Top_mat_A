import tensorflow as tf
import tensorflow.keras as k
import numpy as np

from aux_weno import *
from aux_fronteira import *

class WENO_layer(k.layers.Layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,equation,WENO_method,WENO_type='temporal',conv_size=5,regul_weight=0,mapping=null_mapping, map_function=lambda x:x,p=2,ativ_func=tf.nn.sigmoid):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_layer, self).__init__(name='WENO_layer',dtype=float_pres) # Chamando o inicializador da superclasse
        self.simulation=simulation(API_TensorFlow,equation,WENO_method,network=self.network_graph,p=p,mapping=mapping, map_function=map_function)
        self.config={
            'equation':equation,
            'WENO_method':WENO_method,
            'conv_size':conv_size,
            'regul_weight':regul_weight,
            'p':p,
            'ativ_func':ativ_func,
            'mapping':mapping,
            'map_function':map_function
        }

        self.Sim, self.Sim_step, self.DerivadaEspacial, self.Get_weights=self.simulation.Sim, self.simulation.Sim_step, self.simulation.DerivadaEspacial, self.simulation.Get_weights
        self.Sim_graph, self.Sim_step_graph, self.DerivadaEspacial_graph, self.Get_weights_graph=self.simulation.Sim_graph, self.simulation.Sim_step_graph, self.simulation.DerivadaEspacial_graph, self.simulation.Get_weights_graph

        if WENO_type=='temporal':
            self.exec=self.Sim_step_graph
        elif WENO_type=='spatial':
            self.exec=lambda U, Δt, Δx, AdicionaGhostPoints: self.DerivadaEspacial_graph(U, Δx, AdicionaGhostPoints)

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
    def call(self,u, Δt, Δx, fronteira):
        return self.exec(u, Δt, Δx, fronteira)

class WENO_temporal_layer(WENO_layer):
    """Criando uma camada de rede neural cuja superclasse é a camada
    do keras para integrar o algoritmo do WENO com a rede neural"""
    
    def __init__(self,equation,WENO_method,Δx,Δt,fronteira,WENO_type='temporal',conv_size=5,regul_weight=0,mapping=null_mapping, map_function=lambda x:x,p=2,ativ_func=tf.nn.sigmoid):
        """
        Construtor da classe
        --------------------------------------------------------------------------------------
        t_final      (float): tamanho máximo da variação temporal
        Δx           (float): distância espacial dos pontos na malha utilizada
        CFL          (float): constante utilizada para determinar o tamanho da malha temporal
        fronteira (function): função que determina o comportamento do algoritmo na fronteira
        --------------------------------------------------------------------------------------
        """
        super(WENO_temporal_layer, self).__init__(equation,WENO_method,WENO_type,conv_size,regul_weight,mapping, map_function,p,ativ_func) # Chamando o inicializador da superclasse
        self.simulation=simulation(API_TensorFlow,equation,WENO_method,network=self.network_graph,p=p,mapping=mapping, map_function=map_function)
        self.Δx=Δx
        self.Δt=Δt
        self.fronteira=fronteira
    def call(self,u):
        return self.exec(u, self.Δt, self.Δx, self.fronteira)

class WENO_Z_plus(WENO_layer):
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

class MES_relative(k.losses.Loss):
    """Criando uma função de custo cuja superclasse é a de funções de
    custo do keras"""
    def __init__(self,scale):
        super(MES_relative,self).__init__()
        self.scale=scale
    
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
        
        loss = tf.reduce_mean(
            tf.math.square(y_pred - y_true), axis=-1)/self.scale
        
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