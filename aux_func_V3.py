# Código para implementar o WENO-Z utilizando o tensorflow
# Nesta versão, os pesos do WENO-Z são modificados fazendo uma combinação convexa entre os pesos ideias e os pesos do WENO-Z, sendo que o valor da combinação convexa é escolhido pela rede neural.

# Importando os módulos que serão utilizados

import tensorflow as tf
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import Null_net

float_pres='float64' # Definindo a precisão padrão para as análises

def Graph_Burgers(u, Δt, Δx, fronteira, network=Null_net.Network):
    """
    Função que recebe um tensor u contendo os valores de uma determinada função e retorna os 
    valores após Δt unidades de tempo estimados de acordo com a equação de Burgers
    -------------------------------------------------------------------------------------------
    u           (tensor): valores da condição inicial da função
    Δt           (float): variação temporal do estado inicial para o final
    Δx           (float): distância espacial dos pontos na malha utilizada
    fronteira (function): função que determina o comportamento do algoritmo na fronteira
    network   (function): função que computa o valor de saída da rede neural a partir do valor 
                          de entrada
    -------------------------------------------------------------------------------------------
    u           (tensor): valores da função após Δt unidades de tempo
    -------------------------------------------------------------------------------------------
    """
    
    # SSP Runge-Kutta 3,3
    u1 = u - Δt*DerivadaEspacial(u, Δx, fronteira, network)
    u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, fronteira, network)) / 4.0
    u  = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, fronteira, network)) / 3.0
    
    return u

def Graph_Burgers2(u, t_final, Δx, CFL, fronteira, network=Null_net.Network):
    """
    Função que recebe um tensor u contendo os valores de uma determinada função e retorna os 
    valores após Δt unidades de tempo estimados de acordo com a equação de Burgers
    -------------------------------------------------------------------------------------------
    u           (tensor): valores da condição inicial da função
    t_final      (float): unidades de tempo a serem avançadas
    Δx           (float): distância espacial dos pontos na malha utilizada
    CFL          (float): constante utilizada para determinar o tamanho da malha temporal
    fronteira (function): função que determina o comportamento do algoritmo na fronteira
    network   (function): função que computa o valor de saída da rede neural a partir do valor 
                          de entrada
    -------------------------------------------------------------------------------------------
    u           (tensor): valores da função após Δt unidades de tempo
    -------------------------------------------------------------------------------------------
    """
    
    t = 0.0 # Instante de tempo incial para a computação

    while tf.math.reduce_any(t < t_final):
        
        # Valor utilizado para obter o Δt
        Λ  = tf.math.reduce_max(tf.abs(u), axis=1, keepdims=True)
        
        # Obtendo o valor de Δt a partir de CFL
        Δt = Δx*CFL/Λ
        
        # Caso o passo temporal utrapasse o valor de t_final então o 
        # tamanho do passo se torna o tempo que falta para se obter o 
        # t_final
        Δt = tf.where(t + Δt > t_final, t_final - t, Δt)
        
        # SSP Runge-Kutta 3,3
        u1 = u - Δt*DerivadaEspacial(u, Δx, fronteira, network)
        u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, fronteira, network)) / 4.0
        u  = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, fronteira, network)) / 3.0
        
        t  = t + Δt # Avançando no tempo
        
    return u

@tf.function # Ornamento que transforma a função em uma função do tensorflow
def Burgers(u, Δt, Δx, CFL, fronteira, network=Null_net.Network):
    """Função wrapper de 'Graph_Burgers2'"""
    return Graph_Burgers2(u, Δt, Δx, CFL, fronteira, network)

def FronteiraFixa(U):
     """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    fixa, repetindo os valores nos extremos da malha
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = tf.concat([
        U[:,0:1],
        U[:,0:1],
        U[:,0:1],
        U,
        U[:,-1:],
        U[:,-1:],
        U[:,-1:]],
        axis=1)
    return U


def FronteiraPeriodica(U):
    """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    periódica, continuado os valores de acordo com os extremos opostos
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (tensor): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = tf.concat([U[:,-3:], U, U[:,:3]], axis=1)
    return U

def DerivadaEspacial(U, Δx, AdicionaGhostPoints, network):
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
    
    Fhat = []                  # Variável que salva os valores estimados da função
    U = AdicionaGhostPoints(U) # Estende a malha de pontos de acordo com as condições de fronteira
    
    # Calculando o input para a rede neural
    # Entrada 1: U_{i+1} - U_{i-1}
    # Entrada 2: U_{i+1} - 2*U_{i} + U_{i-1}
    U_diff = tf.concat([U[:,2:]-U[:,:-2], U[:,2:]-2*U[:,1:-1]+U[:,:-2]], axis=2)
    
    U = U[:,:,0]                     # Remove a terceira dimensão de U
    beta_weight = network(U_diff)    # Passa U_diff como input para a rede salva o output
    beta_weight = beta_weight[:,:,0] # Remove a terceira dimensão de beta_weights
    
    # Funções auxiliares que serão utilizadas dentro de tf.map_fn
    helper1 = lambda i: U[:,i:i+6]
    helper2 = lambda i: beta_weight[:,i:i+3]
    
    # Utilizando um loop implícito (com tf.map_fn) para obter cada subestêncil de 
    # 5 pontos e depois concatenar em um único tensor (com tf.concat) na primeira
    # dimensão
    U_full = tf.concat(
        tf.map_fn(
            helper1,                    # Função a ser executada a cada iteração do loop
            tf.range(U.shape[1]-5),     # Índices utilizados no loop
            fn_output_signature=U.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        ), axis=0)
    
    # Utilizando um loop implícito (com tf.map_fn) para obter cada trio de modificadores
    # dos indicadores de suavidade e depois concatenar em um único tensor (com tf.concat)
    # na primeira dimensão
    beta_weight_full = tf.concat(
        tf.map_fn(
            helper2,                              # Função a ser executada a cada iteração do loop
            tf.range(beta_weight.shape[1]-2),     # Índices utilizados no loop
            fn_output_signature=beta_weight.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        ), axis=0)
    
    U_full = tf.transpose(U_full,[1,0,2])                     # Mudando a ordem das dimensões do tensor
    beta_weight_full = tf.transpose(beta_weight_full,[1,0,2]) # Mudando a ordem das dimensões do tensor
    
    # Setup para equação de Burgers
    M = tf.math.reduce_max(tf.abs(U_full),axis=2,keepdims=True) # Valor utilizado para realizar a separação de fluxo
    f_plus  = (U_full**2/2 + M*U_full) / 2                      # Fluxo positivo
    f_minus = (U_full**2/2 - M*U_full) / 2                      # Fluxo negativo
    
    # Setup para equação do transporte
    M = 1                           # Valor utilizado para realizar a separação de fluxo
    f_plus  = (U_full + M*U_full)/2 # Fluxo positivo
    f_minus = (U_full - M*U_full)/2 # Fluxo negativo
    
    # Aplicar WENO em cada variável característica separadamente para depois juntar
    f_half_minus = WenoZ5ReconstructionMinus(f_plus[:,:,:-1], beta_weight_full[:,:-1,:]) 
    f_half_plus  = WenoZ5ReconstructionPlus( f_minus[:,:,1:], beta_weight_full[:,1:,:])
    Fhat         = (f_half_minus + f_half_plus)
    
    # Calculando uma estimava da derivada a partir de diferenças finitas
    Fhat = tf.transpose((Fhat[:,1:] - Fhat[:,:-1]) / Δx)
    
    return Fhat

"""
Obtendo matrizes de constantes convenintes para executar o WENO-Z
utilizando operações tensoriais, uma vez que permite a integração
com o tensorflow
"""

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
A = tf.cast(tf.stack([A0,A1,A2], axis=0), dtype=float_pres) 
A = tf.expand_dims(A, axis=1)

B = tf.constant([[1,0,0],[0,6,0],[0,0,3]], dtype=float_pres)/10                # Matriz B
C = tf.constant([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]], dtype=float_pres)/6 # Matriz C
C = tf.transpose(C)

def WenoZ5ReconstructionMinus(u0, beta_weight):
    """
    Calcula o WENO modificado para obter uma estimativa da função utilizando um 
    subestêncil de cinco pontos
    --------------------------------------------------------------------------------
    u0          (tensor): tensor com o substêncil de 5 pontos
    beta_weight (tensor): modificadores dos pesos do substêncil
    --------------------------------------------------------------------------------
    fhat        (tensor): estimativa da função
    --------------------------------------------------------------------------------
    """
    ɛ = 10.0**(-40)
    
    # Calcula os indicadores de suavidade locais
    u = tf.stack([u0, u0, u0], axis=0)
    
    β = tf.math.reduce_sum(u * (u @ A), axis=3)
    β = tf.transpose(β, [1,2,0])
    
    # Calcula o indicador de suavidade global
    τ = tf.abs(β[:,:,0:1] - β[:,:,2:3])
    
    # Calcula os pesos do WENO-Z
    α    = (1 + (τ/(β + ɛ))**2) @ B
    soma = tf.math.reduce_sum(α, axis=2, keepdims=True)
    ω    = α / soma
    ω = (1-beta_weight)*ω + beta_weight*tf.constant([[[1,6,3]]],dtype=float_pres)/10
    
    # Calcula os fhat em cada subestêncil
    fhat = u0 @ C
    
    # Calcula o fhat do estêncil todo
    fhat = tf.transpose(tf.math.reduce_sum(ω * fhat, axis=2, keepdims=True))
    
    return fhat

def WenoZ5ReconstructionPlus(u0, beta_weight):
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
    fhat = WenoZ5ReconstructionMinus(tf.reverse(u0,axis=[2]), tf.reverse(beta_weight,axis=[2]))
    
    return fhat
    
def Gera_dados():
    """
    Função que gera dados para servirem de treino para a rede neural
    -----------------------------------------------------------------
    U (tensor): tensor de estados armazenados
    -----------------------------------------------------------------
    """
    Δx  = 0.01                                  # Distância espacial dos pontos na malha fina utilizada
    Δx2 = 4*Δx                                  # Distância espacial dos pontos na malha grossa utilizada
    x   = tf.range(-2, 2, Δx, dtype=float_pres) # Gerando a malha de pontos no espaço unidimensional
    
    u_list = [] # Lista para armazenar as condições inicias geradas
    
    # Gerando condições iniciais
    while len(u_list) < 10000:
        
        k1 = tf.random.uniform([1], 0, 20, dtype='int32') # Amostrando uma frequência aleatória para a função seno
        k1 = tf.cast(k1, dtype=float_pres)                # Mudando o tipo do tensor
        k2 = tf.random.uniform([1], 0, 20, dtype='int32') # Amostrando uma frequência aleatória para a função seno
        k2 = tf.cast(k2, dtype=float_pres)                # Mudando o tipo do tensor
        
        a  = tf.random.uniform([1], 0, 1, dtype=float_pres)       # Amostrando um peso aleatória para ponderar as funções seno
        b  = tf.random.uniform([1], 0, 2, dtype=float_pres)       # Amostrando um modificador de amplitude aleatório
        u1 =     a * tf.expand_dims(tf.math.sin(k1*pi*x), axis=1) # Gerando pontos de acordo com a primeira função seno
        u2 = (1-a) * tf.expand_dims(tf.math.sin(k2*pi*x), axis=1) # Gerando pontos de acordo com a segunda função seno

        u = b*(u1+u2)    # Obtendo a condição inicial a partir das funções senos
        u_list.append(u) # Salvando o valor em uma lista de condições iniciais
    
    u = tf.stack(u_list, axis=0) # Obtendo um tensor com todas as condições iniciais
    u = u.numpy()                # Transformando em um array do numpy
    U = []                       # Lista que irá conter todos os estados simulados
    U.append([u])                # Inserindo as condições iniciais
    
    CFL    = 0.5                          # Constante utilizada para determinar o tamanho da malha temporal
    T      = tf.range(0.0, 2.0, Δt_max)   # Tensor de instantes de tempo cujo resultado deve ser armazenado
    T      = tf.cast(T, dtype=float_pres) # Mudando o tipo do tensor
    Δt_max = 0.01                         # Δt entre estados armazenados
    
    
    
    # Simulando os novos estados a partir das condições iniciais
    for i in range(200):
        
        # Subconjunto da malha mais fina
        short_u = tf.gather(u, tf.range(100)*4, axis=1)
        
        # Valor utilizado para obter o Δt
        Λ = tf.math.reduce_max(tf.abs(short_u), axis=1, keepdims=True)
        
        # Obtendo o valor de Δt a partir de CFL
        Δt = Δx2*CFL/Λ
        
        # Caso o passo temporal utrapasse o valor de t_final então o 
        # tamanho do passo se torna o tempo que falta para se obter o 
        # t_final
        Δt = tf.where(Δt > Δt_max, Δt_max, Δt)
        
        # Calculando os próximos estados a serem armazenados
        u = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica).numpy()
        
        # Armazenando os estados simulados
        U.append([u])
        
        # Verificando em qual etapa está a geração de dados
        print(i) 

    return U

def AnimaçãoBurgers():
    """
    Função que gera uma sequência de gráficos obtidos a partir da evolução de 
    uma condição inicial de acordo com a equação de Burgers
    --------------------------------------------------------------------------
    Δt_list (list): instantes de tempo para os quais o estado do sistema foi
                    computado
    --------------------------------------------------------------------------
    """
    
    Δx = 0.01                                 # Distância espacial dos pontos na malha utilizada
    x = tf.range(-2, 2, Δx, dtype=float_pres) # Gerando a malha de pontos no espaço unidimensional
    
    # Gerando uma condição inicial aleatória
    #------------------------------------------------------------------------------------------------------------------
    k1 = tf.random.uniform([1], 0, 20, dtype='int32')   # Amostrando uma frequência aleatória para a função seno
    k1 = tf.cast(k1, dtype=float_pres)                  # Mudando o tipo do tensor
    k2 = tf.random.uniform([1], 0, 20, dtype='int32')   # Amostrando uma frequência aleatória para a função seno
    k2 = tf.cast(k2, dtype=float_pres)                  # Mudando o tipo do tensor
    a  = tf.random.uniform([1], 0, 1, dtype=float_pres) # Amostrando um peso aleatória para ponderar as funções seno
    b  = tf.random.uniform([1], 0, 2, dtype=float_pres) # Amostrando um modificador de amplitude aleatório
    #------------------------------------------------------------------------------------------------------------------

    # Fixando a condição inicial
    #-----------------------------------------------
#     k1 = 1.0 # Frequência para a função seno
#     k2 = 2.0 # Frequência para a função seno
#     a  = 0.5 # Peso para ponderar as funções seno
#     b  = 0.5 # Modificador de amplitude
    #-----------------------------------------------

    u1 =     a * tf.expand_dims(tf.math.sin(k1*pi*x), axis=1) # Gerando pontos de acordo com a primeira função seno
    u2 = (1-a) * tf.expand_dims(tf.math.sin(k2*pi*x), axis=1) # Gerando pontos de acordo com a segunda função seno
    
    u = b*(u1+u2)                 # Obtendo a condição inicial a partir das funções senos
    u = tf.expand_dims(u, axis=0) # Acrescentando uma dimensão
    
    CFL = 0.5                          # Constante utilizada para determinar o tamanho da malha temporal
    Δt  = 0.01                         # Δt entre cada frame de animação
    T   = tf.range(0.0, 2.0, Δt)       # Frames da animação
    T   = tf.cast(T, dtype=float_pres) # Mudando o tipo do tensor

    t = 0.0 # Instante de tempo inicial
    
    
    # Gerando os gráficos a partir de funções do matplotlib
    
    fig = plt.figure(1, constrained_layout=True, figsize=(6,6))
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_ylim(-2, 2)
    # ax.set_xlim(0,1)
    line = ax.plot(x, tf.squeeze(u))
    hfig = display(fig, display_id=True)
    Δt_list = tf.zeros([0], dtype=float_pres)
    
    while t < T[-1]:
        # Calculando próximo estado a ser exibido
        u, elem_Δt = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica)[1:] # AVISO: Conferir saída da função 'Burgers'
        squeezed_u = tf.squeeze(u)
        Δt_list = tf.concat([Δt_list, elem_Δt], axis=0)
        
        t += Δt # Avançando no tempo
        
        # Detectando se há pontos com valor NaN
        if tf.math.reduce_any(tf.math.is_nan(squeezed_u)):
            
            # Exibindo pontos com valor NaN
            line = ax.plot(x, tf.cast(tf.math.is_nan(squeezed_u), 'float32').numpy(), 'ro')
            fig.canvas.draw()
            hfig.update(fig)
            return Δt_list
        
        # Exibindo graficamente os valores obtidos
        line[0].set_ydata(squeezed_u.numpy())
        fig.canvas.draw()
        hfig.update(fig)
        
    return Δt_list