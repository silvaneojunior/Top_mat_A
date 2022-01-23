# Código para implementar o WENO-Z utilizando o numpy
# Importando os módulos que serão utilizados

import numpy as np
import matplotlib.pyplot as plt

def Burgers(u, t_final, Δx, CFL, fronteira):
    """
    Função que recebe um array u contendo os valores de uma determinada função e 
    retorna os valores após t_final unidades de tempo estimados de acordo com a 
    equação de Burgers
    -------------------------------------------------------------------------------------------
    u          (ndarray): valores da condição inicial da função
    t_final      (float): unidades de tempo a serem avançadas
    Δx           (float): distância espacial dos pontos na malha utilizada
    CFL          (float): constante utilizada para determinar o tamanho da malha temporal
    fronteira (function): função que determina o comportamento do algoritmo na fronteira
    -------------------------------------------------------------------------------------------
    u          (ndarray): valores da função após t_final unidades de tempo
    Δt_list       (list): lista com os Δt's utilizados na computação até chegar no t_final
    -------------------------------------------------------------------------------------------
    """
    
    t = 0.0      # Instante de tempo incial para a computação
    Δt_list = [] # Lista de instantes de tempo a ser retornada
    
    while t < t_final: # Loop que garante que o algoritmo será executado até t_final
        
        Λ  = np.max(np.abs(u)) # Valor utilizado para obter o Δt
        Δt = Δx*CFL/Λ          # Obtendo o valor de Δt a partir de CFL
        if t + Δt > t_final:
             Δt = t_final - t  # Caso o passo temporal utrapasse o valor de t_final
                               # então o tamanho do passo se torna o tempo que 
                               # falta para se obter o t_final
                        
        Δt_list.append(Δt)     # Incluindo os Δt's utilizados na lista
        
        # SSP Runge-Kutta 3,3
        u1 = u - Δt*DerivadaEspacial(u, Δx, fronteira)
        u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, fronteira)) / 4.0
        u  = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, fronteira)) / 3.0
        
        t = t + Δt # Avançando no tempo

    return u, Δt_list

def FronteiraFixa(U):
    """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    fixa, repetindo os valores nos extremos da malha
    ----------------------------------------------------------------------------
    U (ndarray): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (ndarray): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = np.concatenate([
        U[0:1,:],
        U[0:1,:],
        U[0:1,:],
        U,
        U[-1:,:],
        U[-1:,:],
        U[-1:,:]],
        axis=0)
    
    return U


def FronteiraPeriodica(U):
    """
    Função que adicionada pontos na malha de acordo com a condição de fronteira
    periódica, continuado os valores de acordo com os extremos opostos
    ----------------------------------------------------------------------------
    U (ndarray): malha de pontos a ser estendida de acordo
    ----------------------------------------------------------------------------
    U (ndarray): malha de pontos estendida
    ----------------------------------------------------------------------------
    """
    U = np.concatenate([
        U[-3:-2,:],
        U[-2:-1,:],
        U[-1:,:],
        U,
        U[:1,:],
        U[1:2,:],
        U[2:3,:]],
        axis=0)
    
    return U

def DerivadaEspacial(U, Δx, AdicionaGhostPoints):
    """
    Calcula a derivada espacial numa malha de pontos utilizando o WENO-Z
    ---------------------------------------------------------------------------------
    U                    (ndarray): valores da função para o cálcula da derivada
    Δx                     (float): distância espacial dos pontos na malha utilizada
    AdicionaGhostPoints (function): função que adicionada pontos na malha de acordo 
                                    com a condição de fronteira
    ---------------------------------------------------------------------------------
    Fdif                 (ndarray): derivada espacial
    ---------------------------------------------------------------------------------
    """
    
    Fhat = np.zeros(U.shape[0]+1) # Variável que salva os valores estimados da função
    U    = AdicionaGhostPoints(U) # Estende a malha de pontos de acordo com as condições de fronteira
    
    for i in range(2, U.shape[0]+1-3-1):
        
        u_i     = U[i-2:i+3+1,:]         # Estêncil de 6 pontos a ser utilizado pelo algoritmo
        M       = np.max(np.abs(u_i))    # Valor utilizado para realizar a separação de fluxo
        f_plus  = (u_i**2/2 + M*u_i) / 2 # Fluxo positivo
        f_minus = (u_i**2/2 - M*u_i) / 2 # Fluxo negativo
        
        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = WenoZ5ReconstructionMinus(f_plus[0],  f_plus[1],  f_plus[2],  f_plus[3],  f_plus[4])
        f_half_plus  = WenoZ5ReconstructionPlus(f_minus[1], f_minus[2], f_minus[3], f_minus[4], f_minus[5])
        Fhat[i-2]    = (f_half_minus + f_half_plus)[0]
    
    # Calculando uma estimava da derivada a partir de diferenças finitas
    Fhat = np.expand_dims(Fhat, axis=1)
    Fdif = (Fhat[1:] - Fhat[:-1]) / Δx
    
    return Fdif

def WenoZ5ReconstructionMinus(u1, u2, u3, u4, u5):
    """
    Calcula o WENO-Z para obter uma estimativa da função utilizando um subestêncil 
    de cinco pontos
    --------------------------------------------------------------------------------
    u1   (ndarray): array com os pontos de número 1 do subestêncil 
    u2   (ndarray): array com os pontos de número 2 do subestêncil 
    u3   (ndarray): array com os pontos de número 3 do subestêncil 
    u4   (ndarray): array com os pontos de número 4 do subestêncil 
    u5   (ndarray): array com os pontos de número 5 do subestêncil 
    --------------------------------------------------------------------------------
    fhat (ndarray): estimativa da função
    --------------------------------------------------------------------------------
    """
    # Constante para 
    ɛ = 10.0**(-40)
    
    # Calcula os indicadores de suavidade locais
    β0 = ( 1/2.0*u1 - 2*u2 + 3/2.0*u3)**2 + 13/12.0*(u1 - 2*u2 + u3)**2
    β1 = (-1/2.0*u2        + 1/2.0*u4)**2 + 13/12.0*(u2 - 2*u3 + u4)**2
    β2 = (-3/2.0*u3 + 2*u4 - 1/2.0*u5)**2 + 13/12.0*(u3 - 2*u4 + u5)**2
    
    # Calcula o indicador de suavidade global
    τ = np.abs(β0 - β2)
    
    # Calcula os pesos do WENO-Z
    α0 = (1/10) * (1 + (τ/(β0 + ɛ))**2)
    α1 = (6/10) * (1 + (τ/(β1 + ɛ))**2)
    α2 = (3/10) * (1 + (τ/(β2 + ɛ))**2)
    soma = α0 + α1 + α2
    ω0 = α0 / soma
    ω1 = α1 / soma
    ω2 = α2 / soma
    
    # Calcula os fhat em cada subestêncil
    fhat0 = (2*u1 - 7*u2 + 11*u3)/6
    fhat1 = ( -u2 + 5*u3 +  2*u4)/6
    fhat2 = (2*u3 + 5*u4 -    u5)/6
    
    # Calcula o fhat do estêncil todo
    fhat = ω0*fhat0 + ω1*fhat1 + ω2*fhat2
    
    return fhat

def WenoZ5ReconstructionPlus(u1, u2, u3, u4, u5):
    """
    Calcula o WENO-Z para obter uma estimativa da função utilizando um subestêncil 
    de cinco pontos
    --------------------------------------------------------------------------------
    u1   (ndarray): array com os pontos de número 1 do subestêncil 
    u2   (ndarray): array com os pontos de número 2 do subestêncil 
    u3   (ndarray): array com os pontos de número 3 do subestêncil 
    u4   (ndarray): array com os pontos de número 4 do subestêncil 
    u5   (ndarray): array com os pontos de número 5 do subestêncil 
    --------------------------------------------------------------------------------
    fhat (ndarray): estimativa da função
    --------------------------------------------------------------------------------
    """
    # Reciclando a função anterior WenoZ5ReconstructionMinus
    fhat = WenoZ5ReconstructionMinus(u5, u4, u3, u2, u1)
    
    return fhat

def Gera_dados():
    """
    Função que gera dados para servirem de treino para a rede neural
    -----------------------------------------------------------------
    U (ndarray): Array de estados armazenados
    -----------------------------------------------------------------
    """

    Δx = 0.01                 # Distância espacial dos pontos na malha utilizada
    x  = np.arange(-2, 2, Δx) # Gerando a malha de pontos no espaço unidimensional
    
    k1 = np.round(np.random.uniform(-0.5, 20.5))    # Amostrando uma frequência aleatória para a função seno
    k2 = np.round(np.random.uniform(-0.5, 20.5))    # Amostrando uma frequência aleatória para a função seno
    a  = np.random.uniform(0, 1)                    # Amostrando um peso aleatória para ponderar as funções seno
    b  = np.random.uniform(0, 2)                    # Amostrando um modificador de amplitude aleatório
    u1 =     a * np.asarray([np.sin(k1*np.pi*x)]).T # Gerando pontos de acordo com a primeira função seno
    u2 = (1-a) * np.asarray([np.sin(k2*np.pi*x)]).T # Gerando pontos de acordo com a segunda função seno
    
    u = b*(u1+u2) # Obtendo a condição inicial a partir das funções senos
    
    CFL = 0.5                        # Constante utilizada para determinar o tamanho da malha temporal
    Δt  = 0.01                       # Δt entre estados armazenados
    T   = np.arange(0.0, 2, Δt)      # Array de instantes de tempo cujo resultado deve ser armazenado
    U   = np.zeros([len(u), len(T)]) # Gerando um array de duas dimensões para conter todos os estados obtidos
    
    U[:,0:1] = u # Condição inicial
    t = 0.0      # Instante de tempo inicial
    i = 2        # Índice
    
    while t < T[-1]:
        # Calculando próximo estado a ser armazenado
        u = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica)[0]
        
        # Armazenando o estado
        U[:,i:i+1] = u
        
        i += 1
        t += Δt
        
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
    
    Δx = 0.01               # Distância espacial dos pontos na malha utilizada
    x  = np.arange(-2,2,Δx) # Gerando a malha de pontos no espaço unidimensional
    
    # Condição inicial aleatória
    #----------------------------------------------------------------------------------------------------------
    k1 = np.round(np.random.uniform(-0.5,20.5)) # Amostrando uma frequência aleatória para a função seno
    k2 = np.round(np.random.uniform(-0.5,20.5)) # Amostrando uma frequência aleatória para a função seno
    a  = np.random.uniform(0,1)                 # Amostrando um peso aleatória para ponderar as funções seno
    b  = np.random.uniform(0,2)                 # Amostrando um modificador de amplitude aleatório
    #----------------------------------------------------------------------------------------------------------
    
    # Condição inicial fixada
    #-----------------------------------------------
#     k1 = 1.0 # Frequência para a função seno
#     k2 = 2   # Frequência para a função seno
#     a  = 0.5 # Peso para ponderar as funções seno
#     b  = 0.5 # Modificador de amplitude
    #-----------------------------------------------
    
    u1 =     a * np.asarray([np.sin(k1*np.pi*x)]).T # Gerando pontos de acordo com a primeira função seno
    u2 = (1-a) * np.asarray([np.sin(k2*np.pi*x)]).T # Gerando pontos de acordo com a segunda função seno
    
    u = b*(u1+u2) # Obtendo a condição inicial a partir das funções senos
    
    CFL = 0.5                        # Constante utilizada para determinar o tamanho da malha temporal
    Δt  = 0.01                       # Δt entre cada frame de animação
    T   = np.arange(0.0, 2.0,Δt)     # Frames da animação
    U   = np.zeros([len(u), len(T)]) # Gerando um array para conter todos os estados obtidos
                                     # durante a animação
    U[:,0:1] = u # Condição inicial
    t = 0.0      # Instante de tempo inicial
    i = 2        # Índice
    
    
    # Gerando os gráficos a partir de funções do matplotlib
    
    fig = plt.figure(1, constrained_layout=True, figsize=(6,6))
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_ylim(-2, 2)
    #ax.set_xlim(0,1)
    line = ax.plot(x,u)
    hfig = display(fig, display_id=True)
    Δt_list = []
    
    while t < T[-1]:
        # Calculando próximo estado a ser exibido
        u, elem_Δt = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica)
        Δt_list += elem_Δt
        
        # Armazenando o estado
        U[:,i:i+1] = u
        i += 1
        t += Δt
        
        # Exibindo graficamente os valores obtidos
        line[0].set_ydata(u)
        fig.canvas.draw()
        hfig.update(fig)
        
    return Δt_list