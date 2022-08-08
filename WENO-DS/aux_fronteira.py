# Código para implementar o WENO-Z utilizando o tensorflow
# Importando os módulos que serão utilizados

def FronteiraFixa(U, API, n=3, t=None, Δx=None):
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
        API.repeat(U[...,:1], n, axis=-1),
        U,
        API.repeat(U[...,-1:], n, axis=-1)],
        axis=-1)
    return U

def FronteiraPeriodica(U, API, n=3, t=None, Δx=None):
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

def FronteiraReflexiva(U, API, n=3, t=None, Δx=None):
    
    U0 = API.concat([
        API.flip(U[...,0,:n], axis=[-1]),
        U[...,0,:],
        API.flip(U[...,0,-n:], axis=[-1])],
        axis=-1)

    U1 = API.concat([
        -API.flip(U[...,1,:n], axis=[-1]),
        U[...,1,:],
        -API.flip(U[...,1,-n:], axis=[-1])],
        axis=-1)

    U2 = API.concat([
        API.flip(U[...,2,:n], axis=[-1]),
        U[...,2,:],
        API.flip(U[...,2,-n:], axis=[-1])],
        axis=-1)
    
    U=API.stack([U0,U1,U2],axis=-2)
    
    return U

def TitarevToroBoundary(U, API, n=3, t=None, Δx=None):
        
    xr = API.constant([[5+Δx, 5+2*Δx, 5+3*Δx]])
    
    ρ = 1.0 + API.sin(20.0*API.pi*xr)/10
    u = API.zeros([1,3])
    p = API.ones([1,3])
    
    γ  = API.constant(14.0)/10
    ρu = ρ*u
    E  = p/(γ-1) + ρ*(u**2)/2
    
    UR = API.concat([ρ, ρu, E], axis=-2)

    U = API.concat([
        API.repeat(U[...,:1], n, axis=-1),
        U,
        UR],
        axis=-1)
    
#     print(U[:,-5:])
    
    return U