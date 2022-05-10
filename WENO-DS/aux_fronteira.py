# Código para implementar o WENO-Z utilizando o tensorflow
# Importando os módulos que serão utilizados

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
    
    U=API.stack([U0,U1,U2],axis=-2)
    
    return U