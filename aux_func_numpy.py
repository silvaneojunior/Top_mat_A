import numpy as np
import matplotlib.pyplot as plt

def Burgers(u, t_final, Δx, CFL, fronteira):
    t=0.0
    Δt_list=[]
    while t<t_final:
        Λ = np.max(np.abs(u))
        Δt = Δx*CFL/Λ #Condicao de CFL - deve ser atualizada a cada passo de tempo
        if t + Δt > t_final:
             Δt = t_final - t
        Δt_list.append(Δt)
        #SSP Runge-Kutta 3,3
        u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira, t)
        Λ = np.max(np.abs(u1))
        u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira, t + Δt)) / 4.0
        Λ = np.max(np.abs(u2))
        u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira, t + Δt/2)) / 3.0
        t = t + Δt
    return u,Δt_list

def FronteiraFixa(U, t):
    
    return np.concatenate([U[0:1,:],
                           U[0:1,:],
                           U[0:1,:],
                           U,
                           U[-1:,:],
                           U[-1:,:],
                           U[-1:,:]],
                         axis=0)


def FronteiraPeriodica(U, t):
    return np.concatenate([U[-3:-2,:],
                           U[-2:-1,:],
                           U[-1:,:],
                           U,
                           U[:1,:],
                           U[1:2,:],
                           U[2:3,:]],
                         axis=0)

def DerivadaEspacial(U, Δx, Λ, AdicionaGhostPoints, t):
    Fhat = np.zeros(U.shape[0]+1)
    U = AdicionaGhostPoints(U, t) # Adiciona ghost cells de acordo com as condições de fronteira
    for i in range(2,U.shape[0]+1-3-1):
        u_i = U[i-2:i+3+1,:] # Estêncil de 6 pontos onde vamos trabalhar

        M = np.max(np.abs(u_i)) #Λ
        f_plus = (u_i**2/2 + M*u_i) / 2
        f_minus = (u_i**2/2 - M*u_i) / 2

        f_half_minus = WenoZ5ReconstructionMinus(f_plus[0], f_plus[1], f_plus[2], f_plus[3], f_plus[4]) # Aplicar WENO em cada variável característica separadamente
        f_half_plus = WenoZ5ReconstructionPlus(f_minus[1], f_minus[2], f_minus[3], f_minus[4], f_minus[5])
        
        Fhat[i-2] = (f_half_minus + f_half_plus)[0]
    Fhat=np.expand_dims(Fhat,axis=1)
    return (Fhat[1:] - Fhat[:-1]) / Δx

def WenoZ5ReconstructionMinus(u1, u2, u3, u4, u5):
    ɛ = 10.0**(-40)
    # Calcula os indicadores de suavidade locais
    β0 = (1/2.0*u1 - 2*u2 + 3/2.0*u3)**2 + 13/12.0*(u1 - 2*u2 + u3)**2
    β1 = (-1/2.0*u2 + 1/2.0*u4)**2 + 13/12.0*(u2 - 2*u3 + u4)**2
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
    fhat1 = (-u2 + 5*u3 + 2*u4)/6
    fhat2 = (2*u3 + 5*u4 - u5)/6
    #Finalmente, calcula o fhat do estêncil todo
    return ω0*fhat0 + ω1*fhat1 + ω2*fhat2

def WenoZ5ReconstructionPlus(u1, u2, u3, u4, u5):
    return WenoZ5ReconstructionMinus(u5, u4, u3, u2, u1)

def Gera_dados():
    Δx = 0.01
    x = np.arange(-2,2,Δx)
    
    k1=np.round(np.random.uniform(-0.5,20.5))
    k2=np.round(np.random.uniform(-0.5,20.5))
    a=np.random.uniform(0,1)
    b=np.random.uniform(0,2)
    u1 = a*np.asarray([np.sin(k1*np.pi*x)]).T
    u2 = (1-a)*np.asarray([np.sin(k2*np.pi*x)]).T
    
    u=b*(u1+u2)
    
    CFL = 0.5

    Δt = 0.01          # Δt da animação
    T = np.arange(0.0,2,Δt)    # Frames da animação

    U = np.zeros([len(u), len(T)])
    U[:,0:1] = u
    t = 0.0
    i = 2
    
    while t < T[-1]:
        u = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica)
        U[:,i:i+1] = u
        i += 1
        t += Δt
    return U

def AnimaçãoBurgers():
    Δx = 0.01
    x = np.arange(-2,2,Δx)
    k1=np.round(np.random.uniform(-0.5,20.5))
    k2=np.round(np.random.uniform(-0.5,20.5))
    a=np.random.uniform(0,1)
    b=np.random.uniform(0,2)
#     k1=1.0
#     k2=2
#     a=0.5
#     b=0.5
    u1 = a*np.asarray([np.sin(k1*np.pi*x)]).T
    u2 = (1-a)*np.asarray([np.sin(k2*np.pi*x)]).T
    
    u=b*(u1+u2)
    CFL = 0.5

    Δt = 0.01          # Δt da animação
    T = np.arange(0.0,2.0,Δt)    # Frames da animação

    U = np.zeros([len(u), len(T)])
    U[:,0:1] = u
    t = 0.0
    i = 2
    
    fig = plt.figure(1, constrained_layout=True,figsize=(6,6))
    ax = fig.add_subplot(1,1,1);
    ax.set_ylim(-2, 2);
    #ax.set_xlim(0,1);
    line=ax.plot(x,u)
    hfig = display(fig, display_id=True)
    Δt_list=[]
    
    while t < T[-1]:
        u,elem_Δt = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica)
        Δt_list+=elem_Δt
        
        U[:,i:i+1] = u
        i += 1
        t += Δt
        #line.set_xdata(x)
        line[0].set_ydata(u)
        
        fig.canvas.draw()
        hfig.update(fig)
        #time.sleep(0.01);
    return Δt_list