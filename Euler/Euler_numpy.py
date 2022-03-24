import numpy as np

def Euler1D_step(Q, γ, Δx, CFL, Δt, BoundaryCondition):
    # 3rd order TVD Runge-Kutta scheme
    Q1 =    Q        -   Δt * FluxDerivative(Q, γ, Δx, BoundaryCondition)
    Q2 = (3*Q +   Q1 -   Δt * FluxDerivative(Q1, γ, Δx, BoundaryCondition))/4
    Q  = (  Q + 2*Q2 - 2*Δt * FluxDerivative(Q2, γ, Δx, BoundaryCondition))/3
    return Q


def Euler1D(Q, γ, Δx, CFL, FinalTime, BoundaryCondition):
    Time = 0.0
    while Time < FinalTime:
        Δt = CFL*Δx/MaximumEigenvalue(Q, γ)
        if Δt > (FinalTime - Time):
            Δt = FinalTime - Time;
        Q=Euler1D_step(Q, γ, Δx, CFL, Δt, BoundaryCondition)
        Time += Δt;
    return Q

def FluxDerivative(Q, γ, Δx, BoundaryCondition):
    Ord = 5 # The order of the scheme
    Q = BoundaryCondition(Q)

    N = Q.shape[0]
    F_half = np.zeros([N-Ord, 3])
    G_half = np.zeros([1, 3])
    #Qi = np.zeros([Ord+1, 3])

    M = MaximumEigenvalue(Q, γ)

    for i in range(N-Ord):
        #Qi[1:Ord+1, 1:3] = Q[i:i+Ord, 1:3]
        Qi = Q[i:i+Ord+1,:]
        Λ = Eigenvalues(Qi, γ)

        Qa = Average(Qi)

        R, L = Eigensystem(Qa, γ)
        W = np.matmul(Qi,L)       # Transforms into characteristic variables
        G = Λ*W       # The flux for the characteristic variables is Λ * L*Q
        for j in range(3):    # WENO reconstruction of the flux G
            G_half[:,j] = ReconstructedFlux(G[:,j], W[:,j], M)
        F_half[i,:] = np.matmul(G_half,R) # Brings back to conservative variables
    return (F_half[1:,:] - F_half[:-1,:])/Δx # Derivative of Flux

def Average(Q):
    r = 2
    return (Q[r,:] + Q[r+1,:])/2;

def Pressure_1d(Q, γ):
    return (γ-1)*(Q[2] - Q[1]**2 / Q[0]/2)

def Pressure_2d(Q, γ):
    return (γ-1)*(Q[:,2] - Q[:,1]**2 / Q[:,0]/2)

def Eigensystem(Q, γ):
    U = Q[1]/Q[0]
    P = Pressure_1d(Q, γ)
    A = np.sqrt(γ*P/Q[0]) # Sound speed
    H = (Q[2] + P)/Q[0]    # Enthalpy
    h = 1/(2*H - U**2)

    R = np.asarray(
        [[1,U - A,H - U*A],
         [1,U    ,U**2/2],
         [1,U + A,H + U*A]]
    )

    L = np.asarray(
        [[U/(2*A) + U**2*h/2,2 - (2*H)*h,U**2*h/2 - U/(2*A)],
         [-U*h - 1/(2*A)   ,(2*U)*h    ,1/(2*A) - U*h],
         [h                ,-2*h       ,h]]
    )
    return R, L

def Eigenvalues(Q, γ):
    U = Q[:,1]/Q[:,0]
    P = Pressure_2d(Q, γ)
    A = np.sqrt(γ*P/Q[:,0])

    return np.stack([U-A, U, U+A],axis=1)

def MaximumEigenvalue(Q, γ):
    return np.max(np.abs(Eigenvalues(Q, γ)))

def ReconstructedFlux(F, Q, M):
    F_plus  = (F + M*Q)/2
    F_minus = (F - M*Q)/2

    F_half_plus  = WenoZ5ReconstructionLTR(F_plus)
    F_half_minus = WenoZ5ReconstructionLTR(np.flip(F_minus,axis=0))

    return F_half_plus + F_half_minus

def WenoZ5ReconstructionLTR(Q):
    # Constante para 
    ɛ = 1e-40
    
    u1, u2, u3, u4, u5=Q[:-1]
    
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

def FronteiraPeriodica(U):
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

def FronteiraReflexiva(U):
    U = np.concatenate([
        U[2:3,:],
        U[1:2,:],
        U[:1,:],
        U,
        U[-1:,:],
        U[-2:-1,:],
        U[-3:-2,:]],
        axis=0)
    
    U[:3,1]=-U[:3,1]
    U[-3:,1]=-U[-3:,1]
    
    return U

def FronteiraFixa(U):
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