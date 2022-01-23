import tensorflow as tf
import numpy as np

float_dtype='float64'

#@tf.function()
def Euler1D_step(Q, γ, Δx, CFL, Δt, BoundaryCondition):
    # 3rd order TVD Runge-Kutta scheme
    Q1 =    Q        -   Δt * FluxDerivative(Q, γ, Δx, BoundaryCondition)
    Q2 = (3*Q +   Q1 -   Δt * FluxDerivative(Q1, γ, Δx, BoundaryCondition))/4
    Q  = (  Q + 2*Q2 - 2*Δt * FluxDerivative(Q2, γ, Δx, BoundaryCondition))/3
    return Q


def Euler1D(Q, γ, Δx, CFL, FinalTime, BoundaryCondition):
    Time = tf.Variable(tf.zeros(Q.shape[1],dtype=float_dtype))
    while tf.math.reduce_any(Time < FinalTime):
        Δt = CFL*Δx/MaximumEigenvalue(Q, γ)
        Δt = tf.where(Δt>(FinalTime - Time),FinalTime,Δt)
        Q=Euler1D_step(Q, γ, Δx, CFL, Δt, BoundaryCondition)
        Time.assign(Time+Δt);
    return Q

def FluxDerivative(Q, γ, Δx, BoundaryCondition):
    Ord = 5 # The order of the scheme
    Q = BoundaryCondition(Q)

    N = Q.shape[1]
#     F_half = tf.zeros([Q.shape[0],N-Ord, 3],float_dtype)
#     G_half = tf.zeros([Q.shape[0],1, 3],float_dtype)
    #Qi = np.zeros([Ord+1, 3])

    M = MaximumEigenvalue(Q, γ)
    slice_Q= lambda i: Q[:,i:i+Ord+1,:]
    Qi=tf.map_fn(slice_Q,tf.range(N-Ord),fn_output_signature=Q.dtype)
    Qi=tf.transpose(Qi,[1,0,2,3])
    
    Λ = Eigenvalues(Qi, γ)
    
    r = 2
    Qa=(Qi[:,:,r,:] + Qi[:,:,r+1,:])/2;

    R, L = Eigensystem(Qa, γ)

    W = tf.matmul(Qi,L)       # Transforms into characteristic variables
    G = Λ*W       # The flux for the characteristic variables is Λ * L*Q
    M = tf.expand_dims(M[2:N-3],0)
    M = tf.expand_dims(M,2)
    M = tf.expand_dims(M,3)

    G_half= ReconstructedFlux(G, W, M)

    F_half = tf.matmul(G_half,R) # Brings back to conservative variables

    return (F_half[:,1:,:] - F_half[:,:-1,:])/Δx # Derivative of Flux

def Pressure(Q, γ):
    Q0,Q1,Q2=tf.unstack(Q,axis=-1)
    a=γ-1.0
    b=Q1**2
    c=Q0/2
    d=b/c
    e=Q2-d
    
    return a*e

def Eigensystem(Q, γ):
    Q0,Q1,Q2=tf.unstack(Q,axis=-1)
    
    U = Q1/Q0
    P = Pressure(Q, γ)
    A = tf.math.sqrt(γ*P/Q0) # Sound speed
    H = (Q2 + P)/Q0    # Enthalpy
    h = 1/(2*H - U**2)
    
    ones_ref=tf.ones(tf.shape(U),dtype=U.dtype)
    R1c=tf.stack([ones_ref,U-A,H - U*A],axis=-2)
    R2c=tf.stack([ones_ref,U  ,U**2/2 ],axis=-2)
    R3c=tf.stack([ones_ref,U+A,H + U*A],axis=-2)
    
    R = tf.stack([R1c,R2c,R3c],axis=-2)
    
    L1c=tf.stack([U/(2*A) + U**2*h/2,2 - (2*H)*h,U**2*h/2 - U/(2*A)],axis=2)
    L2c=tf.stack([-U*h - 1/(2*A)    ,(2*U)*h    ,1/(2*A) - U*h     ],axis=2)
    L3c=tf.stack([h                 ,-2*h       ,h                 ],axis=2)
    
    L = tf.stack([L1c,L2c,L3c],axis=-2)
    return R,L

def Eigenvalues(Q, γ):
    Q0,Q1,Q2=tf.unstack(Q,axis=-1)
    
    U = Q1/Q0
    P = Pressure(Q, γ)
    A = tf.math.sqrt(γ*P/Q0)

    return tf.stack([U-A, U, U+A],axis=-1)

def MaximumEigenvalue(Q, γ):
    eig_val=tf.abs(Eigenvalues(Q, γ))
    return tf.math.reduce_max(tf.math.reduce_max(eig_val,axis=-1),axis=0)

def ReconstructedFlux(F, Q, M):
    F_plus  = (F + M*Q)/2
    F_minus = (F - M*Q)/2

    F_half_plus  = WenoZ5ReconstructionLTR(F_plus)
    F_half_minus = WenoZ5ReconstructionLTR(tf.reverse(F_minus,axis=2))

    return F_half_plus + F_half_minus

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
A = tf.cast(tf.stack([A0,A1,A2], axis=0), dtype=float_dtype) 
A = tf.expand_dims(A, axis=1)
A = tf.expand_dims(A, axis=1)

B = tf.constant([[1,0,0],[0,6,0],[0,0,3]], dtype=float_dtype)/10                # Matriz B
C = tf.constant([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]], dtype=float_dtype)/6 # Matriz C
#C = tf.transpose(C)

ω0 = tf.constant([[[1/10, 6/10, 3/10]]], dtype = float_dtype)
α0 = tf.constant(10.0, dtype = float_dtype)

b0  = tf.math.lgamma(α0)
b0 -= tf.math.reduce_sum(tf.math.lgamma(α0*ω0))
b0 -= tf.math.lgamma(tf.constant(3, float_dtype))
a0  = α0*ω0-1

def WenoZ5ReconstructionLTR(Q0):
    ɛ = 10.0**(-40)
    
#     # Calcula os indicadores de suavidade locais
#     Q = tf.stack([Q, Q, Q], axis=0)
    Q=Q0[:,:,:-1]
    
    β = tf.math.reduce_sum(Q * (A @ Q), axis=2)
    #β = β*(beta_weight+0.01)
    
    # Calcula o indicador de suavidade global
    τ = tf.abs(β[:,:,0:1] - β[:,:,2:3])
    
    # Calcula os pesos do WENO-Z
    α    = (1 + (τ/(β + ɛ))**2) @ B
    soma = tf.math.reduce_sum(α, axis=2, keepdims=True)
    ω    = α / soma
    
    # Calcula os fhat em cada subestêncil
    fhat =  C @ Q
    
    # Calcula o fhat do estêncil todo
    fhat = tf.math.reduce_sum(tf.broadcast_to(ω,tf.shape(fhat)) * fhat, axis=2, keepdims=True)
    
    return fhat

def FronteiraPeriodica(U):
    U = tf.concat([
        U[:,-3:-2,:],
        U[:,-2:-1,:],
        U[:,-1:,:],
        U,
        U[:,:1,:],
        U[:,1:2,:],
        U[:,2:3,:]],
        axis=1)
    
    return U

def FronteiraReflexiva(U):
    U = tf.concat([
        U[:,2:3,:]*tf.constant([[[1,-1,1]]]),
        U[:,1:2,:]*tf.constant([[[1,-1,1]]]),
        U[:,:1,:]*tf.constant([[[1,-1,1]]]),
        U,
        U[:,-1:,:]*tf.constant([[[1,-1,1]]]),
        U[:,-2:-1,:]*tf.constant([[[1,-1,1]]]),
        U[:,-3:-2,:]*tf.constant([[[1,-1,1]]])],
        axis=1)
    
    return U

def FronteiraFixa(U):
    U = tf.concat([
        U[:,0:1,:],
        U[:,0:1,:],
        U[:,0:1,:],
        U,
        U[:,-1:,:],
        U[:,-1:,:],
        U[:,-1:,:]],
        axis=1)
    
    return U