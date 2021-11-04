import tensorflow as tf
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import Null_net

float_pres='float64'

def Graph_Burgers(u, Δt, Δx, fronteira,network=Null_net.Network):
#     t=tf.constant(0.0,dtype=float_pres)

    Λ = tf.math.reduce_max(tf.abs(u),axis=1,keepdims=True)
#     Δt = Δx*CFL/Λ #Condicao de CFL - deve ser atualizada a cada passo de tempo
#     #SSP Runge-Kutta 3,3
#     Δt = tf.cond(t + Δt > t_final,lambda:  t_final - t,lambda:  Δt)

    u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira,network)
    Λ = tf.math.reduce_max(tf.abs(u1))
    u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira,network)) / 4.0
    Λ = tf.math.reduce_max(tf.abs(u2))
    u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira,network)) / 3.0
    #t = t + Δt
    return u
    
#     while t<t_final:
#         Λ = tf.math.reduce_max(tf.abs(u))
#         Δt = Δx*CFL/Λ #Condicao de CFL - deve ser atualizada a cada passo de tempo
#         #SSP Runge-Kutta 3,3
#         u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira,network)
#         Λ = tf.math.reduce_max(tf.abs(u1))
#         u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira,network)) / 4.0
#         Λ = tf.math.reduce_max(tf.abs(u2))
#         u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira,network)) / 3.0
#         t = t + Δt

#     n=10
#     Λ = 2.0
#     Δt = t_final/10
#     for i in range(10):
#         u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira,network)
#         u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira,network)) / 4.0
#         u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira,network)) / 3.0
#         t = t + Δt
#     return u

def Graph_Burgers2(u, t_final, Δx, CFL, fronteira,network=Null_net.Network):
    t=t_final*0

    while tf.math.reduce_any(t<t_final):
        Λ = tf.math.reduce_max(tf.abs(u),axis=1,keepdims=True)
        Δt = Δx*CFL/Λ #Condicao de CFL - deve ser atualizada a cada passo de tempo
        #SSP Runge-Kutta 3,3
        Δt=tf.where(t + Δt > t_final,t_final - t,Δt)
        
        u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira,network)
        Λ = tf.math.reduce_max(tf.abs(u1))
        u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira,network)) / 4.0
        Λ = tf.math.reduce_max(tf.abs(u2))
        u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira,network)) / 3.0
        t = t + Δt
    return u

#     n=10
#     Λ = 2.0
#     Δt = t_final/10
#     for i in range(10):
#         u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira,network)
#         u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira,network)) / 4.0
#         u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira,network)) / 3.0
#         t = t + Δt
#     return u

    

@tf.function
def Burgers(u, Δt, Δx, CFL, fronteira,network=Null_net.Network):
    return Graph_Burgers2(u, Δt, Δx, CFL, fronteira,network)

def FronteiraFixa(U):
    return tf.concat([U[:,0:1],
                           U[:,0:1],
                           U[:,0:1],
                           U,
                           U[:,-1:],
                           U[:,-1:],
                           U[:,-1:]],
                         axis=1)


def FronteiraPeriodica(U):
    placeholder=tf.concat([U[:,-3:],U,U[:,:3]],axis=1)
    return placeholder

def DerivadaEspacial(U, Δx, Λ, AdicionaGhostPoints,network):
    Fhat = []
    U = AdicionaGhostPoints(U) # Adiciona ghost cells de acordo com as condições de fronteira
    
    U_diff=tf.concat([U[:,2:]-U[:,:-2],U[:,2:]-2*U[:,1:-1]+U[:,:-2]],axis=2)
    
    beta_weight=network(U_diff)
    U=U[:,:,0]
    beta_weight=beta_weight[:,:,0]
    
    helper1=lambda i: U[:,i:i+6]
    helper2=lambda i: beta_weight[:,i:i+3]
    U_full=tf.concat(tf.map_fn(helper1,tf.range(U.shape[1]-5),fn_output_signature=U.dtype),axis=0)
    beta_weight_full=tf.concat(
        tf.map_fn(helper2,
                  tf.range(beta_weight.shape[1]-2),
                  fn_output_signature=beta_weight.dtype
                 ),
        axis=0)
    U_full=tf.transpose(U_full,[1,0,2])
    beta_weight_full=tf.transpose(beta_weight_full,[1,0,2])
    
    M = tf.math.reduce_max(tf.abs(U_full),axis=2,keepdims=True) #Λ
    #M = 1
    
    #f_plus = (U_full + M*U_full)/2
    #f_minus = (U_full - M*U_full)/2
    
    f_plus = (U_full**2/2 + M*U_full) / 2
    f_minus = (U_full**2/2 - M*U_full) / 2
    
    f_half_minus = WenoZ5ReconstructionMinus(f_plus[:,:,:-1],beta_weight_full[:,:-1,:]) # Aplicar WENO em cada variável característica separadamente
    f_half_plus = WenoZ5ReconstructionPlus(f_minus[:,:,1:],beta_weight_full[:,1:,:])
    
    Fhat=(f_half_minus + f_half_plus)
    
    return tf.transpose((Fhat[:,1:] - Fhat[:,:-1]) / Δx)

a=np.asarray([[0.5],[-2],[3/2],[0],[0]])
b=np.asarray([[1],[-2],[1],[0],[0]])
c=13/12

A0=np.dot(a,a.T)+c*np.dot(b,b.T)

a=np.asarray([[0],[-0.5],[0],[0.5],[0]])
b=np.asarray([[0],[1],[-2],[1],[0]])
c=13/12

A1=np.dot(a,a.T)+c*np.dot(b,b.T)

a=np.asarray([[0],[0],[-3/2],[2],[-1/2]])
b=np.asarray([[0],[0],[1],[-2],[1]])
c=13/12

A2=np.dot(a,a.T)+c*np.dot(b,b.T)

A=tf.cast(tf.stack([A0,A1,A2],axis=0),dtype=float_pres)
A=tf.expand_dims(A,axis=1)

B=tf.constant([[1,0,0],[0,6,0],[0,0,3]],dtype=float_pres)/10
C=tf.transpose(
    tf.constant([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]],dtype=float_pres)/6
)

def WenoZ5ReconstructionMinus(u0,beta_weight):#u1, u2, u3, u4, u5):
    ɛ = 10.0**(-40)
    # Calcula os indicadores de suavidade locais
    u=tf.stack([u0,u0,u0],axis=0)

    β = tf.math.reduce_sum(u * (u @ A),axis=3)
    β = tf.transpose(β,[1,2,0])
    #β = β*(beta_weight+0.01)

    #β = beta_weight

    # Calcula o indicador de suavidade global
    τ = tf.abs(β[:,:,0:1] - β[:,:,2:3])
    # Calcula os pesos do WENO-Z
    α = (1 + (τ/(β + ɛ))**2) @ B
    soma = tf.math.reduce_sum(α,axis=2,keepdims=True)
    ω = α / soma
    ω = (1-beta_weight)*ω + beta_weight*tf.constant([[[1,6,3]]],dtype=float_pres)/10
    # Calcula os fhat em cada subestêncil
    fhat=u0 @ C
    #Finalmente, calcula o fhat do estêncil todo
    final_weigth=tf.transpose(tf.math.reduce_sum(ω * fhat,axis=2,keepdims=True))
    return final_weigth

def WenoZ5ReconstructionPlus(u0,beta_weight):
    return WenoZ5ReconstructionMinus(tf.reverse(u0,axis=[2]),tf.reverse(beta_weight,axis=[2]))

def Gera_dados():
    Δx = 0.01
    x = tf.range(-2,2,Δx,
                dtype=float_pres)
    
    u_list=[]
    while len(u_list)<10000:
        k1=tf.cast(
            tf.random.uniform([1],0,20,dtype='int32'),
            dtype=float_pres
        )
        k2=tf.cast(
            tf.random.uniform([1],0,20,dtype='int32'),
            dtype=float_pres
        )
        a=tf.random.uniform([1],0,1,
                            dtype=float_pres)
        b=tf.random.uniform([1],0,2,
                            dtype=float_pres)

        u1 = a*tf.expand_dims(tf.math.sin(k1*pi*x),axis=1)
        u2 = (1-a)*tf.expand_dims(tf.math.sin(k2*pi*x),axis=1)

        u=b*(u1+u2)
        #u=tf.expand_dims(u,axis=0)
        u_list.append(u)
    u=tf.stack(u_list,axis=0).numpy()
    print(u.shape)
        
    
    CFL = 0.5

    Δt_max = 0.01          # Δt da animação
    T = tf.range(0.0,2,Δt_max,
                dtype=float_pres)    # Frames da animação

    U = []
    U.append([u])
    #t = 0.0
    #i = 2
    
    t=tf.constant(0.0,dtype=float_pres)
    Δx2=4*Δx
    
    #while t < T[-1]:
    for i in range(200):
        print(i)
        short_u=tf.gather(u,tf.range(100)*4,axis=1)
        Λ = tf.math.reduce_max(tf.abs(short_u),axis=1,keepdims=True)
        Δt = Δx2*CFL/Λ
        Δt = tf.where(Δt > Δt_max,Δt_max,Δt)
        
        u = Burgers(u, Δt, Δx,CFL, FronteiraPeriodica).numpy()
        U.append([u])
        #i += 1
        #t += Δt
    return U

def AnimaçãoBurgers():
    Δx = 0.01
    x = tf.range(-2,2,Δx,
                dtype=float_pres)
    
    k1=tf.cast(
        tf.random.uniform([1],0,20,dtype='int32'),
        dtype=float_pres
    )
    k2=tf.cast(
        tf.random.uniform([1],0,20,dtype='int32'),
        dtype=float_pres
    )
    a=tf.random.uniform([1],0,1,
                        dtype=float_pres)
    b=tf.random.uniform([1],0,2,
                        dtype=float_pres)
#     k1=1.0
#     k2=2.0
#     a=0.5
#     b=0.5
    u1 = a*tf.expand_dims(tf.math.sin(k1*pi*x),axis=1)
    u2 = (1-a)*tf.expand_dims(tf.math.sin(k2*pi*x),axis=1)
    u=b*(u1+u2)
    
#     u=tf.expand_dims(tf.exp(-10*x**2),axis=1)
    u=tf.expand_dims(u,axis=0)
    
    CFL = 0.5

    Δt = 0.01          # Δt da animação
    T = tf.range(0.0,2,Δt,
                 dtype=float_pres)    # Frames da animação

    t = 0.0
    
    fig = plt.figure(1, constrained_layout=True,figsize=(6,6))
    ax = fig.add_subplot(1,1,1);
    ax.set_ylim(-2, 2);
    #ax.set_xlim(0,1);
    line=ax.plot(x,tf.squeeze(u))
    hfig = display(fig, display_id=True)
    Δt_list=tf.zeros([0],dtype=float_pres)
    
    
    while t < T[-1]:
        u, elem_Δt = Burgers(u, Δt, Δx, CFL, FronteiraPeriodica)[1:]
        squeezed_u=tf.squeeze(u)
        Δt_list=tf.concat([Δt_list,elem_Δt],axis=0)
        t += Δt
        
        if tf.math.reduce_any(tf.math.is_nan(squeezed_u)):
            line=ax.plot(x,tf.cast(tf.math.is_nan(squeezed_u),'float32').numpy(),'ro')
            fig.canvas.draw()
            hfig.update(fig)
            return(Δt_list)
        #line.set_xdata(x)
        line[0].set_ydata(squeezed_u.numpy())
        
        fig.canvas.draw()
        hfig.update(fig)
        #time.sleep(0.01);
    return Δt_list