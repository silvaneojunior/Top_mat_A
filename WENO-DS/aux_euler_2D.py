from aux_base import dtype, const, API_Numpy

import numpy as np

## Condições iniciais

def MalhaRetangular(xlim,ylim,Δ,Δx=None,Δy=None):
    if Δ is None and (Δx is None or Δy is None):
        raise(TypeError("Se Δx ou Δy é None, Δ precisa ser passado."))
    if Δx is None:
        Δx=Δ
    if Δy is None:
        Δy=Δ
    Δx = Δy; Δy = Δx
    x = API_Numpy.arange(xlim[0],xlim[1],Δx,dtype=dtype)
    y = API_Numpy.arange(ylim[0],ylim[1],Δy,dtype=dtype)

    X = API_Numpy.stack([x]*y.shape[0],axis=1)
    Y = API_Numpy.stack([y]*x.shape[0],axis=0)

    return X, Y, Δx, Δy

def CondiçãoInicialShockEntropy2D(X, Y, γ,θ):
   R = API_Numpy.zeros(X.shape,dtype=dtype)
   P = API_Numpy.ones(X.shape,dtype=dtype)
   U = API_Numpy.zeros(X.shape,dtype=dtype)
   V = API_Numpy.zeros(X.shape,dtype=dtype)

   I = API_Numpy.ones(X.shape,dtype=dtype)

   l=X < -4
   r=X >=-4

   R[l]=I[l]*const(27.0,API_Numpy)/7.0
   U[l]=I[l]*const(4,API_Numpy)*API_Numpy.sqrt(const(35.0,API_Numpy))/9.0
   P[l]=I[l]*const(31.0,API_Numpy)/3.0

   R[r] = 1.0 + API_Numpy.sin(X[r]*API_Numpy.cos(const(θ,API_Numpy))*2.0*API_Numpy.pi + Y[r]*API_Numpy.sin(const(θ,API_Numpy))*2.0*API_Numpy.pi)/5.0

   E = P/(γ-1.0) + R*(U**2 + V**2)/2.0
   Q0 = API_Numpy.stack([R, R*U, R*V, E],axis=0)
   return Q0

def CondiçãoInicialShockEntropy2D_N(N, γ,θ):
   X, Y, Δx, Δy = MalhaRetangular(Δ=const(10,API_Numpy)/N,xlim=(-5,5),ylim=(-1,1))
   Q0 = CondiçãoInicialShockEntropy2D(X, Y, γ,θ)
   return X, Y, Δx, Δy, Q0

def CondiçãoInicialDoubleMach(X, Y, γ,x0):
   Reg1=X<x0+Y/np.sqrt(const(3,API_Numpy))
   Reg2=X>=x0+Y/np.sqrt(const(3,API_Numpy))

   R0 = API_Numpy.zeros(X.shape,dtype=dtype)
   P0 = API_Numpy.ones(X.shape,dtype=dtype)
   U0 = API_Numpy.zeros(X.shape,dtype=dtype)
   V0 = API_Numpy.zeros(X.shape,dtype=dtype)

   R0[Reg1]=8
   U0[Reg1]=8.25*np.sqrt(const(3,API_Numpy))/2
   V0[Reg1]=-8.25/2
   P0[Reg1]=116.5

   8.25

   R0[Reg2]=const(14,API_Numpy)/10

   Q0 = API_Numpy.stack([R0, R0*U0, R0*V0, P0/(γ-1)+R0*(U0**2+V0**2)/2],axis=0)
   return Q0

def CondiçãoInicialDoubleMach_N(N, γ,x0):
   X, Y, Δx, Δy = MalhaRetangular(Δ=const(1,API_Numpy)/N,xlim=(0,4+const(1,API_Numpy)/N),ylim=(0,1+const(1,API_Numpy)/N))
   Q0 = CondiçãoInicialDoubleMach(X, Y, γ,x0)
   return X, Y, Δx, Δy, Q0

def CondiçãoInicialRayleighTaylor(X, Y, γ):
   R = API_Numpy.zeros(X.shape,dtype=dtype)
   P = API_Numpy.zeros(X.shape,dtype=dtype)
   U = API_Numpy.zeros(X.shape,dtype=dtype)
   V = API_Numpy.zeros(X.shape,dtype=dtype)
   for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if Y[i,j] < 0.5:
         R[i,j] = 2.0
         P[i,j] = 2.0*Y[i,j] + 1.0
      else:
         R[i,j] = 1.0
         P[i,j] = Y[i,j] + 1.5
         
      a = API_Numpy.sqrt(γ * P[i,j] / R[i,j])
      V[i,j] = -const(25,API_Numpy)/1000 * a * API_Numpy.cos(8.0*API_Numpy.pi*X[i,j])

   E = P/(γ-1.0) + R*(U**2 + V**2)/2.0
   Q0 = API_Numpy.stack([R, R*U, R*V, E],axis=0)
   n=X.shape[0]//2
   Q0[:,-n:,:]=API_Numpy.reverse(Q0[:,:n,:].copy(),axis=1)
   return Q0


def CondiçãoInicialRayleighTaylor_N(N, γ):
   X, Y, Δx, Δy = MalhaRetangular(Δ=const(1,API_Numpy)/N,xlim=(0,0.25+1/N),ylim=(1/N,1))
   Q0 = CondiçãoInicialRayleighTaylor(X, Y, γ)
   return X, Y, Δx, Δy, Q0

## Condições de fronteira

class GhostPoints:
    def __init__(self,dtype=dtype):
        self.dtype=dtype

class ShockEntropy2DGhostPointsX(GhostPoints):
    def __init__(self,γ,Δx,y,θ,dtype=dtype):
        super(ShockEntropy2DGhostPointsX,self).__init__(dtype=dtype)
        self.γ=API_Numpy.cast(γ,self.dtype)
        self.Δx=API_Numpy.cast(Δx,self.dtype)
        self.y=y
        self.θ=API_Numpy.cast(θ,self.dtype)
    def __call__(self,Q,API,t=None):
        ρl = API.cast(27,self.dtype)/7.0
        ul = 4.0*API.sqrt(API.cast(35,self.dtype))/9.0
        pl = API.cast(31,self.dtype)/3.0
        El = pl/(self.γ-1.0) + ρl*(ul**2)/2.0
        El = API.cast(El,self.dtype)

        raw_ref_shape=API.shape(Q)[:-3]
        out_shape=API.concat([API.shape(Q)[:-2],[3],API.shape(Q)[-1:]],axis=0)
        ref_shape=API.concat([API.ones(API.shape(raw_ref_shape),dtype='int32'),[4],[1],[1]],axis=0)

        vals_left=API.reshape([ρl, ρl*ul, 0.0, El],ref_shape)

        Xr = API.reshape([5.0+API.cast(self.Δx,dtype=dtype), 5.0+2*API.cast(self.Δx,dtype=dtype), 5.0+3*API.cast(self.Δx,dtype=dtype)], [3, 1])
        Yr = self.y[...,:3,:]
        ρr = 1.0 + API.sin(5.0*Xr*API.cos(self.θ) + 5.0*Yr*API.sin(self.θ))/5.0
        pr = API.ones(ρr.shape,dtype=self.dtype)
        ur = API.zeros(ρr.shape,dtype=self.dtype)
        vr = API.zeros(ρr.shape,dtype=self.dtype)
        Er = pr/(self.γ-1.0)
        Qr = API.stack([ρr, ur, vr, Er],axis=-3)

        Qg = [API.ones(out_shape,dtype=self.dtype)*API.cast(vals_left,self.dtype),
        Q,
        API.cast(Qr,self.dtype)]

        return API.concat(Qg,-2)

class ShockEntropy2DGhostPointsY(GhostPoints):
    def __call__(self,Q,API,t=None):
        Qg = API.concat(
            [Q[...,:,-3:],
                Q,
                Q[...,:,:3]],axis=-1)
        return Qg

class DoubleMachGhostPointsX(GhostPoints):
    def __init__(self,L,R,γ,dtype=dtype):
        super(DoubleMachGhostPointsX,self).__init__(dtype=dtype)
        self.γ=API_Numpy.cast(γ,dtype=dtype)
        self.L=L
        self.R=R
        self.uL=np.asarray([
            8,
            4*8.25*np.sqrt(API_Numpy.cast(3,dtype=dtype)),
            -4*8.25,
            116.5/(γ - 1) + 4*(API_Numpy.cast(8.25,dtype=dtype))**2
            ],
            dtype=self.dtype)
    def __call__(self,Q,API,t=None):
        pre_shape=API.shape(Q)[:-3]
        uL_shape=API.cast(API.concat([pre_shape,(4,1,1)],axis=0),'int32')
        ones_shape=API.cast(API.concat([pre_shape,(1,self.L,API.shape(Q)[-1])],axis=0),'int32')
        UL=API.reshape(self.uL,uL_shape)*API.ones(ones_shape,dtype=self.dtype)
        UR=API.flip(Q[...,-self.R-1:-1,:],axis=(-2,))

        Qg = [UL,Q,UR]

        return API.concat(Qg,-2)

class DoubleMachGhostPointsY(GhostPoints):
    def __init__(self,L,R,γ,x0,x,dtype=dtype):
        super(DoubleMachGhostPointsY,self).__init__(dtype=dtype)
        self.γ=API_Numpy.cast(γ,dtype=dtype)
        self.L=L
        self.R=R
        self.x0=x0
        self.x=x[:,0]
        self.xL=x[:,0]<x0
        self.xR=x[:,0]>=x0
        self.uL=np.asarray([
            8,
            4*8.25*np.sqrt(API_Numpy.cast(3,dtype=dtype)),
            -4*8.25,
            116.5/(γ - 1) + 4*(API_Numpy.cast(8.25,dtype=dtype))**2
            ],
            dtype=self.dtype)
        self.uR=np.asarray([
            API_Numpy.cast(14,dtype=dtype)/10,
            0,
            0,
            1/(γ-1)
            ],
            dtype=self.dtype)
        self.Reflect=np.asarray([1,1,-1,1],dtype=self.dtype) # indice da dimensão que será refletida
    def __call__(self,Q,API,t):
        Lu=API.shape(Q)[-3]
        Lx=API.shape(Q)[-2]
        pre_shape=API.shape(Q)[:-3]

        uLR_shape=API.cast(API.concat([pre_shape,(1,Lx,1)],axis=0),'int32')
        ones_L_shape=API.cast(API.concat([pre_shape,(Lu,1,self.L)],axis=0),'int32')
        ones_R_shape=API.cast(API.concat([pre_shape,(Lu,1,self.R)],axis=0),'int32')
        var_L_shape=API.cast(API.concat([pre_shape,(1,Lx,self.L)],axis=0),'int32')
        var_R_shape=API.cast(API.concat([pre_shape,(1,Lx,self.R)],axis=0),'int32')

        UL_condL=API.cast(API.reshape(self.xL,uLR_shape),self.dtype)
        UL_condL=API.cast(UL_condL*API.ones(ones_L_shape,dtype=self.dtype),'bool')
        UL_condR=API.cast(API.reshape(self.xR,uLR_shape),self.dtype)
        UL_condR=API.cast(UL_condR*API.ones(ones_L_shape,dtype=self.dtype),'bool')

        reflect_shape=API.cast(API.concat([pre_shape,(4,1,1)],axis=0),'int32')
        reflect_value=API.reshape(self.Reflect,reflect_shape)*API.ones(var_L_shape,dtype=self.dtype)
        
        UL=API.reshape(self.uL,reflect_shape)*API.ones(var_L_shape,dtype=self.dtype)
        UL=API.where(
            UL_condL,
            UL,
            reflect_value*API.flip(Q[...,:,1:self.L+1],axis=(-1,))
            )

        s=self.x0 + (1 + 20*t)/np.sqrt(API_Numpy.cast(3,dtype=dtype))
        xR=self.x<s
        UR_condL=API.cast(API.reshape(xR,uLR_shape),self.dtype)
        UR_condL=API.cast(UR_condL*API.ones(ones_R_shape,dtype=self.dtype),'bool')
        #xR=self.x>=s
        UR=API.where(
            UR_condL,
            API.reshape(self.uL,reflect_shape)*API.ones(var_R_shape,dtype=self.dtype),
            API.reshape(self.uR,reflect_shape)*API.ones(var_R_shape,dtype=self.dtype))

        Qg = [UL,Q,UR]

        return API.concat(Qg,-1)

class RayleighTaylorGhostPointsX(GhostPoints):
    def __call__(self,Q,API,t=None):
        left=API.stack([Q[...,0,:3,:],-Q[...,1,:3,:],Q[...,2,:3,:],Q[...,3,:3,:]],axis=-3)
        left=API.reverse(left,axis=[-2])

        center=Q

        right=API.stack([Q[...,0,-3:,:],-Q[...,1,-3:,:],Q[...,2,-3:,:],Q[...,3,-3:,:]],axis=-3)
        right=API.reverse(right,axis=[-2])

        return API.concat([left,center,right],axis=-2)

class RayleighTaylorGhostPointsY(GhostPoints):
    def __init__(self,γ,dtype=dtype):
        super(RayleighTaylorGhostPointsY,self).__init__(dtype=dtype)
        self.γ=API_Numpy.cast(γ,dtype=dtype)
    def __call__(self,Q,API,t=None):
        raw_ref_shape=API.shape(Q)[:-3]
        out_shape=API.concat([API.shape(Q)[:-1],[3]],axis=0)
        ref_shape=API.concat([API.ones(API.shape(raw_ref_shape),dtype='int32'),[4],[1],[1]],axis=0)
        vals_left=API.reshape([2.0,0.0,0.0,1.0/(self.γ-1.0)],ref_shape)
        vals_right=API.reshape([1.0,0.0,0.0,2.5/(self.γ-1.0)],ref_shape)
        Qg = [API.ones(out_shape,dtype=self.dtype)*API.cast(vals_left,self.dtype),
                Q,
                API.ones(out_shape,dtype=self.dtype)*API.cast(vals_right,self.dtype)]
        return API.concat(Qg,-1)

## Gravidade

def NullForce(Q,API):
   return 0.0

def RayleighTaylorGravity(Q,API):
   g = -1.0
   # for i = 1:size(U,1)
   #    for j = 1:size(U,2)
   #       F[i,j,3] = -g * U[i,j,1]
   #       F[i,j,4] = -g * U[i,j,3]
   #    end
   # end
   Z = Q[...,0:1,:,:]*0
   F = API.concat([Z,Z,-g*Q[...,0:1,:,:],-g*Q[...,2:3,:,:]],axis=-3)
   return F