from aux_mapping import *

"""
Obtendo matrizes de constantes convenintes para executar o WENO-Z
utilizando operações tensoriais, uma vez que permite a integração
com o tensorflow
"""
ɛ = 10.0**(-40)

B = np.asarray([[1,0,0],[0,6,0],[0,0,3]], dtype=float_pres)/10                # Matriz B
C = np.asarray([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]], dtype=float_pres)/6 # Matriz C
C = np.transpose(C)

class equation:
    def __init__(self, API, WENO, network, mapping=null_mapping, map_function=lambda x:x, p=2):
        
        self.API     = API
        self.WENO    = WENO
        self.network = network
        self.p       = p

        self.mapping      = mapping
        self.map_function = map_function

    def Get_weights_graph(self, u, Δx, AdicionaGhostPoints=None):

        if AdicionaGhostPoints is not None:
            u = AdicionaGhostPoints(u, self.API, n=2)
            u = slicer(u,5,self.API)
            
        if self.network is not None:
            δ = self.network(self.API.concat([
                u[...,0:1,0]  , 
                u[...,0:1,1]  , 
                u[...,2]      , 
                u[...,-1:,-2] , 
                u[...,-1:,-1]
            ], axis=-1))
            δ = slicer(δ, 3, self.API)
            
        else:
            δ=(u-u)[...,1:-1]+1-0.1
        
        # Calcula os indicadores de suavidade locais
        β0 = self.API.square( 1/2.0*u[...,0] - 2*u[...,1] + 3/2.0*u[...,2]) + 13/12.0*self.API.square(u[...,0] - 2*u[...,1] + u[...,2])
        β1 = self.API.square(-1/2.0*u[...,1]              + 1/2.0*u[...,3]) + 13/12.0*self.API.square(u[...,1] - 2*u[...,2] + u[...,3])
        β2 = self.API.square(-3/2.0*u[...,2] + 2*u[...,3] - 1/2.0*u[...,4]) + 13/12.0*self.API.square(u[...,2] - 2*u[...,3] + u[...,4])
        
        β = self.API.stack([β0, β1, β2], axis=-1)
        
        α    = self.WENO(β, δ, self.API, Δx, mapping=self.mapping, map_function=self.map_function)
        soma = self.API.sum(α, axis=-1, keepdims=True)
        ω    = α / soma

        return ω, α, β, δ

    def ReconstructionMinus(self, u, Δx):
        
        ω, α, β, δ = self.Get_weights_graph(u, Δx)
        # Calcula os fhat em cada subestêncil
        fhat = self.API.matmul(u, C)
        # Calcula o fhat do estêncil todo
        fhat = self.API.sum(ω * fhat, axis=-1)
        return fhat

    def ReconstructionPlus(self, u, Δx):
        fhat = self.ReconstructionMinus(self.API.reverse(u, axis=[-1]) ,Δx)
        return fhat
        
    def flux_sep(self,U):
        pass

    def DerivadaEspacial(self, U, Δx, AdicionaGhostPoints):
        U = AdicionaGhostPoints(U, self.API) # Estende a malha de pontos de acordo com as condições de fronteira

        f_plus,f_minus = self.flux_sep(U)

        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = self.ReconstructionMinus(f_plus[...,:-1], Δx) 
        f_half_plus  = self.ReconstructionPlus( f_minus[...,1:], Δx)
        Fhat         = (f_half_minus + f_half_plus)

        # Calculando uma estimava da derivada a partir de diferenças finitas
        Fhat = (Fhat[...,1:] - Fhat[...,:-1]) / Δx

        return Fhat

class transp_equation(equation):
    
    def maximum_speed(self, U):
        return self.API.cast(1, float_pres)
    
    def flux_sep(self, U):
        
        U = slicer(U, 6, self.API)
        M = self.maximum_speed(U)  # Valor utilizado para realizar a separação de fluxo
        
        # Setup para equação do transporte
        f_plus  = (U + M*U)/2 # Fluxo positivo
        f_minus = (U - M*U)/2 # Fluxo negativo
        
        return f_plus, f_minus

class burgers_equation(equation):
    
    def maximum_speed(self,U):
        return self.API.max(self.API.abs(U), axis=-1, keepdims=True)
    
    def flux_sep(self,U):
        
        # Valor utilizado para realizar a separação de fluxo
        M = self.maximum_speed(U)
        M = self.API.expand_dims(M,-1)
        U = slicer(U, 6, self.API)
        
        # Setup para equação de Burguers
        f_plus  = (U**2/2 + M*U) / 2  # Fluxo positivo
        f_minus = (U**2/2 - M*U) / 2  # Fluxo negativo
        
        return f_plus, f_minus

class diff_equation(equation):
    
    def maximum_speed(self,U):
        return 1
    
    def flux_sep(self,U):

        U = slicer(U, 6, self.API)
        f_plus  = U / 2  # Fluxo positivo
        f_minus = U / 2  # Fluxo negativo
        
        return f_plus, f_minus

# Dummy comment # Dummy comment of the dummy comment
γ = 1.4

class euler_equation(equation):
    
    def Pressure(self, Q):
        
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        a = γ-1.0
        b = Q1**2
        c = 2*Q0
        d = b/c
        e = Q2-d
        
        return a*e

    def Eigensystem(self, Q):
        
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        U = Q1/Q0
        P = self.Pressure(Q)
        A = self.API.sqrt(γ*P/Q0) # Sound speed
        H = (Q2 + P)/Q0           # Enthalpy
        h = 1/(2*H - U**2)
        
        ones_ref = self.API.ones(self.API.shape(U), dtype=U.dtype)
        
        R1c = self.API.stack([ones_ref, U-A, H - U*A], axis=-2)
        R2c = self.API.stack([ones_ref, U  , U**2/2 ], axis=-2)
        R3c = self.API.stack([ones_ref, U+A, H + U*A], axis=-2)
        
        R = self.API.stack([R1c, R2c, R3c], axis=-2)
        
        L1c = self.API.stack([U/(2*A) + U**2*h/2, 2 - (2*H)*h, U**2*h/2 - U/(2*A)], axis=-2)
        L2c = self.API.stack([-U*h - 1/(2*A)    , (2*U)*h    , 1/(2*A) - U*h     ], axis=-2)
        L3c = self.API.stack([h                 , -2*h       , h                 ], axis=-2)
        
        L = self.API.stack([L1c, L2c, L3c], axis=-2)
        
        return R, L

    def Eigenvalues(self, Q):
        
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        U = Q1/Q0
        P = self.Pressure(Q)
        A = self.API.sqrt(γ*P/Q0)

        return self.API.stack([U-A, U, U+A], axis=-2)

    def maximum_speed(self, U):
        eig_val = self.API.abs(self.Eigenvalues(U))
        return self.API.max(eig_val, axis=(-1,-2), keepdims=True)

    def ReconstructedFlux(self, F, Q, M, Δx):
        
#         M = self.API.expand_dims(M,axis=-3)
        F_plus  = (F + M*Q)/2
        F_minus = (F - M*Q)/2

        F_plus  = self.API.einsum('...ijk->...jik', F_plus)
        F_minus = self.API.einsum('...ijk->...jik', F_minus)
        
        F_half_plus  = self.ReconstructionMinus(F_plus[...,:-1], Δx)
        F_half_minus = self.ReconstructionPlus( F_minus[...,1:], Δx)

        return F_half_plus + F_half_minus

    def DerivadaEspacial(self, Q, Δx, AdicionaGhostPoints):
        
        Ord = 5 # The order of the scheme
        Q   = AdicionaGhostPoints(Q, self.API)

        Qi = slicer(Q, 6, self.API)
        Qi = self.API.einsum('...ijk->...jik', Qi)
        M  = self.maximum_speed(Qi)
        
        Λ = self.Eigenvalues(Qi)
        
        r  = 2
        Qa = (Qi[...,r] + Qi[...,r+1])/2
        Qa = self.API.einsum('...ij->...ji', Qa)
        
        R, L = self.Eigensystem(Qa)
            
        W = self.API.einsum('...nvc,...uvn -> ...nuc', Qi, L) # Transforms into characteristic variables
        G = Λ*W                                               # The flux for the characteristic variables is Λ * L*Q
        
        G_half = self.ReconstructedFlux(G, W, M, Δx)
        F_half = self.API.einsum('...vn,...uvn -> ...un', G_half, R) # Brings back to conservative variables
        
        return (F_half[...,1:] - F_half[...,:-1])/Δx # Derivative of Flux

def slicer(data, n, API):
    helper = lambda i: data[...,i:i+n]

    data_sliced = API.einsum(
    'i...j -> ...ij',
        API.map_fn(
            helper,                             # Função a ser executada a cada iteração do loop
            API.range(API.shape(data)[-1]-n+1), # Índices utilizados no loop
            fn_output_signature=data.dtype      # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        )
    )

    return data_sliced