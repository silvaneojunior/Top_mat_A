from math import gamma
from aux_mapping import null_mapping
from aux_base import dtype, ε_default, const, API_Numpy

class equation:
    def __init__(self,API, WENO, network,mapping=null_mapping, map_function=lambda x:x,p=2,ε=ε_default,γ=None):
        self.API=API
        self.WENO=WENO
        self.network=network
        self.p=const(p, API)
        self.ε=ε
        self.γ=γ

        self.mapping=mapping
        self.map_function=map_function
    
    def Get_weights_graph(self, u, Δx, d, AdicionaGhostPoints=None, t=None):

        if AdicionaGhostPoints is not None:
            u = AdicionaGhostPoints(u,self.API,t=t,n=2,Δx=Δx)
            u = slicer(u,5,self.API)
        if self.network is not None:
            δ = self.network(self.API.concat([
                u[...,0:1,0]  , 
                u[...,0:1,1]  , 
                u[...,2]      , 
                u[...,-1:,-2] , 
                u[...,-1:,-1]
            ], axis=-1))
            δ = slicer(δ,3,self.API)
        else:
            δ=self.API.real((u-u)[...,1:-1]+1-0.1)
        
        # Calcula os indicadores de suavidade locais
        β0 = self.API.square(self.API.abs( const(1, self.API)/2.0*u[...,0] - 2*u[...,1] + const(3, self.API)/2.0*u[...,2])) + const(13, self.API)/12.0*self.API.square(self.API.abs(u[...,0] - 2*u[...,1] + u[...,2]))
        β1 = self.API.square(self.API.abs(-const(1, self.API)/2.0*u[...,1]              + const(1, self.API)/2.0*u[...,3])) + const(13, self.API)/12.0*self.API.square(self.API.abs(u[...,1] - 2*u[...,2] + u[...,3]))
        β2 = self.API.square(self.API.abs(-const(3, self.API)/2.0*u[...,2] + 2*u[...,3] - const(1, self.API)/2.0*u[...,4])) + const(13, self.API)/12.0*self.API.square(self.API.abs(u[...,2] - 2*u[...,3] + u[...,4]))
        
        β = self.API.stack([β0, β1, β2], axis=-1)
        
        α, λ = self.WENO(β, δ, d, self.API, Δx=Δx, mapping=self.mapping, map_function=self.map_function, p=self.p, ε=self.ε)

        soma = self.API.sum(α, axis=-1, keepdims=True)
        ω    = α / soma
        
        return ω, α, β, δ, λ

    def ReconstructionMinus(self, u, Δx, d, C):
        
        ω = self.Get_weights_graph(u, Δx, d)[0]
        # Calcula os fhat em cada subestêncil
        fhat = self.API.matmul(u, C)
        # Calcula o fhat do estêncil todo
        fhat = self.API.sum(ω*fhat, axis=-1)
        
        return fhat
    
    def ReconstructionPlus(self, u, Δx, d, C):
        fhat = self.ReconstructionMinus(self.API.reverse(u, axis=[-1]), Δx, d, C)
        return fhat
    
    def flux_sep(self, U):
        pass

    def DerivadaEspacial(self, U, Δx, AdicionaGhostPoints, d, C, n_sep=6, t=None, n_ghostpoints=3):
        
        U = AdicionaGhostPoints(U, self.API, n=n_ghostpoints, t=t, Δx=Δx) # Estende a malha de pontos de acordo com as condições de fronteira
        
        f_plus, f_minus = self.flux_sep(U, n=n_sep)

        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = self.ReconstructionMinus(f_plus[...,:-1], Δx, d, C)
        f_half_plus  = self.ReconstructionPlus( f_minus[...,1:], Δx, d, C)
        
        Fhat = (f_half_minus + f_half_plus)

        return Fhat

# Classe equation para esquemas com RK explícito
#---------------------------------------------------------------------------------------------------------------------------------------
class equation_RK(equation):
    
    def Get_modifiers(self, u, Δx):
        
        if self.network is not None:
            δ = self.network(self.API.concat([
                u[...,0:1,0]  , 
                u[...,0:1,1]  , 
                u[...,2]      , 
                u[...,-1:,-2] , 
                u[...,-1:,-1]
            ], axis=-1))
            δ = slicer(δ,3,self.API)
        else:
            δ = self.API.real((u-u)[...,1:-1]+1-0.1)
        
        # Calcula os indicadores de suavidade locais
        β0 = self.API.square(self.API.abs( const(1, self.API)/2.0*u[...,0] - 2*u[...,1] + const(3, self.API)/2.0*u[...,2])) + const(13, self.API)/12.0*self.API.square(self.API.abs(u[...,0] - 2*u[...,1] + u[...,2]))
        β1 = self.API.square(self.API.abs(-const(1, self.API)/2.0*u[...,1]              + const(1, self.API)/2.0*u[...,3])) + const(13, self.API)/12.0*self.API.square(self.API.abs(u[...,1] - 2*u[...,2] + u[...,3]))
        β2 = self.API.square(self.API.abs(-const(3, self.API)/2.0*u[...,2] + 2*u[...,3] - const(1, self.API)/2.0*u[...,4])) + const(13, self.API)/12.0*self.API.square(self.API.abs(u[...,2] - 2*u[...,3] + u[...,4]))
        
        β = self.API.stack([β0, β1, β2], axis=-1)
        λ = self.WENO(β, δ, self.API, Δx=Δx, p=self.p, ε=self.ε)
        α = self.mapping(λ, self.API, self.map_function)
        
        return α, β, δ, λ
        
    def Get_weights_graph(self, α, d):
        
        α    = self.API.matmul(α, d)
        soma = self.API.sum(α, axis=-1, keepdims=True)
        ω    = α / soma
        
        return ω, α
    
    def ReconstructionMinus(self, u, α, d, C):
        
        ω = self.Get_weights_graph(α, d)[0]
        # Calcula os fhat em cada subestêncil
        fhat = self.API.matmul(u, C)
        # Calcula o fhat do estêncil todo
        fhat = self.API.sum(ω*fhat, axis=-1)
        
        return fhat
    
    def DerivadaEspacial(self, f_plus, f_minus, α_plus, α_minus, d, C):
        
        # Aplicar WENO em cada variável característica separadamente para depois juntar
        f_half_minus = self.ReconstructionMinus(f_plus , α_plus , d, C)
        f_half_plus  = self.ReconstructionMinus(f_minus, α_minus, d, C)
        
        fhat = (f_half_minus + f_half_plus)

        return fhat
    
    def Pre_Treatment(self, u, API, fronteira, n_ghostpoints=4, n=6, t=None, Δx=None):
        
        U = fronteira(u, API, n=n_ghostpoints, t=t, Δx=Δx) # Estende a malha de pontos de acordo com as condições de fronteira
        f_plus, f_minus = self.flux_sep(U, n=6)
        f_minus = self.API.reverse(f_minus, [-1])
        
        α_plus  = self.Get_modifiers(f_plus , Δx)[0]
        α_minus = self.Get_modifiers(f_minus, Δx)[0]
        
        aux = API.zeros(f_plus.shape)
        aux = API.expand_dims(aux, axis = -1)
        
        return [f_plus, f_plus], [f_minus, f_minus], [α_plus, α_plus], [α_minus, α_minus], [aux, aux]
    
    def Post_Treatment(self, f_half, R):
        return f_half
    
class transp_equation_RK(equation_RK):
    
    def maximum_speed(self,U):
        return self.API.cast(1, dtype)
    
    def flux_sep(self, U, n=6):
        U  = slicer(U, n, self.API)
        # Setup para equação do transporte
        M = self.maximum_speed(U)  # Valor utilizado para realizar a separação de fluxo
        f_plus  = (U + M*U)/2 # Fluxo positivo
        f_minus = (U - M*U)/2 # Fluxo negativo
        return f_plus, f_minus

class burgers_equation_RK(equation_RK):
    
    def maximum_speed(self, U):
        return self.API.max(self.API.abs(U), axis=-1, keepdims=True)
    
    def flux_sep(self, U, n=6):
        M = self.maximum_speed(U)                          # Valor utilizado para realizar a separação de fluxo
        M = self.API.expand_dims(M, -1)
        U = slicer(U, n, self.API)
        # Setup para equação do transporte
        f_plus  = (U**2/2 + M*U) / 2                      # Fluxo positivo
        f_minus = (U**2/2 - M*U) / 2                      # Fluxo negativo
        return f_plus, f_minus

class diff_equation_RK(equation_RK):
    
    def maximum_speed(self, U):
        return 1
    
    def flux_sep(self, U, n=6):
        U = slicer(U, n, self.API)
        f_plus  = U / 2                      # Fluxo positivo
        f_minus = U / 2                      # Fluxo negativo
        return f_plus, f_minus
    
class euler_equation_RK(equation_RK):
    
    def __init__(self, API, WENO, network,mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default, γ=const(14, API_Numpy)/10):
        super(euler_equation_RK, self).__init__(API, WENO, network, mapping=mapping, map_function=map_function, p=p, ε=ε, γ=γ)
        self.γ = γ

    def maximum_speed(self, U):
        eig_val = self.API.abs(self.Eigenvalues(U))
        return self.API.max(eig_val, axis=(-1,-2), keepdims=True)
    
    def Eigenvalues(self,Q):
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        U = Q1/Q0
        P = self.Pressure(Q)
        A = self.API.sqrt(self.γ*P/Q0)

        return self.API.stack([U-A, U, U+A], axis=-2)
    
    def Pressure(self, Q):
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        a = self.γ-1.0
        b = Q1**2
        c = 2*Q0
        d = b/c
        e = Q2-d
        
        return a*e
    
    def Eigensystem(self, Q, Average):
        
        U, A, H, h = Average(Q)
        
        ones_ref = self.API.ones( self.API.shape(U), dtype=U.dtype)
        zero_ref = self.API.zeros(self.API.shape(U), dtype=U.dtype)
        
        R1 = self.API.stack([ones_ref, U-A, H - U*A], axis=-1)
        R2 = self.API.stack([ones_ref, U  , U**2/2 ], axis=-1)
        R3 = self.API.stack([ones_ref, U+A, H + U*A], axis=-1)
        
        L1 = self.API.stack([U/(2*A) + U**2*h/2, 2 - (2*H)*h, U**2*h/2 - U/(2*A)], axis=-1)
        L2 = self.API.stack([-U*h - 1/(2*A)    , (2*U)*h    , 1/(2*A) - U*h     ], axis=-1)
        L3 = self.API.stack([h                 , -2*h       , h                 ], axis=-1)
        
        Λ1 = self.API.stack([     U-A, zero_ref, zero_ref], axis=-1)
        Λ2 = self.API.stack([zero_ref,        U, zero_ref], axis=-1)
        Λ3 = self.API.stack([zero_ref, zero_ref,      U+A], axis=-1)
        
        R = self.API.stack([R1,R2,R3], axis=-2)
        L = self.API.stack([L1,L2,L3], axis=-2)
        Λ = self.API.stack([Λ1,Λ2,Λ3], axis=-2)
        
        return R, L, Λ
    
    def ArithmeticAverage(self, Q):
        r  = 2
        Qa = (Q[...,r] + Q[...,r+1])/2
        
        Q0, Q1, Q2 = self.API.unstack(Qa, axis=-1)
        
        U = Q1/Q0
        P = self.Pressure(self.API.expand_dims(Qa, axis=-1))
        P = self.API.squeeze(P, axis=-1)
        A = self.API.sqrt(self.γ*P/Q0) # Sound speed
        H = (Q2 + P)/Q0                # Enthalpy
        h = 1/(2*H - U**2)
        
        return U, A, H, h
    
    def NoAverage(self, Q):
        r  = 2
        Qa = Q[...,r]
        
        Q0, Q1, Q2 = self.API.unstack(Qa, axis=-1)
        
        U = Q1/Q0
        P = self.Pressure(self.API.expand_dims(Qa, axis=-1))
        P = self.API.squeeze(P, axis=-1)
        A = self.API.sqrt(self.γ*P/Q0) # Sound speed
        H = (Q2 + P)/Q0                # Enthalpy
        h = 1/(2*H - U**2)
        
        return U, A, H, h
    
    def Pre_Treatment(self, u, API, fronteira, n_ghostpoints=4, n=6, t=None, Δx=None):
        
        Q  = fronteira(u, API, n=n_ghostpoints, t=t, Δx=Δx)
        Qi = slicer(Q, n, self.API)                  # (  3, n+3, 6)
        Qi = self.API.einsum('...ijk -> ...jik', Qi) # (n+3,   3, 6)
        
        f_plus, f_minus, α_plus, α_minus, R = [None, None], [None, None], [None, None], [None, None], [None, None]
        
        R[0], L, Λ = self.Eigensystem(Qi, self.ArithmeticAverage) # (n+3, 3, 3), (n+3, 3, 3), (n+3, 3, 3)
        
        M = self.API.max(self.API.abs(Λ), axis=(-1,-2), keepdims=True) # (n+3, 1, 1) 
        W = self.API.einsum('...ki, ...kj -> ...ji', Qi, L)            # (n+3, 3, 6) Transforms into characteristic variables
        G = Λ @ W                                                      # (n+3, 3, 6) The flux for the characteristic variables is Λ*L*Q
        
        f_plus[0]  = (G + M*W)/2 # (n+3, 3, 6)
        f_minus[0] = (G - M*W)/2 # (n+3, 3, 6)
        
        f_plus[0]  = self.API.einsum('...ijk->...jik', f_plus[0])  # (3, n+3, 6)
        f_minus[0] = self.API.einsum('...ijk->...jik', f_minus[0]) # (3, n+3, 6)
        f_minus[0] = self.API.reverse(f_minus[0], [-1])
        
        α_plus[0]  = self.Get_modifiers(f_plus[0] , Δx)[0]
        α_minus[0] = self.Get_modifiers(f_minus[0], Δx)[0]
        
        R[1], L, Λ = self.Eigensystem(Qi, self.NoAverage)         # (n+3, 3, 3), (n+3, 3, 3), (n+3, 3, 3)
        
        M = self.API.max(self.API.abs(Λ), axis=(-1,-2), keepdims=True) # (n+3, 1, 1) 
        W = self.API.einsum('...ki, ...kj -> ...ji', Qi, L)            # (n+3, 3, 6) Transforms into characteristic variables
        G = Λ @ W                                                      # (n+3, 3, 6) The flux for the characteristic variables is Λ*L*Q
        
        f_plus[1]  = (G + M*W)/2 # (n+3, 3, 6)
        f_minus[1] = (G - M*W)/2 # (n+3, 3, 6)
    
        f_plus[1]  = self.API.einsum('...ijk->...jik', f_plus[1])  # (3, n+3, 6)
        f_minus[1] = self.API.einsum('...ijk->...jik', f_minus[1]) # (3, n+3, 6)
        f_minus[1] = self.API.reverse(f_minus[1], [-1])

#-------------------------------------------------------------------
        # Versão mais barata, mas menos correta
#         α_plus[1]  = α_plus[0]
#         α_minus[1] = α_minus[0]
#-------------------------------------------------------------------
        
#-------------------------------------------------------------------
        # Versão mais cara, mas mais correta
        α_plus[1]  = self.Get_modifiers(f_plus[1] , Δx)[0]
        α_minus[1] = self.Get_modifiers(f_minus[1], Δx)[0]
#-------------------------------------------------------------------
        
        return f_plus, f_minus, α_plus, α_minus, R
    
    def Post_Treatment(self, f_half, R):
        f_half = self.API.einsum('...kn, ...nki -> ...in', f_half, R) # (3, n+3) Brings back to conservative variables
        return f_half
    
#---------------------------------------------------------------------------------------------------------------------------------------
    
class transp_equation(equation):
    
    def maximum_speed(self,U):
        return self.API.cast(1, dtype)
    
    def flux_sep(self, U, n=6):
        U  = slicer(U, n, self.API)
        # Setup para equação do transporte
        M = self.maximum_speed(U)                          # Valor utilizado para realizar a separação de fluxo
        f_plus  = (U + M*U)/2 # Fluxo positivo
        f_minus = (U - M*U)/2 # Fluxo negativo
        return f_plus, f_minus
    
class burgers_equation(equation):
    
    def maximum_speed(self, U):
        return self.API.max(self.API.abs(U), axis=-1, keepdims=True)
    
    def flux_sep(self, U, n=6):
        M = self.maximum_speed(U)                          # Valor utilizado para realizar a separação de fluxo
        M = self.API.expand_dims(M, -1)
        U = slicer(U, n, self.API)
        # Setup para equação do transporte
        f_plus  = (U**2/2 + M*U) / 2                      # Fluxo positivo
        f_minus = (U**2/2 - M*U) / 2                      # Fluxo negativo
        return f_plus, f_minus

class diff_equation(equation):
    
    def maximum_speed(self, U):
        return 1
    
    def flux_sep(self, U, n=6):
        U = slicer(U, n, self.API)
        f_plus  = U / 2                      # Fluxo positivo
        f_minus = U / 2                      # Fluxo negativo
        return f_plus, f_minus

class euler_equation(equation):
    
    def __init__(self, API, WENO, network,mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default, γ=const(14, API_Numpy)/10):
        super(euler_equation,self).__init__(API, WENO, network, mapping=mapping, map_function=map_function, p=p, ε=ε, γ=γ)
        self.γ = γ
        
    def Pressure(self,Q):
        
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        a = self.γ-1.0
        b = Q1**2
        c = 2*Q0
        d = b/c
        e = Q2-d
        
        return a*e

    def maximum_speed(self, U):
        eig_val = self.API.abs(self.Eigenvalues(U))
        return self.API.max(eig_val, axis=(-1,-2), keepdims=True)
    
    def Eigensystem(self, Q):
        
        U, A, H, h = self.ArithmeticAverage(Q)
        
        ones_ref = self.API.ones( self.API.shape(U), dtype=U.dtype)
        zero_ref = self.API.zeros(self.API.shape(U), dtype=U.dtype)
        
        R1 = self.API.stack([ones_ref, U-A, H - U*A], axis=-1)
        R2 = self.API.stack([ones_ref, U  , U**2/2 ], axis=-1)
        R3 = self.API.stack([ones_ref, U+A, H + U*A], axis=-1)
        
        L1 = self.API.stack([U/(2*A) + U**2*h/2, 2 - (2*H)*h, U**2*h/2 - U/(2*A)], axis=-1)
        L2 = self.API.stack([-U*h - 1/(2*A)    , (2*U)*h    , 1/(2*A) - U*h     ], axis=-1)
        L3 = self.API.stack([h                 , -2*h       , h                 ], axis=-1)
        
        Λ1 = self.API.stack([     U-A, zero_ref, zero_ref], axis=-1)
        Λ2 = self.API.stack([zero_ref,        U, zero_ref], axis=-1)
        Λ3 = self.API.stack([zero_ref, zero_ref,      U+A], axis=-1)
        
        R = self.API.stack([R1,R2,R3], axis=-2)
        L = self.API.stack([L1,L2,L3], axis=-2)
        Λ = self.API.stack([Λ1,Λ2,Λ3], axis=-2)
        
        return R, L, Λ

    def Eigenvalues(self, Q):
        
        Q0, Q1, Q2 = self.API.unstack(Q, axis=-2)
        
        U = Q1/Q0
        P = self.Pressure(Q)
        A = self.API.sqrt(self.γ*P/Q0)

        return self.API.stack([U-A, U, U+A], axis=-2)

    def ArithmeticAverage(self, Q):
        r  = 2
        Qa = (Q[...,r] + Q[...,r+1])/2
        
        Q0, Q1, Q2 = self.API.unstack(Qa, axis=-1)
        
        U = Q1/Q0
        P = self.Pressure(self.API.expand_dims(Qa, axis=-1))
        P = self.API.squeeze(P, axis=-1)
        A = self.API.sqrt(self.γ*P/Q0) # Sound speed
        H = (Q2 + P)/Q0                # Enthalpy
        h = 1/(2*H - U**2)
        
        return U, A, H, h

    def ReconstructedFlux(self, F, Q, M, Δx, d, C, n_sep):
        
        F_plus  = (F + M*Q)/2 # (n, 3, 6)
        F_minus = (F - M*Q)/2 # (n, 3, 6)

        F_plus  = self.API.einsum('...ijk -> ...jik', F_plus)  # (3, n, 6)
        F_minus = self.API.einsum('...ijk -> ...jik', F_minus) # (3, n, 6)
        
        F_half_plus  = self.ReconstructionMinus(F_plus[...,:-1], Δx, d, C) 
        F_half_minus = self.ReconstructionPlus( F_minus[...,1:], Δx, d, C)

        return F_half_plus + F_half_minus

    def DerivadaEspacial(self, Q, Δx, AdicionaGhostPoints, d, C, n_sep=6, t=None, n_ghostpoints=3):
        
        Q  = AdicionaGhostPoints(Q, self.API, n=n_ghostpoints, t=t, Δx=Δx)
        Qi = slicer(Q, n_sep, self.API)              # (3, n, 6)
        Qi = self.API.einsum('...ijk -> ...jik', Qi) # (n, 3, 6)
        
        R, L, Λ = self.Eigensystem(Qi) # (n, 3, 3), (n, 3, 3), (n, 3, 3)
        
        M = self.API.max(self.API.abs(Λ), axis=(-1,-2), keepdims=True) # (n, 1, 1) 
        W = self.API.einsum('...ki, ...kj -> ...ji', Qi, L)            # (n, 3, 6) Transforms into characteristic variables
        G = self.API.matmul(Λ, W)                                      # (n, 3, 6) The flux for the characteristic variables is Λ*L*Q
        
        G_half = self.ReconstructedFlux(G, W, M, Δx, d, C, n_sep)     # (3, n)
        F_half = self.API.einsum('...kn, ...nki -> ...in', G_half, R) # (3, n) Brings back to conservative variables
        
        return F_half

class euler_equation_2D(equation):
    
    def __init__(self,API, WENO, network,γ,mapping=null_mapping, map_function=lambda x:x,p=2,ε=ε_default):
        super(euler_equation_2D,self).__init__(API, WENO, network, γ=γ, mapping=mapping, map_function=map_function,p=p,ε=ε)
        self.γ=API.cast(γ,dtype=dtype)
        
    def Pressure_1D(self, Q):
        Q1, Q2, Q3, Q4 = self.API.unstack(Q, axis=-2)

        a = self.γ-1.0
        b = Q2**2 + Q3**2
        c = 2*Q1
        d = b/c
        e = Q4-d
        
        return a*e

    def Pressure_2D(self, Q):
        Q1, Q2, Q3, Q4 = self.API.unstack(Q, axis=-3)

        a = self.γ-1.0
        b = Q2**2 + Q3**2
        c = 2*Q1
        d = b/c
        e = Q4-d
        
        return a*e

    def ConservativeVariables(self, Q):
        ρ, u, v, p = self.API.unstack(Q, axis=-3)

        ρu = ρ*u
        ρv = ρ*v
        E  = p/(self.γ-1) + ρ*(u**2+v**2)/2
        return self.API.concat([ρ, ρu, ρv, E], axis=3)

    def PrimitiveVariables_1D(self, Q):
        R = Q[...,0,:]
        U = Q[...,1,:]/R
        V = Q[...,2,:]/R
        P = self.Pressure_1D(Q)
        return R, U, V, P

    def PrimitiveVariables_2D(self, Q):
        R = Q[...,0,:,:]
        U = Q[...,1,:,:]/R
        V = Q[...,2,:,:]/R
        P = self.Pressure_2D(Q)
        return R, U, V, P

    def ArithmeticAverage(self, Q):
        r  = 2
        Qa = (Q[...,r] + Q[...,r+1])/2
        Qa = self.API.einsum('...xyv->...vxy', Qa)

        R, U, V, P = self.PrimitiveVariables_2D(Qa)

        A = self.API.sqrt(self.γ*P/R)     # Sound speed
        H = (Qa[...,3,:,:] + P)/R    # Enthalpy
        h = 1/(2*H - U**2 - V**2)

        return U, V, A, H, h

    def EigensystemX(self,Q): 
        
        U, V, A, H, h   = self.ArithmeticAverage(Q)
        I2A, Hh, Uh, Vh = 0.5/A, H*h, U*h, V*h
        
        ones_ref = self.API.ones( self.API.shape(U), dtype=U.dtype)
        zero_ref = self.API.zeros(self.API.shape(U), dtype=U.dtype)
        
        R1c = self.API.stack([ones_ref  , U-A       , V       , H - U*A      ], axis=-1)
        R2c = self.API.stack([ones_ref  , U         , V       , (U**2+V**2)/2], axis=-1)
        R3c = self.API.stack([ones_ref*0, ones_ref*0, ones_ref, V            ], axis=-1)
        R4c = self.API.stack([ones_ref  , U+A       , V       , H + U*A      ], axis=-1)
        
        L1c = self.API.stack([U*I2A+Hh-0.5, 2.0-2.0*Hh, -V        , Hh-0.5-U*I2A], axis=-1)
        L2c = self.API.stack([-Uh-I2A     , 2.0*Uh    , ones_ref*0, I2A-Uh      ], axis=-1)
        L3c = self.API.stack([-Vh         , 2.0*Vh    , ones_ref  , -Vh         ], axis=-1)
        L4c = self.API.stack([h           , -2*h      , ones_ref*0, h           ], axis=-1)
        
        Λ1 = self.API.stack([     U-A, zero_ref, zero_ref, zero_ref], axis=-1)
        Λ2 = self.API.stack([zero_ref, U       , zero_ref, zero_ref], axis=-1)
        Λ3 = self.API.stack([zero_ref, zero_ref, U       , zero_ref], axis=-1)
        Λ4 = self.API.stack([zero_ref, zero_ref, zero_ref, U+A     ], axis=-1)

        R = self.API.stack([R1c, R2c, R3c, R4c], axis=-2)
        L = self.API.stack([L1c, L2c, L3c, L4c], axis=-2)
        Λ = self.API.stack([Λ1 , Λ2 , Λ3 , Λ4 ], axis=-2)

        return R, L, Λ

    def EigensystemY(self,Q):
        
        U, V, A, H, h   = self.ArithmeticAverage(Q)
        I2A, Hh, Uh, Vh = 0.5/A, H*h, U*h, V*h
        
        ones_ref = self.API.ones( self.API.shape(U), dtype=U.dtype)
        zero_ref = self.API.zeros(self.API.shape(U), dtype=U.dtype)
        
        R1c = self.API.stack([ones_ref  , U       , V-A       , H - V*A      ], axis=-1)
        R2c = self.API.stack([ones_ref  , U       , V         , (U**2+V**2)/2], axis=-1)
        R3c = self.API.stack([ones_ref*0, ones_ref, ones_ref*0, U            ], axis=-1)
        R4c = self.API.stack([ones_ref  , U       , V+A       , H + V*A      ], axis=-1)
        
        L1c = self.API.stack([V*I2A+Hh-0.5, 2.0-2.0*Hh, -U        , Hh-0.5-V*I2A], axis=-1)
        L2c = self.API.stack([-Uh         , 2.0*Uh    , ones_ref  , -Uh         ], axis=-1)
        L3c = self.API.stack([-Vh-I2A     , 2.0*Vh    , ones_ref*0, I2A-Vh      ], axis=-1)
        L4c = self.API.stack([h           ,-2*h       , ones_ref*0, h           ], axis=-1)
        
        Λ1 = self.API.stack([V-A     , zero_ref, zero_ref, zero_ref], axis=-1)
        Λ2 = self.API.stack([zero_ref, V       , zero_ref, zero_ref], axis=-1)
        Λ3 = self.API.stack([zero_ref, zero_ref, V       , zero_ref], axis=-1)
        Λ4 = self.API.stack([zero_ref, zero_ref, zero_ref, V+A     ], axis=-1)
        
        R = self.API.stack([R1c, R2c, R3c, R4c], axis=-2)
        L = self.API.stack([L1c, L2c, L3c, L4c], axis=-2)
        Λ = self.API.stack([Λ1 , Λ2 , Λ3 , Λ4 ], axis=-2)

        return R, L, Λ

    def EigenvaluesX(self,Q):
        R, U, V, P = self.PrimitiveVariables_1D(Q)
        A = self.API.sqrt(self.γ*P/R)

        return self.API.stack([U-A, U, U, U+A], axis=-2)

    def EigenvaluesY(self,Q):
        R, U, V, P = self.PrimitiveVariables_1D(Q)
        A = self.API.sqrt(self.γ*P/R)

        return self.API.stack([V-A, V, V, V+A], axis=-2)

    def maximum_speed(self,Q):
        R, U, V, P = self.PrimitiveVariables_2D(Q)
        A = self.API.sqrt(self.γ*P/R)

        max_U=self.API.stack([(U-A)**2, (U+A)**2],-1)
        max_V=self.API.stack([(V-A)**2, (V+A)**2],-1)

        MU = self.API.max(max_U,axis=(-1,-2,-3),keepdims=True)
        MV = self.API.max(max_V,axis=(-1,-2,-3),keepdims=True)
        return self.API.sqrt(MU+MV)

    def ReconstructedFlux(self, F, Q, M, Δx, d, C):
        
        F_plus  = (F + M*Q)/2 # (x, y, 4, 6)
        F_minus = (F - M*Q)/2 # (x, y, 4, 6)

        F_plus  = self.API.einsum('...ijkl -> ...kijl', F_plus)  # (4, x, y, 6)
        F_minus = self.API.einsum('...ijkl -> ...kijl', F_minus) # (4, x, y, 6)
        
        F_half_plus  = self.ReconstructionMinus(F_plus[...,:-1], Δx, d, C) # (4, x, y)
        F_half_minus = self.ReconstructionPlus( F_minus[...,1:], Δx, d, C) # (4, x, y)

        return F_half_plus + F_half_minus

    def DerivadaEspacialX(self, Q, Δx, d, C, AdicionaGhostPoints, t=None, n_ghostpoints=3):
        
        Q  = AdicionaGhostPoints(Q, self.API, n=n_ghostpoints, t=t) # (  4, x+6, y)
        Qi = slicer_X(Q, 6, self.API)                               # (  4, x+1, y, 6)
        Qi = self.API.einsum('...ijkl -> ...jkil', Qi)              # (x+1,   y, 4, 6)
        
        R, L, Λ = self.EigensystemX(Qi) # (x+1, y, 4, 4), (x+1, y, 4, 4), (x+1, y, 4, 4)
        
        M = self.API.max(self.API.abs(Λ), axis=(-1,-2), keepdims=True) # (x+1, y, 1, 1)
        W = self.API.einsum('...ki, ...kj -> ...ji', Qi, L)            # (x+1, y, 4, 6) Transforms into characteristic variables
        G = self.API.matmul(Λ, W)                                      # (x+1, y, 4, 6) The flux for the characteristic variables is Λ * L*Q
                
        G_half = self.ReconstructedFlux(G, W, M, Δx, d, C)               # (4, x+1, y)
        F_half = self.API.einsum('...ixy, ...xyij -> ...jxy', G_half, R) # (4, x+1, y)
        
        return F_half

    def DerivadaEspacialY(self,Q, Δy, d, C, AdicionaGhostPoints, t=None, n_ghostpoints=3):
        
        Q  = AdicionaGhostPoints(Q, self.API, n=n_ghostpoints, t=t) # (4,   x, y+6)
        Qi = slicer_Y(Q, 6, self.API)                               # (4,   x, y+1, 6)
        Qi = self.API.einsum('...ijkl -> ...jkil', Qi)              # (x, y+1,   4, 6)
        
        R, L, Λ = self.EigensystemY(Qi) # (x, y+1, 4, 4), (x, y+1, 4, 4), (x, y+1, 4, 4)
        
        M = self.API.max(self.API.abs(Λ), axis=(-1,-2), keepdims=True) # (x, y+1, 1, 1)
        W = self.API.einsum('...ki, ...kj -> ...ji', Qi, L)            # (x, y+1, 4, 6) Transforms into characteristic variables
        G = self.API.matmul(Λ, W)                                      # (x, y+1, 4, 6) The flux for the characteristic variables is Λ * L*Q
        
        G_half = self.ReconstructedFlux(G, W, M, Δy, d, C)               # (4, x, y+1)
        F_half = self.API.einsum('...ixy, ...xyij -> ...jxy', G_half, R) # (4, x, y+1) Brings back to conservative variables
        
        return F_half

def slicer(data, n, API):
    helper = lambda i: data[...,i:i+n]

    data_sliced = API.einsum(
    'i...j -> ...ij',
        API.map_fn(
            helper,                    # Função a ser executada a cada iteração do loop
            API.range(API.shape(data)[-1]-n+1),     # Índices utilizados no loop
            fn_output_signature=data.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        )
    )

    return data_sliced

def slicer_X(data, n, API):
    helper = lambda i: data[...,i:i+n,:]

    data_sliced = API.einsum(
    'i...lj -> ...ijl',
        API.map_fn(
            helper,                    # Função a ser executada a cada iteração do loop
            API.range(API.shape(data)[-2]-n+1),     # Índices utilizados no loop
            fn_output_signature=data.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        )
    )
    return data_sliced

def slicer_Y(data, n, API):
    helper = lambda i: data[...,i:i+n]

    data_sliced = API.einsum(
    'j...il -> ...ijl',
        API.map_fn(
            helper,                    # Função a ser executada a cada iteração do loop
            API.range(API.shape(data)[-1]-n+1),     # Índices utilizados no loop
            fn_output_signature=data.dtype # Tipo da variável de retorno (epecificado pois o tipo de entrado difere do tipo de saída)
        )
    )
    return data_sliced