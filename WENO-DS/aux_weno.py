from aux_base import *
from aux_base import d1 as d_default
from aux_base import C1 as C_default
from aux_mapping import *


def WENO_linear_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    λ = API.constant([1, 1, 1], dtype=dtype)
    α = API.matmul(λ, d)
    
    return α, λ

def WENO_JS_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):

    β = β*(δ+const(1, API)/10)    
    # Calcula os pesos do WENO-JS
    λ = (1/(β + ε))**p
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_Z_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z
    λ = 1 + (τ/(β + ε))**p
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_ZD_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    # Ajustando inconsistência de escala dos betas
    β = API.stack([β[...,0], 0.5*β[...,1], β[...,2]], axis=-1)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z
    λ = 1 + (τ/(β + ε))**p
    α = mapping(λ, API, map_function, d)
    
    return α, λ

# def WENO_ZD2_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
#     # Ajustando inconsistência de escala dos betas
#     β    = API.stack([β[...,0], β[...,1], β[...,2]], axis=-1)
#     # Calcula o indicador de suavidade global
#     τ = API.abs(β[...,0:1] - β[...,2:3])
#     # Calcula os pesos do WENO-Z
#     λ = 1 + (τ/(β + ε))**p
#     α = mapping(λ, API, map_function, d)
    
#     return α, λ

# def WENO_ZD3_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
#     # Ajustando inconsistência de escala dos betas
#     β = API.stack([β[...,0], β[...,1], β[...,2]], axis=-1)
#     # Calcula o indicador de suavidade global
#     τ = API.abs(β[...,0:1] - β[...,2:3])
#     # Calcula os pesos do WENO-Z
#     λ = 1 + (τ/(β + ε))**p
#     α = mapping(λ, API, map_function, d)
    
#     return α, λ

# def WENO_ZD4_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):

#     # Ajustando inconsistência de escala dos betas
#     β    = API.stack([β[...,0], β[...,1], β[...,2]], axis=-1)
#     # Calcula o indicador de suavidade global
#     τ = API.abs(β[...,0:1] - β[...,2:3])
#     # Calcula os pesos do WENO-Z
#     λ = 1 + (τ/(β + ε))**p
#     α = mapping(λ, API, map_function, d)
    
#     return α, λ

def WENO_ZDp_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    # Normalizing the betas
    soma = API.sum(β, axis=-1, keepdims=True)
    β    = β / (soma + ε)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    ξ = (β/(τ + 0.3))
    # β1 original: c = 0.25, β1 alterado: c = 0.4 com desagravamento de 0.975
    β = API.stack([β[...,0], API.minimum(1, 0.5 + 0.1*API.sum(ξ, axis=-1))*β[...,1], β[...,2]], axis=-1)
    λ = 1 + (τ/(β + ε))**p + ξ
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_Zp_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = 1 + γ**p + (Δx**(const(2, API)/3))/γ
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_Zp_net_expo_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = 1 + γ**p+(Δx**δ)/γ
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_ZC_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = (1 + γ**p)
    α = mapping(λ, API, map_function, d)
    α = α + API.matmul((Δx**(const(2, API)/3))/γ, d)
    
    return α, λ

# def WENO_ZD_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
#     β = β*(δ+const(1, API)/10)
#     # Calcula o indicador de suavidade global
#     τ = API.abs(β[...,0:1] - β[...,2:3])
#     # Calcula os pesos do WENO-Z+
#     λ    = 1 + ((τ + ε)/(β + ε))**p
#     soma = API.sum(λ, axis=-1, keepdims=True)
#     α    = map_function(λ/soma)
#     α    = α + (Δx**(const(2, API)/3))/(soma*α+ε)**(1/p)
#     α    = API.matmul(α, d)
    
#     return α, λ

def WENO_D_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    ϕ = API.sqrt(API.abs(β[...,0:1] - 2*β[...,1:2] + β[...,2:3]))
    Φ = API.minimum(1,ϕ)
    # Calcula os pesos do WENO-Z
    λ = 1 + Φ*(τ/(β + ε))**p
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_A_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    ϕ = API.sqrt(API.abs(β[...,0:1] - 2*β[...,1:2] + β[...,2:3]))
    Φ = API.minimum(1,ϕ)
    # Calcula os pesos do WENO-Z
    λ = API.maximum(1, Φ*(τ/(β + ε))**p)
    α = mapping(λ, API, map_function, d)
    
    return α, λ

def WENO_ZC_net_expo_scheme(β, δ, d, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2, ε=ε_default):
    
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = (1 + γ**p)
    α = mapping(λ, API, map_function, d)
    α = α + API.matmul((Δx**(δ))/γ, d)
    
    return α, λ

# WENOs para esquemas com RK explícito
#------------------------------------------------------------------------
def WENO_linear_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):
    λ = β*0 + const(1, API)
    return λ

def WENO_JS_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):

    β = β*(δ+const(1, API)/10)    
    # Calcula os pesos do WENO-JS
    λ = (1/(β + ε))**p
    
    return λ

def WENO_Z_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):
    
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z
    λ = 1 + (τ/(β + ε))**p
    
    return λ

def WENO_Zp_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = 1 + γ**p + (Δx**(const(2, API)/3))/γ
    
    return λ

def WENO_Zp_net_expo_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):
    
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = 1 + γ**p+(Δx**δ)/γ
    
    return λ

def WENO_D_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    ϕ = API.sqrt(API.abs(β[...,0:1] - 2*β[...,1:2] + β[...,2:3]))
    Φ = API.minimum(1,ϕ)
    # Calcula os pesos do WENO-Z
    λ = 1 + Φ*(τ/(β + ε))**p
    
    return λ

def WENO_A_RK_scheme(β, δ, API, Δx, p=2, ε=ε_default):
    
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    ϕ = API.sqrt(API.abs(β[...,0:1] - 2*β[...,1:2] + β[...,2:3]))
    Φ = API.minimum(1,ϕ)
    # Calcula os pesos do WENO-Z
    λ = API.maximum(1, Φ*(τ/(β + ε))**p)
    
    return λ
#------------------------------------------------------------------------

class simulation:
    def __init__(self, API, equation_class, WENO, γ, mapping=null_mapping, map_function=lambda x:x, network=None, p=2, ε=ε_default):

        self.equation      = equation_class(API, WENO, network=network,mapping=mapping, map_function=map_function, p=p, ε=ε, γ=γ)
        self.API           = API
        self.network       = network
        self.WENO          = WENO
        self.n_ghostpoints = 3
        
        self.p = p
        self.ε = ε
        self.d = d_default
        self.C = C_default
    
        self.mapping      = mapping
        self.map_function = map_function

        self.Sim      = API.function(self.Sim_graph)
        self.Sim_step = API.function(self.Sim_step_graph)

        self.Get_weights_graph = self.equation.Get_weights_graph
        self.Get_weights       = API.function(self.Get_weights_graph)
        self.DerivadaWrapper   = lambda h, Δx: (h[...,1:] - h[...,:-1])/Δx # Derivative of Flux

        self.DerivadaEspacial_graph = lambda u, Δx, fronteira, t: self.DerivadaWrapper(self.equation.DerivadaEspacial(u, Δx, AdicionaGhostPoints=fronteira, d=self.d, C=self.C, n_sep=6, t=t, n_ghostpoints=self.n_ghostpoints), Δx)
        self.DerivadaEspacial       = API.function(self.DerivadaEspacial_graph)

    def Sim_graph(self, u, t_final, Δx, CFL, fronteira):
        t = 0.0*self.equation.maximum_speed(u) # Instante de tempo incial para a computação
        # self.API.pretty_print(self.API.squeeze(t), end='\r')
        
        while self.API.any(t < t_final):
            Λ  = self.equation.maximum_speed(u)

            Δt = Δx*CFL/self.API.abs(Λ)
            Δt = self.API.where(t + Δt > t_final, t_final - t, Δt)
            
            u = self.Sim_step_graph(u, Δt, Δx, fronteira, t=t+Δt)
            t = t + Δt # Avançando no tempo
            
            self.API.pretty_print(self.API.squeeze(t), end='\r')
        
        return u
    
    def Sim_step_graph(self, u, Δt, Δx, fronteira, t=None):
        u1 = u - Δt*self.DerivadaEspacial_graph(u, Δx, fronteira, t=t)
        u2 = (3*u + u1 - Δt*self.DerivadaEspacial_graph(u1, Δx, fronteira, t=t)) / 4.0
        u  = (u + 2*u2 - 2*Δt*self.DerivadaEspacial_graph(u2, Δx, fronteira, t=t)) / 3.0
        return u

class simulation_RK(simulation):
    def __init__(self, API, equation_class, WENO, γ, mapping=null_mapping, map_function=lambda x:x, network=None, p=2, ε=ε_default):
        super(simulation_RK, self).__init__(API=API, equation_class=equation_class, WENO=WENO, γ=γ, mapping=mapping, map_function=map_function, network=network, p=p, ε=ε)
        
        self.n_ghostpoints          = 4
        self.DerivadaEspacial_graph = self.equation.DerivadaEspacial
        self.DerivadaEspacial       = API.function(self.DerivadaEspacial_graph)

    def Sim_step_graph(self, u, Δt, Δx, fronteira, t=None):
        
        # Transformando para as variáveis características
        f_plus, f_minus, α_plus, α_minus, R = self.equation.Pre_Treatment(u, self.API, fronteira, self.n_ghostpoints, n=6, t=t, Δx=Δx)
        
        h1       = self.DerivadaEspacial(f_plus[0][...,1:-1,:-1], f_minus[0][...,1:-1,:-1], α_plus[0][...,1:-1,:], α_minus[0][...,1:-1,:], d1      , C1)
        h2_plus  = self.DerivadaEspacial(f_plus[1][...,1:,:-1]  , f_minus[1][...,:-1,:-1] , α_plus[1][...,1:,:]  , α_minus[1][...,:-1,:] , d2_plus , C2)
        h2_minus = self.DerivadaEspacial(f_plus[1][...,1:,:-1]  , f_minus[1][...,:-1,:-1] , α_plus[1][...,1:,:]  , α_minus[1][...,:-1,:] , d2_minus, C2)
        h3       = self.DerivadaEspacial(f_plus[0][...,:-1]     , f_minus[0][...,:-1]     , α_plus[0]            , α_minus[0]            , d3      , C3)
        h2       = sigma_plus*h2_plus - sigma_minus*h2_minus
        
        # Destransformando das variáveis características
        h1 = self.equation.Post_Treatment(h1, R[0][...,1:-1,:,:])
        h2 = self.equation.Post_Treatment(h2, R[1][...,1:,:,:])
        h3 = self.equation.Post_Treatment(h3, R[0][...,:,:,:])
        
        u1 = (h1[...,1:] -   h1[...,:-1])/Δx
        u2 = (h2[...,2:] - 2*h2[...,1:-1] +   h2[...,:-2])/(Δx**2)
        u3 = (h3[...,3:] - 3*h3[...,2:-1] + 3*h3[...,1:-2] - h3[...,:-3])/(Δx**3)
        
        u = u - Δt*u1 + (Δt**2)*u2/2 - (Δt**3)*u3/6
        
        return u

class simulation_2D(simulation):
    
    def __init__(self,API,equation_class,WENO,γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=2,ε=ε_default):
        super(simulation_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO,γ=γ, mapping=mapping, map_function=map_function,network=network,p=p,ε=ε)

        self.DerivadaWrapperX   = lambda F_half, Δx: (F_half[...,1:,:] - F_half[...,:-1,:])/Δx # Derivative of Flux

        self.DerivadaEspacialX_graph = lambda u, Δx, fronteira, t: self.DerivadaWrapperX(self.equation.DerivadaEspacialX(u, Δx, AdicionaGhostPoints=fronteira,d=self.d, C=self.C, t=t, n_ghostpoints=self.n_ghostpoints),Δx)
        self.DerivadaEspacialX       = API.function(self.DerivadaEspacialX_graph)

        self.DerivadaWrapperY   = lambda F_half, Δx: (F_half[...,1:] - F_half[...,:-1])/Δx # Derivative of Flux

        self.DerivadaEspacialY_graph = lambda u, Δx, fronteira, t: self.DerivadaWrapperY(self.equation.DerivadaEspacialY(u, Δx, AdicionaGhostPoints=fronteira,d=self.d, C=self.C, t=t, n_ghostpoints=self.n_ghostpoints),Δx)
        self.DerivadaEspacialY       = API.function(self.DerivadaEspacialY_graph)

    def Sim_graph(self,u, t_final, Δx, Δy, CFL, fronteiraX, fronteiraY,Force): 
        
        t = 0.0*self.equation.maximum_speed(u) # Instante de tempo incial para a computação
        self.API.pretty_print(self.API.squeeze(t),end='\r')
        Δ=min(Δx, Δy)
        
        while self.API.any(t < t_final):
            Λ  = self.equation.maximum_speed(u)

            Δt = Δ*CFL/Λ  
            Δt = self.API.where(t + Δt > t_final, t_final - t, Δt)
            u=self.Sim_step_graph(u, Δt, Δx, Δy,fronteiraX, fronteiraY,Force, t=t)
            t  = t + Δt # Avançando no tempo
            self.API.pretty_print(self.API.squeeze(t),'                        ',end='\r')
        return u
    
    def Sim_step_graph(self,u, Δt, Δx, Δy,fronteiraX, fronteiraY,Force, t=None):

        duX=self.DerivadaEspacialX(u, Δx, fronteiraX, t=t)
        duY=self.DerivadaEspacialY(u, Δy, fronteiraY, t=t)
        
        # self.API.pretty_print(self.API.shape(u))
        # self.API.pretty_print(self.API.shape(Δt))
        # self.API.pretty_print(self.API.shape(Force(u,self.API)))
        # self.API.pretty_print(self.API.shape(duX))
        # self.API.pretty_print(self.API.shape(duY))
        
        u1 = u - Δt*(duX+duY-Force(u,self.API))

        du1X=self.DerivadaEspacialX(u1, Δx, fronteiraX, t=t)
        du1Y=self.DerivadaEspacialY(u1, Δy, fronteiraY, t=t)

        u2 = (3*u + u1 - Δt*(du1X+du1Y-Force(u1,self.API))) / 4.0

        du2X=self.DerivadaEspacialX(u2, Δx, fronteiraX, t=t)
        du2Y=self.DerivadaEspacialY(u2, Δy, fronteiraY, t=t)

        u  = (u + 2*u2 - 2*Δt*(du2X+du2Y-Force(u2,self.API))) / 3.0
        return u

## 1D
# WENO Linear
class WENO_linear(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_linear,self).__init__(API=API,equation_class=equation_class,WENO=WENO_linear_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

# WENO JS

class WENO_JS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_JS_M(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=post_mapping, map_function=Henrick_mapping,network=None,p=p,ε=ε)

class WENO_JS_MS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_MS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_JS_BI(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_BI,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO Z

class WENO_Z(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_Z_M(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=post_mapping, map_function=Henrick_mapping,network=None,p=p,ε=ε)

class WENO_Z_MS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_MS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Z_BI(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_BI,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO Z+

class WENO_Zp(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_Zp_M(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=post_mapping, map_function=Henrick_mapping,network=None,p=p,ε=ε)

class WENO_Zp_MS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_MS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Zp_BI(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_BI,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO ZC

class WENO_ZC_M(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_ZC_scheme,γ=γ, mapping=post_inv_mapping, map_function=Henrick_mapping,network=None,p=p,ε=ε)

class WENO_ZC_MS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_MS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_ZC_scheme,γ=γ, mapping=pre_inv_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_ZC_BI(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_BI,self).__init__(API=API,equation_class=equation_class,WENO=WENO_ZC_scheme,γ=γ, mapping=pre_inv_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)


## 2D
# WENO JS

class WENO_JS_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_JS_M_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_M_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=post_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_JS_MS_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_MS_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_JS_BI_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_BI_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO Z

class WENO_Z_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_Z_M_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_M_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=post_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Z_MS_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_MS_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Z_BI_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Z_BI_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO Z+

class WENO_Zp_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_Zp_M_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_M_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=post_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Zp_MS_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_MS_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Zp_BI_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_BI_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO ZC

class WENO_ZC_MS_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_MS_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_ZC_scheme,γ=γ, mapping=pre_inv_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_ZC_BI_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_BI_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_ZC_scheme,γ=γ, mapping=pre_inv_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

WENO_dict={
'WENO linear'     : WENO_linear,
'WENO-JS'         : WENO_JS,
'WENO-JS (M)'     : WENO_JS_M,
'WENO-JS (MS)'    : WENO_JS_MS,
'WENO-JS (BI)'    : WENO_JS_BI,
'WENO-Z'          : WENO_Z,
'WENO-Z (M)'      : WENO_Z_M,
'WENO-Z (MS)'     : WENO_Z_MS,
'WENO-Z (BI)'     : WENO_Z_BI,
'WENO-Zp'         : WENO_Zp,
'WENO-Zp (M)'     : WENO_Zp_M,
'WENO-Zp (MS)'    : WENO_Zp_MS,
'WENO-Zp (BI)'    : WENO_Zp_BI,
'WENO-Zp'         : WENO_Zp,
'WENO-ZC (MS)'    : WENO_ZC_MS,
'WENO-ZC (BI)'    : WENO_ZC_BI,
'WENO-JS 2D'      : WENO_JS_2D,
'WENO-JS (M_) 2D' : WENO_JS_M_2D,
'WENO-JS (MS) 2D' : WENO_JS_MS_2D,
'WENO-JS (BI) 2D' : WENO_JS_BI_2D,
'WENO-Z 2D'       : WENO_Z_2D,
'WENO-Z (M) 2D'   : WENO_Z_M_2D,
'WENO-Z (MS) 2D'  : WENO_Z_MS_2D,
'WENO-Z (BI) 2D'  : WENO_Z_BI_2D,
'WENO-Zp 2D'      : WENO_Zp_2D,
'WENO-Zp (M) 2D'  : WENO_Zp_M_2D,
'WENO-Zp (MS) 2D' : WENO_Zp_MS_2D,
'WENO-Zp (BI) 2D' : WENO_Zp_BI_2D,
'WENO-Zp 2D'      : WENO_Zp_2D,
'WENO-ZC (MS) 2D' : WENO_ZC_MS_2D,
'WENO-ZC (BI) 2D' : WENO_ZC_BI_2D
}