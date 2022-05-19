from aux_equation import *

def WENO_JS(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2):
    β = β*(δ+0.1)
    # Calcula os pesos do WENO-JS
    λ = (1/(β + ɛ))**p
    α = mapping(λ, API, map_function)
    return α

def WENO_Z(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2):
    β = β*(δ+0.1)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z
    λ = 1 + (τ/(β + ɛ))**p
    α = mapping(λ, API, map_function)
    return α

def WENO_Zp(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2):
    β = β*(δ+0.1)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ɛ)/(β + ɛ)
    λ = 1 + γ**p + (Δx**(2/3))/γ
    α = mapping(λ, API, map_function)
    return α

def WENO_Zp_net_expo(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ɛ)/(β + ɛ)
    λ = 1 + γ**p+(Δx**δ)/γ
    α = mapping(λ, API, map_function)
    return α

def WENO_ZC(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2):
    β = β*(δ+0.1)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    γ = (τ + ɛ)/(β + ɛ)
    λ = (1 + γ**p)
    α = mapping(λ, API, map_function)
    α = α + API.matmul((Δx**(2/3))/γ, B)
    return α

def WENO_ZC_net_expo(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    γ = (τ + ɛ)/(β + ɛ)
    λ = (1 + γ**p)
    α = mapping(λ, API, map_function)
    α = α + API.matmul((Δx**(δ))/γ, B)
    return α

class simulation:
    
    def __init__(self, API, equation_class, WENO, mapping=null_mapping, map_function=lambda x:x, network=None, p=2):
        
        self.equation = equation_class(API, WENO, network=network, mapping=mapping, map_function=map_function, p=p)
        self.API      = API
        self.WENO     = WENO
        self.network  = network
        self.p        = p

        self.mapping      = mapping
        self.map_function = map_function

        self.Sim      = API.function(self.Sim_graph)
        self.Sim_step = API.function(self.Sim_step_graph)

        self.Get_weights_graph = self.equation.Get_weights_graph
        self.Get_weights       = API.function(self.Get_weights_graph)

        self.DerivadaEspacial_graph = self.equation.DerivadaEspacial
        self.DerivadaEspacial       = API.function(self.DerivadaEspacial_graph)
    
    def Sim_graph(self, u, t_final, Δx, CFL, fronteira):    
        
        t = 0.0*self.equation.maximum_speed(u) # Instante de tempo incial para a computação
        self.API.pretty_print(self.API.squeeze(t), end='\r')
        
        while self.API.any(t < t_final):
            Λ  = self.equation.maximum_speed(u)

            Δt = Δx*CFL/Λ  
            Δt = self.API.where(t + Δt > t_final, t_final - t, Δt)

            u = self.Sim_step_graph(u, Δt, Δx, fronteira)
            t = t + Δt # Avançando no tempo
            
            self.API.pretty_print(self.API.squeeze(t), end='\r')
        
        return u
    
    def Sim_step_graph(self, u, Δt, Δx, fronteira):
        u1 =    u        -   Δt*self.equation.DerivadaEspacial( u, Δx, fronteira)
        u2 = (3*u +   u1 -   Δt*self.equation.DerivadaEspacial(u1, Δx, fronteira)) / 4.0
        u  = (  u + 2*u2 - 2*Δt*self.equation.DerivadaEspacial(u2, Δx, fronteira)) / 3.0
        return u