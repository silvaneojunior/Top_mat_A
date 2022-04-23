from aux_equation import *

def WENO_Z(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    β=β*(δ+0.1)
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    return α

def WENO_Z_alpha(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    α=α*(δ+0.1)
    return α

def WENO_Z_plus(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ =(τ + ɛ)/(β + ɛ)
    α    = API.matmul((1 + γ**p+δ/γ), B)
    return α

def WENO_JS(β,δ,API,p=2):
    β=β*(δ+0.1)
    # Calcula os pesos do WENO-JS
    α    = API.matmul(((1/(β + ɛ))**p), B)
    return α

def WENO_mix(β,δ,API,p=2):
    # Calcula o indicador de suavidade global
    #δ=δ[...,1:2]
    β=β*δ+(1-δ)*np.asarray([[1/10,6/10,3/10]])
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    return α

def Continuous_case(β,δ,API,p=2):
    β=β*0+(1-0)*np.asarray([[1/10,6/10,3/10]])
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z
    α    = API.matmul((1 + (τ/(β + ɛ))**p), B)
    return α

class simulation:
    def __init__(self,API,equation_class,WENO,network=None,p=2):
        self.equation=equation_class(API, WENO, network, p)
        self.API=API
        self.WENO=WENO
        self.network=network
        self.p=p

        self.Sim=API.function(self.Sim_graph)
        self.Sim_step=API.function(self.Sim_step_graph)

        self.Get_weights_graph=self.equation.Get_weights_graph
        self.Get_weights=API.function(self.Get_weights_graph)

        self.DerivadaEspacial_graph=self.equation.DerivadaEspacial
        self.DerivadaEspacial=API.function(self.DerivadaEspacial_graph)
    def Sim_graph(self,u, t_final, Δx, CFL, fronteira):    
        t = 0.0*self.equation.maximum_speed(u) # Instante de tempo incial para a computação
        self.API.pretty_print(self.API.squeeze(t),end='\r')
        while self.API.any(t < t_final):
            Λ  = self.equation.maximum_speed(u)

            Δt = Δx*CFL/Λ  
            Δt = self.API.where(t + Δt > t_final, t_final - t, Δt)

            u=self.Sim_step_graph(u, Δt, Δx, fronteira)
            
            t  = t + Δt # Avançando no tempo
            self.API.pretty_print(self.API.squeeze(t),end='\r')
        return u
    def Sim_step_graph(self,u, Δt, Δx, fronteira):
        u1 = u - Δt*self.equation.DerivadaEspacial(u, Δx, fronteira)
        u2 = (3*u + u1 - Δt*self.equation.DerivadaEspacial(u1, Δx, fronteira)) / 4.0
        u  = (u + 2*u2 - 2*Δt*self.equation.DerivadaEspacial(u2, Δx, fronteira)) / 3.0
        return u