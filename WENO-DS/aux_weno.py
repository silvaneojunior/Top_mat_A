from aux_base import const, B, ε_default
from aux_mapping import *

def WENO_JS_scheme(β,δ,API,Δx,mapping=null_mapping, map_function=lambda x:x,p=2,ε=ε_default):
    β=β*(δ+const(1, API)/10)
    # Calcula os pesos do WENO-JS
    λ = (1/(β + ε))**p
    α = mapping(λ,API,map_function)
    return α

def WENO_Z_scheme(β,δ,API,Δx,mapping=null_mapping, map_function=lambda x:x,p=2,ε=ε_default):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z
    λ = 1 + (τ/(β + ε))**p
    α = mapping(λ, API, map_function)
    return α

def WENO_Zp_scheme(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2,ε=ε_default):
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = 1 + γ**p + (Δx**(const(2, API)/3))/γ
    α = mapping(λ, API, map_function)
    return α

def WENO_Zp_net_expo_scheme(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2,ε=ε_default):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])
    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = 1 + γ**p+(Δx**δ)/γ
    α = mapping(λ, API, map_function)
    return α

def WENO_ZC_scheme(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2,ε=ε_default):
    β = β*(δ+const(1, API)/10)
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = (1 + γ**p)
    α = mapping(λ, API, map_function)
    α = α + API.matmul((Δx**(const(2, API)/3))/γ, B)
    return α

def WENO_ZC_net_expo_scheme(β, δ, API, Δx, mapping=null_mapping, map_function=lambda x:x, p=2,ε=ε_default):
    # Calcula o indicador de suavidade global
    τ = API.abs(β[...,0:1] - β[...,2:3])

    # Calcula os pesos do WENO-Z+
    γ = (τ + ε)/(β + ε)
    λ = (1 + γ**p)
    α = mapping(λ, API, map_function)
    α = α + API.matmul((Δx**(δ))/γ, B)
    return α

class simulation:
    def __init__(self,API,equation_class,WENO,γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=2,ε=ε_default):
        self.equation=equation_class(API, WENO, network=network,mapping=mapping, map_function=map_function, p=p,ε=ε,γ=γ)
        self.API=API
        self.WENO=WENO
        self.network=network
        self.p=p
        self.ε=ε

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

            Δt = Δx*CFL/self.API.abs(Λ)  
            Δt = self.API.where(t + Δt > t_final, t_final - t, Δt)

            u = self.Sim_step_graph(u, Δt, Δx, fronteira, t=t+Δt)
            t = t + Δt # Avançando no tempo
            
            self.API.pretty_print(self.API.squeeze(t), end='\r')
        
        return u
    def Sim_step_graph(self,u, Δt, Δx, fronteira, t=None):
        u1 = u - Δt*self.equation.DerivadaEspacial(u, Δx, fronteira, t=t)
        u2 = (3*u + u1 - Δt*self.equation.DerivadaEspacial(u1, Δx, fronteira, t=t)) / 4.0
        u  = (u + 2*u2 - 2*Δt*self.equation.DerivadaEspacial(u2, Δx, fronteira, t=t)) / 3.0
        return u

class simulation_2D(simulation):
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

        duX=self.equation.DerivadaEspacialX(u, Δx, fronteiraX, t=t)
        duY=self.equation.DerivadaEspacialY(u, Δy, fronteiraY, t=t)
        
        u1 = u - Δt*(duX+duY-Force(u,self.API))

        du1X=self.equation.DerivadaEspacialX(u1, Δx, fronteiraX, t=t)
        du1Y=self.equation.DerivadaEspacialY(u1, Δy, fronteiraY, t=t)

        u2 = (3*u + u1 - Δt*(du1X+du1Y-Force(u1,self.API))) / 4.0

        du2X=self.equation.DerivadaEspacialX(u2, Δx, fronteiraX, t=t)
        du2Y=self.equation.DerivadaEspacialY(u2, Δy, fronteiraY, t=t)

        u  = (u + 2*u2 - 2*Δt*(du2X+du2Y-Force(u2,self.API))) / 3.0
        return u

## 1D
# WENO JS

class WENO_JS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=null_mapping, map_function=lambda x:x,network=None,p=p,ε=ε)

class WENO_JS_M(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_JS_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_JS_scheme,γ=γ, mapping=post_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

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
        super(WENO_Z_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Z_scheme,γ=γ, mapping=post_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

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
        super(WENO_Zp_M,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=post_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Zp_MS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_MS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_Zp_BI(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_Zp_BI,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

# WENO ZC

class WENO_ZC_MS(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_MS,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_inv_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_ZC_BI(simulation):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_BI,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_inv_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)


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
        super(WENO_ZC_MS_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_inv_mapping, map_function=Hong_mapping,network=None,p=p,ε=ε)

class WENO_ZC_BI_2D(simulation_2D):
    def __init__(self,API,equation_class,γ,p=2,ε=ε_default):
        super(WENO_ZC_BI_2D,self).__init__(API=API,equation_class=equation_class,WENO=WENO_Zp_scheme,γ=γ, mapping=pre_inv_mapping, map_function=BI_mapping,network=None,p=p,ε=ε)

WENO_dict={
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