using LinearAlgebra
using .EulerSystem2D

function Euler2D(Q, γ, Δx, Δy, CFL, FinalTime, Force, BoundaryConditionX,  BoundaryConditionY)
   Time = 0.0
   Δ = min(Δx, Δy)

   while Time < FinalTime
      Δt = CFL*Δ/MaximumEigenvalue(Q, γ)
      if Δt > (FinalTime - Time)
         Δt = FinalTime - Time
      end

      # 3rd order TVD Runge-Kutta scheme
      Q1 =      Q          -     Δt * (FluxDerivativeX(Q, γ, Δx, BoundaryConditionX)  + FluxDerivativeY(Q, γ, Δy, BoundaryConditionY)  .- Force(Q))
      Q2 = (3.0*Q +     Q1 -     Δt * (FluxDerivativeX(Q1, γ, Δx, BoundaryConditionX) + FluxDerivativeY(Q1, γ, Δy, BoundaryConditionY) .- Force(Q1))) / 4.0
      Q  = (    Q + 2.0*Q2 - 2.0*Δt * (FluxDerivativeX(Q2, γ, Δx, BoundaryConditionX) + FluxDerivativeY(Q2, γ, Δy, BoundaryConditionY) .- Force(Q2))) / 3.0

      Time  = Time + Δt;
   end
   return Q
end

function FluxDerivativeX(Q, γ, Δx, BoundaryConditionX)
   Ord = 5 # The order of the scheme
   N = size(Q, 1)
   M = size(Q, 2)
   F_half = zeros(N+1, M, 4)
   G_half = zeros(1, 4)
   #Qi = zeros(Ord+1, 3)   # Não adianta pré-alocar assim

   QB = BoundaryConditionX(Q)
   #M = MaximumEigenvalue(QB, γ)

   for j = 1:M
      for i = 1:N+1
         Qi = QB[i:i+Ord, j, :]
         Λ = EigenvaluesX(Qi, γ)
         M = maximum(abs.(Λ))

         R, L = EigensystemX(Qi, γ)

         W = Qi*L       # Transforms into characteristic variables
         G = Λ.*W       # The flux for the characteristic variables is Λ * L*QB
         for k = 1:4    # WENO reconstruction of the flux G
            G_half[k] = ReconstructedFlux(G[:,k], W[:,k], M)
         end

         F_half[i,j,:] = G_half*R # Brings back to conservative variables
      end
   end

   return (F_half[2:end,:,:] - F_half[1:end-1,:,:]) / Δx # Derivative of Flux
end

function FluxDerivativeY(Q, γ, Δy, BoundaryConditionY)
   Ord = 5 # The order of the scheme
   N = size(Q, 1)
   M = size(Q, 2)
   F_half = zeros(N, M+1, 4)
   G_half = zeros(1, 4)
   #Qj = zeros(Ord+1, 3)   # Não adianta pré-alocar assim

   QB = BoundaryConditionY(Q)
   #M = MaximumEigenvalue(QB, γ)

   for j = 1:M+1
      for i = 1:N
         Qj = QB[i, j:j+Ord, :]
         Λ = EigenvaluesY(Qj, γ)
         M = maximum(abs.(Λ))

         R, L = EigensystemY(Qj, γ)

         W = Qj*L       # Transforms into characteristic variables
         G = Λ.*W       # The flux for the characteristic variables is Λ * L*QB
         for k = 1:4    # WENO reconstruction of the flux G
            G_half[k] = ReconstructedFlux(G[:,k], W[:,k], M)
         end

         F_half[i,j,:] = G_half*R # Brings back to conservative variables
      end
   end

   return (F_half[:,2:end,:] - F_half[:,1:end-1,:]) / Δy # Derivative of Flux
end

function ReconstructedFlux(F, Q, M)
   ReconstructionLTR = WenoZ5ReconstructionLTR
   #ReconstructionLTR = WenoM5ReconstructionLTR

   F_plus  = (F + M*Q)/2
   F_minus = (F - M*Q)/2

   F_half_plus  = ReconstructionLTR(F_plus)
   F_half_minus = ReconstructionLTR(F_minus[end:-1:1])

   return F_half_plus + F_half_minus
end

function WenoZ5ReconstructionLTR(Q)
   ɛ = 10.0^(-40)
   # Calcula os indicadores de suavidade locais
   β0 = (1/2*Q[1] - 2*Q[2] + 3/2*Q[3])^2 + 13/12*(Q[1] - 2*Q[2] + Q[3])^2
   β1 = (-1/2*Q[2] + 1/2*Q[4])^2 + 13/12*(Q[2] - 2*Q[3] + Q[4])^2
   β2 = (-3/2*Q[3] + 2*Q[4] - 1/2*Q[5])^2 + 13/12*(Q[3] - 2*Q[4] + Q[5])^2
   # Calcula o indicador de suavidade global
   τ = abs(β0 - β2)
   # Calcula os pesos do WENO-Z
   α0 = (1/10) * (1 + (τ/(β0 + ɛ))^2)
   α1 = (6/10) * (1 + (τ/(β1 + ɛ))^2)
   α2 = (3/10) * (1 + (τ/(β2 + ɛ))^2)
   sum_α = α0 + α1 + α2
   ω0 = α0 / sum_α
   ω1 = α1 / sum_α
   ω2 = α2 / sum_α
   # Calcula os fhat em cada subestêncil
   fhat0 = (2*Q[1] - 7*Q[2] + 11*Q[3])/6
   fhat1 = (-Q[2] + 5*Q[3] + 2*Q[4])/6
   fhat2 = (2*Q[3] + 5*Q[4] - Q[5])/6
   #Finalmente, calcula o fhat do estêncil todo
   return ω0*fhat0 + ω1*fhat1 + ω2*fhat2
end

g(ω, d) = (ω*(d+d^2-3*d*ω+ω^2))/(d^2+ω*(1-2*d))

function WenoM5ReconstructionLTR(Q)
   ɛ = 10.0^(-40)
   # Calcula os indicadores de suavidade locais
   β0 = (1/2*Q[1] - 2*Q[2] + 3/2*Q[3])^2 + 13/12*(Q[1] - 2*Q[2] + Q[3])^2
   β1 = (-1/2*Q[2] + 1/2*Q[4])^2 + 13/12*(Q[2] - 2*Q[3] + Q[4])^2
   β2 = (-3/2*Q[3] + 2*Q[4] - 1/2*Q[5])^2 + 13/12*(Q[3] - 2*Q[4] + Q[5])^2
   # Calcula os pesos do WENO-JS
   α0 = (1/10) * (1/(β0 + ɛ)^2)
   α1 = (6/10) * (1/(β1 + ɛ)^2)
   α2 = (3/10) * (1/(β2 + ɛ)^2)
   sum_α = α0 + α1 + α2
   ω0 = α0 / sum_α
   ω1 = α1 / sum_α
   ω2 = α2 / sum_α
   # Mapeia
   α0 = g(ω0, 1/10)
   α1 = g(ω1, 6/10)
   α2 = g(ω2, 3/10)
   sum_α = α0 + α1 + α2
   ω0 = α0 / sum_α
   ω1 = α1 / sum_α
   ω2 = α2 / sum_α
   # Calcula os fhat em cada subestêncil
   fhat0 = (2*Q[1] - 7*Q[2] + 11*Q[3])/6
   fhat1 = (-Q[2] + 5*Q[3] + 2*Q[4])/6
   fhat2 = (2*Q[3] + 5*Q[4] - Q[5])/6
   #Finalmente, calcula o fhat do estêncil todo
   return ω0*fhat0 + ω1*fhat1 + ω2*fhat2
end

function PeriodicGhostPointsX(Q)
   Qg = [Q[end-2:end,:,:]; Q; Q[1:3,:,:]]
end

function PeriodicGhostPointsY(Q)
   Qg = [Q[:,end-2:end,:] Q Q[:,1:3,:]]
end
