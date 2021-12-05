using LinearAlgebra

#global γ = 1.4

function Euler1D(Q, γ, Δx, CFL, FinalTime, BoundaryCondition)
   Time = 0.0
   while Time < FinalTime
      Δt = CFL*Δx/MaximumEigenvalue(Q, γ)
      if Δt > (FinalTime - Time)
         Δt = FinalTime - Time
      end

      # 3rd order TVD Runge-Kutta scheme
      Q1 =    Q        -   Δt * FluxDerivative(Q, γ, Δx, BoundaryCondition)
      Q2 = (3*Q +   Q1 -   Δt * FluxDerivative(Q1, γ, Δx, BoundaryCondition))/4
      Q  = (  Q + 2*Q2 - 2*Δt * FluxDerivative(Q2, γ, Δx, BoundaryCondition))/3

      Time  = Time + Δt;
   end
   return Q
end

function FluxDerivative(Q, γ, Δx, BoundaryCondition)
   Ord = 5 # The order of the scheme
   Q = BoundaryCondition(Q)

   N = size(Q, 1)
   F_half = zeros(N-Ord, 3)
   G_half = zeros(1, 3)
   Qi = zeros(Ord+1, 3)

   M = MaximumEigenvalue(Q, γ)

   for i = 1:N-Ord
      Qi[1:Ord+1, 1:3] = Q[i:i+Ord, 1:3]
      Λ = Eigenvalues(Qi, γ)

      Qa = Average(Qi)
      R, L = Eigensystem(Qa, γ)

      W = Qi*L       # Transforms into characteristic variables
      G = Λ.*W       # The flux for the characteristic variables is Λ * L*Q
      for j = 1:3    # WENO reconstruction of the flux G
         G_half[j] = ReconstructedFlux(G[:,j], W[:,j], M)
      end

      F_half[i,:] = G_half*R # Brings back to conservative variables
   end

   return (F_half[2:end,:] - F_half[1:end-1,:])/Δx # Derivative of Flux
end

function Average(Q)
   r = 3
   return (Q[r,:] + Q[r+1,:])/2;
end

function Pressure(Q :: Transpose{Float64,Array{Float64,1}}, γ)
   return (γ-1)*(Q[3] - Q[2].^2 ./ Q[1]/2)
end

function Pressure(Q :: Array{Float64,1}, γ)
   return (γ-1)*(Q[3] - Q[2].^2 ./ Q[1]/2)
end

function Pressure(Q :: Array{Float64,2}, γ)
   return (γ-1)*(Q[:,3] - Q[:,2].^2 ./ Q[:,1]/2)
end

function Eigensystem(Q, γ)
   U = Q[2]/Q[1]
   P = Pressure(Q, γ)
   A = sqrt.(γ*P/Q[1]) # Sound speed
   H = (Q[3] + P)/Q[1]    # Enthalpy
   h = 1/(2*H - U^2)

   R = [1  U - A  H - U*A;
        1  U      U^2/2;
        1  U + A  H + U*A]

   L = [U/(2*A) + U^2*h/2  2 - (2*H)*h  U^2*h/2 - U/(2*A);
        -U*h - 1/(2*A)     (2*U)*h      1/(2*A) - U*h;
        h                  -2*h         h]

   return R, L
end

function Eigenvalues(Q, γ)
  U = Q[:,2]./Q[:,1]
  P = Pressure(Q, γ)
  A = sqrt.(γ*P./Q[:,1])

  return [U-A U U+A]
end

function MaximumEigenvalue(Q, γ)
   return maximum(abs.(Eigenvalues(Q, γ)))
end

function ReconstructedFlux(F, Q, M)
   F_plus  = (F + M*Q)/2
   F_minus = (F - M*Q)/2

   F_half_plus  = WenoZ5ReconstructionLTR(F_plus)
   F_half_minus = WenoZ5ReconstructionLTR(F_minus[end:-1:1])

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
