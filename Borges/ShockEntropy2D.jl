using Plots
plotly()

function MalhaRetangular(N)
   Δ = 10.0/N
   Δx = Δ; Δy = Δ
   x = -5.0:Δx:5.0
   y = -1.0:Δy:1.0-Δy

   X = repeat(reshape(x, length(x), 1), 1, length(y))
   Y = repeat(reshape(y, 1, length(y)), length(x), 1)

   return X, Y, Δ
end

function CondiçãoInicialShockEntropy2D(X :: Array{Float64, 2}, Y :: Array{Float64, 2}, γ :: Float64, θ :: Float64)
   R = zeros(size(X))
   U = zeros(size(X))
   V = zeros(size(X))
   P = ones(size(X))

   I = ones(size(X))
   l = X .< -4.0
   r = X .>= -4.0

   R[l] = I[l] * 27.0/7.0
   U[l] = I[l] * 4.0*√(35.0)/9.0
   P[l] = I[l] * 31.0/3.0

   R[r] = 1.0 .+ sin.(X[r]*cos(θ)*2.0*π + Y[r]*sin(θ)*2.0*π)/5.0

   E = P/(γ-1.0) + R.*(U.^2 + V.^2)/2.0

   #Q0 = cat(R, R.*U, R.*V, E, dims=3)
   Q0 = [R;;; R.*U;;; R.*V;;; E]
   return Q0
end

function CondiçãoInicialShockEntropy2D(N :: Int64, γ :: Float64, θ :: Float64)
   X, Y, Δ = MalhaRetangular(N)
   Q0 = CondiçãoInicialShockEntropy2D(X, Y, γ, θ)
   return X, Y, Δ, Q0
end

function NullForce(Q :: Array{Float64,3})
   return 0.0
end

function ShockEntropy2DGhostPointsX(Q, γ, Δx, y, θ)
   ρl, ul, pl = 27.0/7.0, 4.0*√(35.0)/9.0, 31.0/3.0
   El = pl/(γ-1.0) + ρl.*(ul.^2)/2.0
   Ql = repeat([ρl;;; ρl*ul;;; 0.0;;; El], 3, size(Q,2))

   Xr = repeat([5.0+Δx; 5.0+2*Δx; 5.0+3*Δx], 1, size(Q,2))
   Yr = y[1:3,:]
   ρr = 1.0 .+ sin.(5.0*Xr*cos(θ) + 5.0*Yr*sin(θ))/5.0
   pr = ones(size(ρr))
   ur = zeros(size(ρr))
   vr = zeros(size(ρr))
   Er = pr/(γ-1.0)
   Qr = [ρr;;; ur;;; vr;;; Er]

   Qg = [Ql; Q; Qr]
   return Qg
end

function ShockEntropy2DGhostPointsY(Q)
   Qg = [Q[:,end-2:end,:];;
         Q;;
         Q[:,1:3,:]]
   return Qg
end

function grafico(x, y, U, título)
   for k = 1:1
      gui(plot(x, y, U[:,:,k], st = :surface, aspect_ratio = 0.75, camera = (0,89.5), title = título * "$k", size = (800,800)))
   end
end

function ShockEntropy2D(N :: Integer)
   γ = 7.0/5.0
   θ = π/6.0
   #N = 160
   x, y, Δ, U0 = CondiçãoInicialShockEntropy2D(N, γ, θ)
   cfl = 0.5
   t_final = 1.8
   #grafico(x, y, U0, "Condição Inicial")
   GhostPointsX(U) = ShockEntropy2DGhostPointsX(U, γ, Δ, y, θ)
   GhostPointsY(U) = ShockEntropy2DGhostPointsY(U)
   U = Euler2D(U0, γ, Δ, Δ, cfl, 0.001, NullForce, GhostPointsX, GhostPointsY)
   U = Euler2D(U0, γ, Δ, Δ, cfl, t_final, NullForce, GhostPointsX, GhostPointsY)
   grafico(x, y, U, "T = $t_final")
   return nothing
end
