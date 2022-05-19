using Plots
plotly()

function MalhaRetangular(N)
   Δ = 1.0/N
   Δx = Δ; Δy = Δ
   x = 0.0:Δx:0.25
   y = Δy:Δy:1.0-Δy

   X = repeat(reshape(x, length(x), 1), 1, length(y))
   Y = repeat(reshape(y, 1, length(y)), length(x), 1)

   return X, Y, Δ
end

function CondiçãoInicialRayleighTaylor(X :: Array{Float64, 2}, Y :: Array{Float64, 2}, γ)
   R = zeros(size(X))
   P = zeros(size(X))
   U = zeros(size(X))
   V = zeros(size(X))
   for i = eachindex(R)
      if Y[i] < 0.5
         R[i] = 2.0
         P[i] = 2.0*Y[i] + 1.0
      else
         R[i] = 1.0
         P[i] = Y[i] + 1.5
      end
      a = √(γ * P[i] / R[i])
      V[i] = -0.025 * a * cos(8.0*π*X[i]);
   end

   E = P/(γ-1.0) + R.*(U.^2 + V.^2)/2.0

   #Q0 = cat(R, R.*U, R.*V, E, dims=3)
   Q0 = [R;;; R.*U;;; R.*V;;; E]
   return Q0
end

function CondiçãoInicialRayleighTaylor(N :: Int64, γ)
   X, Y, Δ = MalhaRetangular(N)
   Q0 = CondiçãoInicialRayleighTaylor(X, Y, γ)
   return X, Y, Δ, Q0
end

function RayleighTaylorGravity(Q :: Array{Float64,3})
   g = -1.0
   # for i = 1:size(U,1)
   #    for j = 1:size(U,2)
   #       F[i,j,3] = -g * U[i,j,1]
   #       F[i,j,4] = -g * U[i,j,3]
   #    end
   # end
   Z = zeros(size(Q,1), size(Q,2), 1)
   F = [Z;;;Z;;;-g*Q[:,:,1];;;-g*Q[:,:,3]]
   return F
end

function RayleighTaylorGhostPointsX(Q)
   #c = [1.0,-1.0,1.0,1.0]
   # for i = 1:3; for k = 1:4; Ug[i,k] = c[k]*U[end-i,j,k]; end; end
   # for i = 4:size(Ug,1)-3; for k = 1:4; Ug[i,k] = U[i-3,j,k]; end; end
   # for i = -2:0; for k = 1:4; Ug[size(Ug,1)+i,k] = c[k]*U[1-i,j,k]; end; end
   Qg = [[Q[3:-1:1, :, 1];;;-Q[3:-1:1, :, 2];;;Q[3:-1:1, :, 3];;;Q[3:-1:1, :, 4]];
         Q;
         [Q[end:-1:end-2, :, 1];;;-Q[end:-1:end-2, :, 2];;;Q[end:-1:end-2, :, 3];;;Q[end:-1:end-2, :, 4]]]
   return Qg
end

function RayleighTaylorGhostPointsY(Q, γ)
   # for j = 1:3
   #    Ug[j,1] = 2.0
   #    Ug[j,2] = 0.0
   #    Ug[j,3] = 0.0
   #    Ug[j,4] = 1.0/(5.0/3.0 - 1.0)
   # end
   # for j = 4:size(Ug,1)-3; for k = 1:4; Ug[j,k] = U[i,j-3,k]; end; end
   # for j = -2:0
   #    Ug[size(Ug,1)+j,1] = 1.0
   #    Ug[size(Ug,1)+j,2] = 0.0
   #    Ug[size(Ug,1)+j,3] = 0.0
   #    Ug[size(Ug,1)+j,4] = 2.5/(5.0/3.0 - 1.0)
   # end
   M = size(Q,1)
   Qg = [repeat([2.0;;;0.0;;;0.0;;;1.0/(γ-1.0)], M, 3);;
         Q;;
         repeat([1.0;;;0.0;;;0.0;;;2.5/(γ-1.0)], M, 3)]
   return Qg
end

function grafico(x, y, U, título)
   gui(plot(x, y, U[:,:,1], st = :surface, aspect_ratio = 0.75, camera = (0,89.5), title = título, size = (800,800)))
end

function RayleighTaylor(N :: Integer)
   γ = 5.0/3.0
   #N = 160
   x, y, Δ, U0 = CondiçãoInicialRayleighTaylor(N, γ)
   cfl = 0.6
   t_final = 1.95
   GhostPointsX(U) = RayleighTaylorGhostPointsX(U)
   GhostPointsY(U) = RayleighTaylorGhostPointsY(U, γ)
   U = Euler2D(U0, γ, Δ, Δ, cfl, t_final, RayleighTaylorGravity, GhostPointsX, GhostPointsY)
   grafico(x, y, U, "T = $t_final")
end
