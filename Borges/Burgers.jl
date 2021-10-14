using Plots

function Burgers(u, t_final, Δx, CFL, fronteira)
   t = 0.0
   while t < t_final
      Λ = maximum(abs.(u))
      Δt = Δx*CFL/Λ #Condicao de CFL - deve ser atualizada a cada passo de tempo
      if t + Δt > t_final
         Δt = t_final - t
      end
      #SSP Runge-Kutta 3,3
      u1 = u - Δt*DerivadaEspacial(u, Δx, Λ, fronteira, t)
      Λ = maximum(abs.(u1))
      u2 = (3*u + u1 - Δt*DerivadaEspacial(u1, Δx, Λ, fronteira, t + Δt)) / 4.0
      Λ = maximum(abs.(u2))
      u = (u + 2*u2 - 2*Δt*DerivadaEspacial(u2, Δx, Λ, fronteira, t + Δt/2)) / 3.0
      t = t + Δt
   end
   return u
end

function FronteiraFixa(U, t)
   return [U[1,:]'; U[1,:]'; U[1,:]'; U; U[end,:]'; U[end,:]'; U[end,:]']
end

function FronteiraPeriodica(U, t)
   return [U[end-2,:]'; U[end-1,:]'; U[end,:]'; U; U[1,:]'; U[2,:]'; U[3,:]']
end

function DerivadaEspacial(U, Δx, Λ, AdicionaGhostPoints, t)
   Fhat = zeros(size(U,1)+1)
   U = AdicionaGhostPoints(U, t) # Adiciona ghost cells de acordo com as condições de fronteira
   for i = 3:size(U,1)-3
      u_i = U[i-2:i+3,:] # Estêncil de 6 pontos onde vamos trabalhar

      M = maximum(abs.(u_i)) #Λ
      f_plus = (u_i.^2/2 + M*u_i) / 2
      f_minus = (u_i.^2/2 - M*u_i) / 2

      f_half_minus = WenoZ5ReconstructionMinus(f_plus[1], f_plus[2], f_plus[3], f_plus[4], f_plus[5]) # Aplicar WENO em cada variável característica separadamente
      f_half_plus = WenoZ5ReconstructionPlus(f_minus[2], f_minus[3], f_minus[4], f_minus[5], f_minus[6])

      Fhat[i-2] = f_half_minus + f_half_plus
   end
   return (Fhat[2:end] - Fhat[1:end-1]) / Δx
end

function WenoZ5ReconstructionMinus(u1, u2, u3, u4, u5)
   const ɛ = 10.0^(-40)
   # Calcula os indicadores de suavidade locais
   β0 = (1/2.0*u1 - 2*u2 + 3/2.0*u3)^2 + 13/12.0*(u1 - 2*u2 + u3)^2
   β1 = (-1/2.0*u2 + 1/2.0*u4)^2 + 13/12.0*(u2 - 2*u3 + u4)^2
   β2 = (-3/2.0*u3 + 2*u4 - 1/2.0*u5)^2 + 13/12.0*(u3 - 2*u4 + u5)^2
   # Calcula o indicador de suavidade global
   τ = abs(β0 - β2)
   # Calcula os pesos do WENO-Z
   α0 = (1/10) * (1 + (τ/(β0 + ɛ))^2)
   α1 = (6/10) * (1 + (τ/(β1 + ɛ))^2)
   α2 = (3/10) * (1 + (τ/(β2 + ɛ))^2)
   soma = α0 + α1 + α2
   ω0 = α0 / soma
   ω1 = α1 / soma
   ω2 = α2 / soma
   # Calcula os fhat em cada subestêncil
   fhat0 = (2*u1 - 7*u2 + 11*u3)/6
   fhat1 = (-u2 + 5*u3 + 2*u4)/6
   fhat2 = (2*u3 + 5*u4 - u5)/6
   #Finalmente, calcula o fhat do estêncil todo
   return ω0*fhat0 + ω1*fhat1 + ω2*fhat2
end

function WenoZ5ReconstructionPlus(u1, u2, u3, u4, u5)
   return WenoZ5ReconstructionMinus(u5, u4, u3, u2, u1);
end

#u = Burgers(u0, t_final, Δx, CFL, FronteiraFixa)
#plot(x, u)
