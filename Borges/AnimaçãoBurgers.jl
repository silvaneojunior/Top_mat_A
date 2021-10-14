using Plots

function AnimaçãoBurgers()
   Δx = 0.01
   x = -2:Δx:2
   u = 8*(x.-1/2).*exp.(-16*x.^2)
   CFL = 0.5

   Δt = 0.01          # Δt da animação
   T = 0.0:Δt:2.0     # Frames da animação

   U = zeros(length(u), length(T))
   U[:,1] = u
   t = 0.0
   i = 2
   @gif while t < T[end]
      u = Burgers(u, Δt, Δx, CFL, FronteiraFixa)
      U[:,i] = u
      i += 1
      t += Δt
      plot(x, u, ylims = (-1.2, 1.2))
   end
end
