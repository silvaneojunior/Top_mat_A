module EulerSystem2D

export ConservativeVariables, EigensystemX, EigenvaluesX,
   EigensystemY, EigenvaluesY, MaximumEigenvalue

# function Pressure(Q :: Transpose{Float64,Array{Float64,1}}, γ)
#    return (γ-1)*(Q[3] - Q[2].^2 ./ Q[1]/2)
# end
#
function Pressure(Q :: Array{Float64,1}, γ)
   return (γ-1)*(Q[4] - (Q[2].^2+Q[3].^2) ./ Q[1]/2)
end

function Pressure(Q :: Array{Float64,2}, γ)
   return (γ-1)*(Q[:,4] - (Q[:,2].^2+Q[:,3].^2) ./ Q[:,1]/2)
end

function Pressure(Q :: Array{Float64,3}, γ)
   return (γ-1)*(Q[:,:,4] - (Q[:,:,2].^2+Q[:,:,3].^2) ./ Q[:,:,1]/2)
end

function ConservativeVariables(Q, γ)
   return ConservativeVariables(Q[:,:,1], Q[:,:,2], Q[:,:,3], Q[:,:,4], γ)
end

function ConservativeVariables(ρ, u, v, p, γ)
   ρu = ρ.*u
   ρv = ρ.*v
   E = p/(γ-1) + ρ.*(u.^2+v.^2)/2
   return [ρ;;; ρu;;; ρv;;; E]
end

function PrimitiveVariables(Q :: Array{Float64,1}, γ)
   R = Q[1]
   U = Q[2]/R
   V = Q[3]/R
   P = Pressure(Q, γ)
   return R, U, V, P
end

function PrimitiveVariables(Q :: Array{Float64,2}, γ)
   R = Q[:,1]
   U = Q[:,2]./R
   V = Q[:,3]./R
   P = Pressure(Q, γ)
   return R, U, V, P
end

function PrimitiveVariables(Q :: Array{Float64,3}, γ)
   R = Q[:,:,1]
   U = Q[:,:,2]./R
   V = Q[:,:,3]./R
   P = Pressure(Q, γ)
   return R, U, V, P
end

function ArithmeticAverage(Q, γ)
   r = 3
   Qa = (Q[r,:] + Q[r+1,:])/2

   R, U, V, P = PrimitiveVariables(Qa, γ)
   A = sqrt.(γ*P/R)     # Sound speed
   H = (Qa[4] + P)/R    # Enthalpy
   h = 1/(2*H - U^2 - V^2)

   return U, V, A, H, h
end

function EigensystemX(Q, γ)
   Average = ArithmeticAverage
   #Average = RoeAverage

   U, V, A, H, h = Average(Q, γ)

   R = [1.0  U-A  V    H-U*A;
        1.0  U    V    (U^2+V^2)/2.0;
        0.0  0.0  1.0  V;
        1.0  U+A  V    H+U*A]

   I2A, Hh, Uh, Vh = 0.5/A, H*h, U*h, V*h

   L = [U*I2A+Hh-0.5  2.0-2.0*Hh  -V   Hh-0.5-U*I2A;
        -Uh-I2A       2.0*Uh      0.0  I2A-Uh;
        -Vh           2.0*Vh      1.0  -Vh;
        h             -2.0*h      0.0  h]

   return R, L
end

function EigenvaluesX(Q, γ)
   R, U, V, P = PrimitiveVariables(Q, γ)
   A = sqrt.(γ*P./R)

   return [U-A U U U+A]
end

function EigensystemY(Q, γ)
   Average = ArithmeticAverage
   #Average = RoeAverage

   U, V, A, H, h = Average(Q, γ)

   R = [1.0  U    V-A  H-V*A;
        1.0  U    V    (U^2+V^2)/2.0;
        0.0  1.0  0.0  U;
        1.0  U    V+A  H+V*A]

   I2A, Hh, Uh, Vh = 0.5/A, H*h, U*h, V*h

   L = [V*I2A+Hh-0.5  2.0-2.0*Hh  -U   Hh-0.5-V*I2A;
        -Uh           2.0*Uh      1.0  -Uh;
        -Vh-I2A       2.0*Vh      0.0  I2A-Vh;
        h             -2.0*h      0.0  h]

   return R, L
end

function EigenvaluesY(Q, γ)
   R, U, V, P = PrimitiveVariables(Q, γ)
   A = sqrt.(γ*P./R)

   return [V-A V V V+A]
end

function MaximumEigenvalue(Q, γ)
   R, U, V, P = PrimitiveVariables(Q, γ)
   A = sqrt.(γ*P./R)

   MU = max(maximum((U-A).^2), maximum((U+A).^2))
   MV = max(maximum((V-A).^2), maximum((V+A).^2))
   return sqrt(MU+MV)
end

end
