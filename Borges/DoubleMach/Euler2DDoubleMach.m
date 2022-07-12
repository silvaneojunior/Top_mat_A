function u = Euler2DDoubleMach(Nx, Ny, spatial_scheme, fig, restart, T)

gam = 7/5;
p = @(u) (u(4,:,:,:) - (u(2,:,:,:).^2 + u(3,:,:,:).^2)./u(1,:,:,:)/2)*(gam - 1);
ft = FDDesigner.ConservationLaws.ForcingNull();

x0 = 1/6;

cfl = 0.6;

if (nargin == 4) || ~restart % Se é pra computar desde o começo
   dx = 4/Nx;
   x = 0:dx:4;
   dy = 1/Ny;
   y = 0:dy:1;
   X = repmat(x, [1 1 length(y)]);
   Y = repmat(y, [1 1 length(x)]);
   Y = permute(Y, [1 3 2]);

   R1 = (X < x0 + Y/sqrt(3));       % Região 1
   R2 = (X >= x0 + Y/sqrt(3));      % Região 2

   r0 = zeros(length(x), length(y));
   u0 = zeros(length(x), length(y));
   v0 = zeros(length(x), length(y));
   p0 = ones(length(x), length(y));

   r0(R1) = 8;
   u0(R1) = 8.25*sqrt(3)/2;
   v0(R1) = -8.25/2;
   p0(R1) = 116.5;

   r0(R2) = 1.4;

   r0 = reshape(r0, [1 size(r0)]);
   u0 = reshape(u0, [1 size(u0)]);
   v0 = reshape(v0, [1 size(v0)]);
   p0 = reshape(p0, [1 size(p0)]);

   U = [r0; r0.*u0; r0.*v0; p0/(gam - 1) + r0.*(u0.^2 + v0.^2)/2];

   if (nargin < 6), T = 0.2; end

   bcx = Euler2DDoubleMachBoundaryConditionX(3, 3, gam);
   bcy = Euler2DDoubleMachBoundaryConditionY(3, 3, gam, x0, x);
   switch fig
      case -1
         out = FDDesigner.Time.Iteration.NoOutput();
      case 0
         filename = ['double-mach-' spatial_scheme '-' num2str(Nx) 'x-' num2str(Ny) 'y'];
         out = FDDesigner.Time.Iteration.MatOutput(x, y, U, T, 100, filename);
      otherwise
         out = Euler2DDoubleMachOutput(fig, x, y);
   end
   u = Euler2DTest(dx, dy, U, 0, T, p, gam, cfl, bcx, bcy, out, ft, spatial_scheme);
else % Se for reiniciar a computação
   filename = ['double-mach-' spatial_scheme '-' num2str(Nx) 'x-' num2str(Ny) 'y'];
   load([filename '-100t'])
   dx = x(2)-x(1);
   dy = y(2)-y(1);
   I = find(T == max(T));
   bcx = FDDesigner.Space.BoundaryConditions.Reflexive(3, 3, [1;-1;1;1]);
   bcy = FDDesigner.Space.BoundaryConditions.Fixed(3, 3, repmat([2; 0; 0; 1/(gam-1)], [1 length(x) 1]), repmat([1; 0; 0; 2.5/(gam-1)], [1 length(x) 1]));
   out = FDDesigner.Time.Iteration.MatOutputRestart(x, y, U, T, 1.95, filename);
   u = Euler2DTest(dx, dy, squeeze(U(I,:,:,:)), T(I), 1.95, p, gam, cfl, bcx, bcy, out, ft, spatial_scheme);
end
end
