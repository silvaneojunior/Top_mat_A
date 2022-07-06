classdef Euler2DDoubleMachBoundaryConditionY < FDDesigner.Space.BoundaryConditions.BoundaryCondition
   properties
      L
      R
      x
      x0
      xL
      xR
      uL
      uR
      Reflect
   end
   
   methods
      function obj = Euler2DDoubleMachBoundaryConditionY(L, R, gam, x0, x)
         obj.L = L;
         obj.R = R;
         obj.x0 = x0;
         obj.x = x;
         obj.xL = (x < x0);
         obj.xR = (x >= x0);
         obj.uL = [8; 4*8.25*sqrt(3); -4*8.25; 116.5/(gam - 1) + 4*(8.25)^2];
         obj.uR = [1.4; 0; 0; 1/(gam - 1)];
         obj.Reflect = [1; 1; -1; 1];
      end
      
      function ug = AddGhostPoints(obj, u, t)
         ug = obj.Particular_AddGhostPoints(u, obj.L, obj.R, t);
      end
      
      function ug = AddOnePoint(obj, u, t)
         ug = obj.Particular_AddGhostPoints(u, 1, 1, t);
      end     
   end
   
   methods(Access=private)
      function ug = Particular_AddGhostPoints(obj, u, L, R, t)
         Lu = size(u,1); Lx = size(u,3);
         
         UL = zeros(Lu, L, Lx);
         UL(:, :, obj.xL) = repmat(obj.uL, [1 L sum(obj.xL(:))]);
         UL(obj.Reflect == 1, :, obj.xR) = u(obj.Reflect == 1, L+1:-1:2, obj.xR);
         UL(obj.Reflect == -1, :, obj.xR) = -u(obj.Reflect == -1, L+1:-1:2, obj.xR);
         
         UR = zeros(Lu, R, Lx);
         s = obj.x0 + (1 + 20*t)/sqrt(3);
         xL = (obj.x < s);
         xR = (obj.x >= s);
         UR(:, :, xL) = repmat(obj.uL, [1 R sum(xL(:))]);
         UR(:, :, xR) = repmat(obj.uR, [1 R sum(xR(:))]);
         
         ug = [UL u UR];
      end
   end
end