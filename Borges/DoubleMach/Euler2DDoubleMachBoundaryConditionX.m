classdef Euler2DDoubleMachBoundaryConditionX < FDDesigner.Space.BoundaryConditions.BoundaryCondition
   properties
      L
      R
      uL
   end
   
   methods
      function obj = Euler2DDoubleMachBoundaryConditionX(L, R, gam)
         obj.L = L;
         obj.R = R;
         obj.uL = [8; 4*8.25*sqrt(3); -4*8.25; 116.5/(gam - 1) + 4*(8.25)^2];
      end
      
      function ug = AddGhostPoints(obj, u, ~)
         ug = obj.Particular_AddGhostPoints(u, obj.L, obj.R);
      end
      
      function ug = AddOnePoint(obj, u, ~)
         ug = obj.Particular_AddGhostPoints(u, 1, 1);
      end     
   end
   
   methods(Access=private)
      function ug = Particular_AddGhostPoints(obj, u, L, R)
         Ly = size(u,3);
         
         UL = repmat(obj.uL, [1 L Ly]);         
         UR = u(:, end-1:-1:end-R, :);
         
         ug = [UL u UR];
      end
   end
end