function undirected_elimination
% http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/...
% 6-438-algorithms-for-inference-fall-2014/lecture-notes/MIT6_438F14_Lec7.pdf
%
% Graph structure:
%  x1---x3---x4
%  |    |    |
%  x2---x5----
%
% Variable domains and potential functions:
% Let's say for now that all variables are binary. Potential functions
% return 1 when variables are all the same (i.e., all zero or all one), and
% return 0 otherwise.

% Query: Marginal distribution of x1 (given nothing)

% Elimination ordering:
I = [5 4 3 2 1];


end

%% Potential functions
function phi = phi12(x1,x2)
phi_tab = [1 0;0 1];
phi = phi_tab(x1+1,x2+1);
end

function phi = phi13(x1,x3)
phi_tab = [1 0;0 1];
phi = phi_tab(x1+1,x3+1);
end

function phi = phi25(x2,x5)
phi_tab = [1 0;0 1];
phi = phi_tab(x2+1,x5+1);
end

function phi = phi345(x3,x4,x5)
phi_tab = zeros(2,2,2);
phi_tab(1,1,1) = 1;
phi_tab(2,2,2) = 1;
phi = phi_tab(x3+1,x4+1,x5+1);
end