% Conditional probability table. The variable of interest should be
% associated with the index.  In other words if you want to marginalize out
% the first variable, you should loop over the values of the first variable
% in the indexing of the table.
%
% P(B=0)   P(B=1)
%   |         |
%   V         V
% [(A=0,B=0) (A=0,B=1)  --> sum across row to get P(A=0)
%  (A=1,B=0) (A=1,B=1)] --> sum across row to get P(A=1)

clear,clc
pAB = [.2 .2;
       .5 .1];

% Try to compute a marginal: P(A=0)
a = 1;
rt = 0; % Running total
for b=1:2
    rt = rt + pAB(a,b);
end
disp(rt)