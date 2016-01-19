function undirected_elimination
clear,clc
% http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/...
% 6-438-algorithms-for-inference-fall-2014/lecture-notes/MIT6_438F14_Lec7.pdf
%
% Graph structure:
%  x1---x3---x4
%  |    |    |
%  x2---x5----
%
% Variable domains and potential functions:
% Let's say for now that all variables are binary.

% Naive method: Compute the full probability table for p(x)
px = nan(2,2,2,2,2);
for x1 = 0:1
    for x2 = 0:1
        for x3 = 0:1
            for x4 = 0:1
                for x5 = 0:1
                    x = [x1 x2 x3 x4 x5];
                    px(x1+1,x2+1,x3+1,x4+1,x5+1) = phi12(x)*...
                        phi13(x)*phi25(x)*phi345(x);
                end
            end
        end
    end
end

% Partition function to normalize probabilities
Z = sum(sum(sum(sum(sum(px)))));

% Perform the normalization
px = px/Z;

% Here is the full probability table for p(x) where the index for a
% variable (which can be 1 or 2 because of MATLAB) corresponding to the
% value of that variable being 0 or 1, respectively (i.e., binary).

% disp(px)

%% Query the model for a marginal, unconditional on other variables
% Very crude query: Suppose we want the marginal probability of x1 given no
% knowledge of the other variables. We simply need to sum over all other
% variables
x1 = 0;
px1equals0 = 0;
for x2 = 0:1
    for x3 = 0:1
        for x4 = 0:1
            for x5 = 0:1
                px1equals0 = px1equals0 + px(x1+1,x2+1,x3+1,x4+1,x5+1);
            end
        end
    end
end
px1equals0

%% Query the model for a marginal, conditioned on some other variable
% Very crude query: Suppose we want the marginal probability of x1 given
% x5. We simply need to sum over all other variables, BUT now we need to
% renormalize because we're not summing over everything. To see this, think
% about:
%
% p(x1 = 0|x5 = 0) = p(x1 = 0,x5 = 0)/p(x5 = 0)
%
% But, the tables we actually posess are p(x1,x2,x3,x4,x5), so to get the
% tables corresponding to p(x1,x5) and p(x5), we need to sum over all the
% other variables that don't appear in these expressions.
%
%
% First compute the joint prob. of x1 and x5 (marginalize out x2, x3, x4)
x1 = 0; % Query state of x1
x5 = 0; % Given knowledge of x5
px1equals0andx5equals0 = 0;
for x2 = 0:1
    for x3 = 0:1
        for x4 = 0:1
                px1equals0andx5equals0 = px1equals0andx5equals0 + ...
                    px(x1+1,x2+1,x3+1,x4+1,x5+1);
        end
    end
end
% disp(px1equals0andx5equals0)

% Now compute the normalization factor, i.e., the marginal: p(x5)
x5 = 0;
px5equals0 = 0;
for x1 = 0:1
    for x2 = 0:1
        for x3 = 0:1
            for x4 = 0:1
                px5equals0 = px5equals0 + ...
                    px(x1+1,x2+1,x3+1,x4+1,x5+1);
            end
        end
    end
end
% disp(px5equals0)

% Put it all together
px1equals0givenx5equals0 = px1equals0andx5equals0/px5equals0;

%% Perform the same inference using the elimination algorithm
% Try to compute p(x1 = 0) using the elimination algorithm.
%
% Inputs:
% 1) Potentials phi_c for c \in C (the set of maximal cliques)
% 2) Subset A of variables for which we are going to comptue the marginal
% 3) Elimination ordering I: in this case, 5,4,3,2,1
%
% Outputs: Marginal p_a(.)
%
% Let PHI be the set of active potentials. Initialize PHI to be the set of
% input potentials: PHI = {phi12, phi13, phi25, phi345}
%
% For node in I that is not in A:
%
%   Let S_i be the set of nodes (not including i) that share a potential
%   with node i.
%
%   Let PHI_i be the set of potentials in PHI involving x_i
%
%   Compute: m_i(x_S_i) = sum_(x_i)prod_(phi \in PHI) phi(x_i U x_S_i)
%
% Remove elements of PHI_i from PHI
% Add m_i to PHI
%
% End For-Loop
% Normalize

% For-loop step iteration 1
% i = 5             <-- first in elim ordering
% S_i = {2,3,4}     <-- nodes 2, 3 and 4 share a potential with 5
% PHI_i = {phi25, phi345}
% m_5(x2,x3,x4) = sum_x5 ( phi25(x2,x5)*phi345(x3,x4,x5) )

m5_234 = zeros(2,2,2);

% COMPUTE OUR FIRST FACTOR PRODUCT: 
%   phi_product(x2,x3,x4,x5) = phi25(x2,x5)*phi345(x3,x4,x5)
% AND SUM OUT x5:
%   m_5(x2,x3,x4) = sum_x5 ( phi_product(x2,x3,x4,x5) )
%
% QUESTION: Can we swap the sum and product operations?
%
% We will be summing over x5 based on x2, x3, x4.
for x5=0:1
    % For each value of x5, loop over the possible x2, x3, and x4s.
    for x2=0:1
        for x3=0:1
            for x4=0:1
                % Shouldn't matter what x1 is, so make it NaN
                x = [nan,x2,x3,x4,x5];
                p = phi25(x)*phi345(x);
                m5_234(x2+1,x3+1,x4+1) = m5_234(x2+1,x3+1,x4+1) + p;
            end
        end
    end
end
% disp(m5_234)

% Now the graph looks like this because we have eliminated x5 and made 2-3-4 into a clique:
%  x1---x3---x4
%  |    |    |
%  -----x2----

% For-loop step iteration 2
% i = 4             <-- second in elim ordering
% S_i = {2,3}       <-- nodes 2 and 3 share a potential with 4
% PHI_i = {phi234}
% m_4(x2,x3) = sum_x4 ( phi234(x2,x3,x4) )

m4_23 = zeros(2,2);
for x4=0:1
    for x2=0:1
        for x3=0:1
            p = m5_234(x2+1,x3+1,x4+1);
            m4_23(x2+1,x3+1) = m4_23(x2+1,x3+1) + p;
        end
    end
end
% disp(' ')
% disp(m4_23)

% Now the graph looks like this because we have eliminated x4 and made 2-3
% into a clique (which it already was, so no new edges have been added).
%  x1---x3
%  |    |
%  -----x2

% For-loop step iteration 3
% i = 3             <-- third in elim ordering
% S_i = {1,2}       <-- nodes 1 and 2 share a potential with 3
% PHI_i = {phi13,m4_23}     <-- potentials that we need to work with
% m_3(x1,x2) = sum_x3 ( phi13(x1,x3)*m4_23(x2,x3 )

m3_12 = zeros(2,2);
for x3=0:1
    for x1=0:1
        for x2=0:1
            x=[x1,x2,x3,nan,nan];
            p = m4_23(x2+1,x3+1)*phi13(x);
            m3_12(x1+1,x2+1) = m3_12(x1+1,x2+1) + p;
        end
    end
end
% disp(' ')
% disp(m3_12)

% Now the graph looks like this because we have eliminated x3 and made 1-2
% into a clique (which it already was, so no new edges have been added).
%  x1----
%  |    |
%  -----x2

% For-loop step iteration 4
% i = 2             <-- fourth in elim ordering
% S_i = {1}         <-- node 1 shares potential(s) with 2
% PHI_i = {phi12,m3_12}     <-- potentials that we need to work with
% m_2(x1) = sum_x2 ( phi12(x1,x2)*m3_12(x1,x2 )

m2_1 = zeros(2,1);
for x2=0:1
    for x1=0:1
        x = [x1,x2,nan,nan,nan];
        p = m3_12(x1+1,x2+1)*phi12(x);
        m2_1(x1+1) = m2_1(x1+1) + p;
    end
end
% disp(' ')
% disp(m2_1)

% Now we have a function that takes in x1 only.  This is our unnormalized
% marginal probability distribution, and we need to normalize it.
px1 = m2_1/sum(m2_1);
px1equals0 = px1(1)

end

function potential = phi12(x)
x1 = x(1);
x2 = x(2);
if (x1 == 1 && x2 == 1)
    potential = 1;
elseif (x1 == 0 && x2 == 0)
    potential = 1;
else
    potential = 0;
end
end

function potential = phi13(x)
x1 = x(1);
x3 = x(3);
if (x1 == 1 && x3 == 1)
    potential = 1;
elseif (x1 == 0 && x3 == 0)
    potential = 1;
else
    potential = 0;
end
end

function potential = phi25(x)
x2 = x(2);
x5 = x(5);
if (x2 == 1 && x5 == 1)
    potential = 1;
elseif (x2 == 0 && x5 == 0)
    potential = 1;
else
    potential = 0;
end
end

function potential = phi345(x)
x3 = x(3);
x4 = x(4);
x5 = x(5);
if (x3 == 1 && x4 == 1 && x5 == 1)
    potential = 1;
elseif (x3 == 0 && x4 == 0 && x5 == 0)
    potential = 1;
else
    potential = 0;
end
end