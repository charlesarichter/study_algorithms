function gp
clear,clc,close all

%% Setting up a Gaussian process prior
% A GP is characterized by a set of inputs and a covariance function
% computing the relationship between those inputs.  Samples from this GP
% are outputs corresponding to the set of inputs.  Keep sampling over and
% over and you will get different outputs each time.

% Size of the collection of inputs. Recall that a Gaussian process is
% defined as a collection of random variables, any finite number of which
% is jointly normally distributed (GPML section 2.2, page 13).
n = 500;

% Vector of inputs
X = linspace(-5,5,n)';

% Kernel width
kw = 1;

% Covariance function: covariance between outputs is written as a function
% of the inputs (specifically distance between points in input space). See
% GPML section 2.2, page 14.
%
% The epsilon * identity is added to make the kernel matrix clearly
% positive definite for numerical stability reasons in the Cholesky decomp.
k = kernel(X,X,kw) + 1e-6*eye(n);

% Now take a sample of ouputs (Y) from the Gaussian process defined by
% this covariance function
R = chol(k);
Y = randn(1,n)*R;

% Display the sample from this GP
plot(X,Y)

%% Suppose we have some noise-free observations of real data points
% We will still want to supply a vector of inputs to query the GP and plot
% the results, but these are not training inputs.  These are test inputs.
% Our training inputs will be the real data points.

% Training data points
X = [-1;0;1];
f = [.2;.6;-.5];

% ALSO: In the noise-free case, if you give it a dataset with data points
% too close together, Sigma will be singular!
% X = [-1;0;1;1];
% f = [.2;.6;-.5;-.6];

% Query data points
Xstar = linspace(-5,5,n)';

% Kernel matrices for combinations of training and query inputs
k_x_x = kernel(X,X,kw);
k_xstar_x = kernel(Xstar,X,kw);
k_x_xstar = kernel(X,Xstar,kw);
k_xstar_xstar = kernel(Xstar,Xstar,kw);

% Predictive distribution: Condition the prior on the observations
mu = k_xstar_x / k_x_x * f;
Sigma = k_xstar_xstar - k_xstar_x / k_x_x * k_x_xstar;

% Draw a sample from the predictive distribution
R = chol(Sigma + 1e-6*eye(n));
Ystar = mu' + randn(1,n)*R;

% Display the sampled function
figure(1),clf,subplot(1,2,1),hold on
plot(Xstar,Ystar,'--')
plot(X,f,'ro','markersize',10,'linewidth',2)

% Display the mean and n-sigma band around the mean:
% At each query point in Xstar, the predictive distribution gives us back
% the mean and variance
plot(Xstar,mu,'b-','linewidth',2)
plot(Xstar,mu+sqrt(diag(Sigma)),'r-','linewidth',1)
plot(Xstar,mu-sqrt(diag(Sigma)),'r-','linewidth',1)

%% Suppose, instead, that we have some noisy measurements of real data
% i.e., y = f(x) + noise, where "noise" is i.i.d. Gaussian with variance
% sigma_n^2.  All we need to do is add I*sigma_n^2 to k_x_x and follow
% through the same math as above.
sigma_n = .05;

% Note that now, with some measurement noise, we can have two conflicting
% measurements at the same input location without having Sigma be singular!
X = [-1;0;1;1];
y = [.2;.6;-.5;-.6];

% Query data points
Xstar = linspace(-5,5,n)';

% Kernel matrices for combinations of training and query inputs
k_x_x = kernel(X,X,kw);
k_xstar_x = kernel(Xstar,X,kw);
k_x_xstar = kernel(X,Xstar,kw);
k_xstar_xstar = kernel(Xstar,Xstar,kw);

% Reusing the covariance functions from above...
mu = k_xstar_x / (k_x_x + eye(numel(X))*sigma_n^2) * y;
Sigma = k_xstar_xstar - k_xstar_x / (k_x_x + eye(numel(X))*sigma_n^2) * k_x_xstar;

% Draw a sample from the predictive distribution
R = chol(Sigma + 1e-6*eye(n));
Ystar = mu' + randn(1,n)*R;

% Display the sampled function
subplot(1,2,2),hold on
plot(Xstar,Ystar,'--')
plot(X,y,'ro','markersize',10,'linewidth',2)

% Display the mean and n-sigma band around the mean:
% At each query point in Xstar, the predictive distribution gives us back
% the mean and variance
plot(Xstar,mu,'b-','linewidth',2)
plot(Xstar,mu+sqrt(diag(Sigma)),'r-','linewidth',1)
plot(Xstar,mu-sqrt(diag(Sigma)),'r-','linewidth',1)

end

% Each row of x0 is an observation, each column is a separate variable
function k = kernel(x0,x1,kernel_width)
k = exp(-(1/2)*pdist2(x0,x1).^2/kernel_width^2);
end