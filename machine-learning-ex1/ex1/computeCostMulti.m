function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% hypothesis = X * theta (m x 3) (3 x 1) = (m x 1)
hypothesis = X * theta; % (m x 1)

% hypothesis - y = (m x 1) - (m x 1) = (m x 1)
squareError = (hypothesis - y).^2;

J = (1/(2 * m)) * sum(squareError);

% =========================================================================

end
