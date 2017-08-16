function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
regcost = lambda/(2 * m) * (sum(theta .* theta) - (theta(1) * theta(1)));
hx = sigmoid(X * theta);
J = -(1/m) * sum((y .* log(hx)) + ((1 - y) .* log (1 - hx))) + regcost; 

% grad is d(J(theta)) / d(thetaj)
grad(1) = (1/m) * sum((hx - y) .* X(:, 1));
for j = 2:size(theta, 1)
    grad(j) = (1 / m) * sum ((hx - y) .* X(:, j)) + lambda / m * theta(j);
endfor

% =============================================================

end