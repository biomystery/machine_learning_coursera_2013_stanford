function [all_theta] = onevsall(x, y, num_labels, lambda)
%onevsall trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = onevsall(x, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% some useful variables
m = size(x, 1);
n = size(x, 2);

% you need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% add ones to the x data matrix
x = [ones(m, 1) x];

% ====================== your code here ======================
% instructions: you should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.
%
% hint: theta(:) will return a column vector.
%
% hint: you can use y == c to obtain a vector of 1's and 0's that tell use
%       whether the ground truth is true/false for this class.
%
% note: for this assignment, we recommend using fmincg to optimize the cost
%       function. it is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% example code for fmincg:
for c = 1:num_labels
    % set initial theta
    initial_theta = zeros(n + 1, 1);
    
    % set options for fminunc
    options = optimset('gradobj', 'on', 'maxiter', 50);
    
    % run fmincg to obtain the optimal theta
    % this function will return theta and the cost
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, x, (y == c), lambda)), ...
        initial_theta, options);
    all_theta(c,:) = theta;
end
% =========================================================================


end
