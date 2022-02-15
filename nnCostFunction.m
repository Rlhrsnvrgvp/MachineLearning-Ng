function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%% Extend All Elements Version
%{
I = eye(num_labels);
for i = 1:m
  %Y_output = zeros(num_labels, num_labels);
  %Y_output(y(i), y(i)) = 1;
  y_output = I(y(i),:);                        % 10*1
  a1 = [ones(m,1) X](i,:);                    % 1*401
  z2 = a1 * Theta1';                          % (1*401) * (401*25) => 1*25
  a2 = [ones(size(z2, 1),1) sigmoid(z2)];     % 1*26
  z3 = a2 * Theta2';                          % (1*26) * (26*10) => 1*10
  a3 = sigmoid(z3);                           % 1*10
  h = a3;                                     % 1*10
  for k = 1:num_labels
    y_k = y_output(k);
    J = J - log(h(k))*y_k  - log(1 - h(k))*(1 - y_k);
  endfor
endfor
J = J / m
%}

%% Matrix Version
a1 = [ones(m,1) X];                        % 5000*401
z2 = a1 * Theta1';                         % (5000*401) * (401*25) = 5000*25
a2 = [ones(size(z2, 1),1) sigmoid(z2)];    % 5000*26
z3 = a2 * Theta2';                         % (5000*26) * (26*10) = 5000*10
a3 = sigmoid(z3);                          % 5000*10
h = a3;                                    % 5000*10

I = eye(num_labels);                       % 10*10
Y = zeros(m, num_labels);                  % 5000*10
for i = 1:m
  Y(i,:) = I(y(i), :);                     % add value to each sample
endfor
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m;

%% J with Regularization
J = J + (lambda /(2*m)) * (sum(sum(Theta1(:,2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta3 = a3 - Y;                                                           % 5000*10
delta2 = (delta3 * Theta2) .* sigmoidGradient([zeros(size(z2, 1),1) z2]);  % (5000*10) * (10*26) .* (5000*26) = 5000*26
delta2 = delta2(:, 2:end);                                                 % 5000*25

Delta_1 = zeros(size(Theta1));                % 25*401
Delta_2 = zeros(size(Theta2));                % 10*26
Delta_1 = Delta_1 + delta2' * a1;             % 25*401 + (25*5000) * (5000*401) = 25*401
Delta_2 = Delta_2 + delta3' * a2;             % 10*26 + (10*5000) * (5000*26) = 10*26

Theta1_grad = Delta_1 / m;
Theta2_grad = Delta_2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Gradients with Regularization
%{
Theta1_grad = Delta_1 / m + (lambda / m) * Theta1;
Theta2_grad = Delta_2 / m + (lambda / m) * Theta2;
Theta1_grad(:, 1) = Delta_1(:, 1) / m;
Theta2_grad(:, 1) = Delta_2(:, 1) / m;
%}

%{
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;
Theta1_grad(:, 1) = Delta_1(:, 1) / m;
Theta2_grad(:, 1) = Delta_2(:, 1) / m;
%}

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
