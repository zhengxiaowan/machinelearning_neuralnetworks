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

X = [ones(m, 1) X]; 

a1=X*Theta1'; 

a2=sigmoid(a1);

a2=[ones(m, 1) a2];

a3=a2*Theta2';

h1=sigmoid(a3);

h2=1.-h1;

for i=1:m

Y=zeros(num_labels,1);
number=y(i);
Y(number)=1;
h1new=h1(i,:);
h2new=h2(i,:);
J=J+(1.0/m).*(-log(h1new)*Y-log(h2new)*(1.-Y));

end

Theta1_new=Theta1(:,2:input_layer_size+1);
Theta2_new=Theta2(:,2:hidden_layer_size+1);

firstproduct=Theta1_new.^2;
secondproduct=Theta2_new.^2;

Jtheta=(lambda/(2*m))*(sum(firstproduct(:))+sum(secondproduct(:)));

J=J+Jtheta;


%
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

for t=1:m

a1=X(t,:);
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(1, 1), a2];
z3=a2*Theta2';
a3=sigmoid(z3);

Y=zeros(1,num_labels);
number=y(t);
Y(number)=1;

delta_3=a3.-Y;

sigmoidgradient=a2.*(1.-a2);

delta_2=(delta_3*Theta2).*sigmoidgradient;

delta_2=delta_2(2:end);

Theta1_grad=Theta1_grad+delta_2'*a1;
Theta2_grad=Theta2_grad+delta_3'*a2;

t=t+1;

end

Theta1_grad=Theta1_grad ./m;
Theta2_grad=Theta2_grad ./m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_reg=lambda.*Theta1 ./m;
Theta2_reg=lambda.*Theta2 ./m;

Theta1_reg=Theta1_grad(:,2:end)+Theta1_reg(:,2:end);
Theta2_reg=Theta2_grad(:,2:end)+Theta2_reg(:,2:end);

Theta1_grad=[Theta1_grad(:,1),Theta1_reg];

Theta2_grad=[Theta2_grad(:,1),Theta2_reg];













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
