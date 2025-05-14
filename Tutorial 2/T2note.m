% %% SVD 
% 
% d1 = 10; d2 = 6;
% A = rand(d1,d2);
% [U,S,V] = svd(A,'econ');
% % check result
% Af = U*S*V';
% dA = norm(Af(:)-A(:));
% 
% 
% %% Spectral decomposition / Eigen-decomposition
% d = 10; C = rand(d);
% H = 0.5*(A+A'); %random Hermitian

%%%%% Ex2.4(a): truncated svd
d = 10; A = rand(d,d,d,d,d);
[Um,S,Vm] = svd(reshape(A,[d^3,d^2]),'econ');
U = reshape(Um,[d,d,d,d^2]);
V = reshape(Vm,[d,d,d^2]);
% form approximation
chi = 100;
Vtilda = V(:,:,1:chi);
Stilda = S(1:chi,1:chi);
Utilda = U(:,:,:,1:chi);
B = ncon({Utilda,Stilda,Vtilda},{[-1,-2,-3,1],[1,2],[-4,-5,2]});
% compare
epsAB = norm(A(:)-B(:)) / norm(A(:));
% disp(epsAB);

%%%%% Ex2.4(d): effective rank
% Generate toeplitz matrix
d = 500;
A = zeros(d,d);
A(1:d-1,2:d) = A(1:d-1,2:d)+eye(d-1);
A(2:d,1:d-1) = A(2:d,1:d-1)+eye(d-1);
A = A / norm(A(:)); %normalize

% compute effective rank to accuracy 'deltaval'
deltaval = 1e-2; 
S = svd(A);                    % In descending order 
r_delta = sum(S > deltaval);   % Counting how many of the values greater than delta value 
eps_err = sqrt(sum(S(r_delta+1:end).^2)); 

a = [5,4,3,2,1];
b = sum(a>=3);
disp(b);
disp(sum(a(b:end)));