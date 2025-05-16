%% a) Original norm 

d = 12;
A = zeros(d,d,d);

for ni = 1:d
    for nj = 1:d 
        for nk = 1:d
            A(ni, nj, nk) = A(ni, nj, nk) + sqrt(ni + 2*nj + 3* nk);
        end
    end
end 

B = A; C = A; % make the copy via assignment already 

TensorArray={A,B,C};
IndexArray={[-1,-2,1], [1,-3,2], [2,-4,-5]};
ContOrder=[1,2];
H = ncon(TensorArray,IndexArray,ContOrder);

H_norm = norm(H(:)); 

%% b) QR decomposition on C 

Cm = reshape(C, [d,d^2]); % reshape to 2D for SVD 
[Um, Sm, Vm] = svd(Cm,'econ');

chi = 2;
Um_trunicated = Um(:,1:chi); 
Sm_trunicated = Sm(1:chi, 1:chi);
Vm_trunicated = Vm(:, 1:chi); 

% Prepare sqrt(S) for Cl and CR
% Sm_trunicated_sqrt = diag(sqrt(diag(Sm_trunicated)));
Sm_trunicated_sqrt = sqrt(Sm_trunicated);

% CL and CR 
CL = Um_trunicated * Sm_trunicated_sqrt; 
CR = Sm_trunicated_sqrt*(Vm_trunicated');
CR = reshape(CR, [2,d,d]);

H1 = ncon({A,B,CL,CR}, {[-1,-2,1],[1,-3,2],[2,3],[3,-4,-5]}, [1,2,3]);

err_1 = norm(H(:) - H1(:)) / H_norm; 

%% c) Pulling through 

[A_Q,A_R] = qr(reshape(A, [d^2,d]),"econ"); 
A_Q = reshape(A_Q,[d,d,d]);
B_prime = ncon({A_R, B}, {[-1,1], [1,-2,-3]});

[B_prime_Q,B_prime_R] = qr(reshape(B_prime, [d^2,d]),"econ"); 
B_prime_Q = reshape(B_prime_Q,[d,d,d]);
C_prime = ncon({B_prime_R, C}, {[-1,1], [1,-2,-3]});

H_prime = ncon({A_Q, B_prime_Q,C_prime}, {[-1,-2,1],[1,-3,2], [2,-4,-5]});

err_2 = norm(H(:) - H_prime(:)) / H_norm;

%% d) Trunicating C_prime and check the error 

[C_prime_um, C_prime_sm, C_prime_vm] = svd(reshape(C_prime,[d,d^2]),'econ');

chi = 2;
C_prime_um_trunicated = C_prime_um(:,1:chi); 
C_prime_sm_trunicated = C_prime_sm(1:chi, 1:chi);
C_prime_vm = C_prime_vm(:, 1:chi); 

C_prime_sm_trunicated_sqrt = sqrt(C_prime_sm_trunicated);

% C_prime_L and C_prime_R 
C_prime_L = C_prime_um_trunicated* C_prime_sm_trunicated_sqrt; 
C_prime_R = C_prime_sm_trunicated_sqrt*(C_prime_vm');
C_prime_R = reshape(C_prime_R, [2,d,d]);

H1_prime = ncon({A_Q, B_prime_Q,C_prime_L, C_prime_R}, {[-1,-2,1],[1,-3,2], [2,3],[3,-4,-5]});

err_3 = norm(H(:) - H1_prime(:)) / H_norm;