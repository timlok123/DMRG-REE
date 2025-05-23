%% b) Create normalized tensor A 

d1=10;d2=8;
A = zeros(d1,d1,d2,d2);
for ni = 1:d1
    for nj = 1:d1
        for nk = 1:d2
            for nl = 1:d2
                A(ni,nj,nk,nl) = sqrt(ni + 2*nj + 3*nk + 4*nl); 
            end
        end
    end
end

% A_norm1 = sqrt(ncon({A, conj(A)}, {[1:ndims(A)], [1:ndims(A)]})); % A^T = conj(A)
A_norm = norm(A(:));
A_normalized = A / A_norm; 

%% c) Performing SVD on normalized A 

A_matrix = reshape(A_normalized, [d1^2,d2^2]); % reshape to matrix 
[U,S,V] = svd(A_matrix,'econ');                % A = USV' 
% Sum_of_square_diagonal = sqrt(trace(S*conj(S))); 
Sum_of_square_diagonal = sqrt(sum(diag(S).^2));


%% d) effective rank 
delta_val = 1e-4;
r_delta = sum(diag(S) > delta_val); % S is sorted in descending order 

%% e) trunciation error from diagonal S
err_0 = sqrt(sum(diag(S(r_delta+1:end,r_delta+1:end)).^2));

%% f) Reconstruct the tensor and compare 
U_selected = U(:, 1:r_delta); 
V_selected = V(:, 1:r_delta); 
S_selected = S(1:r_delta, 1:r_delta);
B = U_selected * S_selected * V_selected';
B = reshape(B,[d1,d1,d2,d2]);
err_1 = sqrt(sum((A_normalized(:) - B(:)).^2));

disp(err_0);
disp(err_1);