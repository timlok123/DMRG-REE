% mainDMRG_MPO
% ------------------------ 
% Script file for initializing the Hamiltonian of a 1D spin chain as an MPO
% before passing to the DMRG routine.
%
% by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 21/1/2019

%% Set simulation options
chi = 64;    % maximum bond dimension
Nsites = 8; % number of lattice sites 

% Define the setting of DMRG 
OPTS.numsweeps = 10; % number of DMRG sweeps
OPTS.display = 1;   % level of output display
OPTS.updateon = 1;  % update MPS tensors
OPTS.maxit = 10;     % iterations of Lanczos method
OPTS.krydim = 10;    % dimension of Krylov subspace

%% Define Hamiltonian MPO (transverse field Ising model)
% H = - J \sum Sz_{i} Sz_{i+1} - h \sum Sx_{i}

chid = 2; % Physical dimension
J = 1;    % Coupling strength
h = 1;    % Transverse field strength

sX = [0, 1; 1, 0];
sZ = [1, 0; 0, -1];
sI = eye(2);

% Prepare the M in between
M = cell(1, Nsites); 

% The first matrix 
M_first = zeros(5, 5, 2, 2); % roll, col 
M_first(1, 1, :, :) = sI;    
M_first(1, 2, :, :) = -J* sZ; 
M_first(1, 3, :, :) = -h* sX;
M_first(1, 4, :, :) = -J* sZ;
M_first(4, 4, :, :) = sI;
M_first(2, 5, :, :) = sZ;
M_first(3, 5, :, :) = sI;
M_first(5, 5, :, :) = sI;

% The last matrix 
M_last = zeros(5, 5, 2, 2); % roll, col 
M_last(1, 1, :, :) = sI;    
M_last(1, 2, :, :) = -J* sZ; 
M_last(1, 3, :, :) = -h* sX;
M_last(4, 4, :, :) = sZ;
M_last(2, 5, :, :) = sZ;
M_last(3, 5, :, :) = sI;
M_last(5, 5, :, :) = sI;

% Handle the M^[2] to M^[N-1]
M_template = zeros(5, 5, 2, 2); % roll, col 
M_template(1, 1, :, :) = sI;    
M_template(1, 2, :, :) = -J* sZ; 
M_template(1, 3, :, :) = -h* sX;
M_template(4, 4, :, :) = sI;
M_template(2, 5, :, :) = sZ;
M_template(3, 5, :, :) = sI;
M_template(5, 5, :, :) = sI;

% Checking M template initialization 
% for i = 1:5
%    for j = 1:5
%       disp(reshape(M_template(i, j, :, :), [2,2]));
%    end
% end

M{1} = M_first; 
for i = 2:Nsites-1
    M{i} = M_template;
end
M{Nsites} = M_last; 

% Define left and right boundary tensors
ML = reshape([1; 0; 0; 0; 0], [5, 1, 1]); % Left boundary tensor
MR = reshape([0; 0; 1; 1; 1], [5, 1, 1]); % Right boundary tensor 


%% Initialize MPS tensors
A = cell(1, Nsites); 
A{1} = rand(1,chid,min(chi,chid));
for k = 2:Nsites
    A{k} = rand(size(A{k-1},3),chid,min(min(chi,size(A{k-1},3)*chid),chid^(Nsites-k)));
end

%% Do DMRG sweeps 
[A,sWeight,B,Ekeep] = doDMRG_MPO(A,ML,M,MR,chi,OPTS);

%%%% Show the ground state energy 
E_gs = Ekeep(end);
disp(['Normalized ground State Energy (64): ', num2str(E_gs/(Nsites))]);
normalized_ground_state_energy = E_gs/(Nsites); 

%%%% Showing the convergence of ground state energy 
% figure;
% semilogy(abs(Ekeep - min(Ekeep)), 'LineWidth', 1.5); % Energy error for chi = 32
% xlabel('Update Step');
% ylabel('Energy Error');
% legend(['\chi = ', num2str(chi)]);
% title('DMRG Energy Convergence');
% grid on;


%% Show the Renyi Entanglement Entropy 
REE_array=zeros(1,Nsites+1); 
normalized_lA_array=(0:Nsites)/Nsites;

for lA = 1:Nsites+1
    REE_array(lA) = -log(trace(sWeight{lA}.^4));
end

% figure;
% plot(normalized_lA_array, REE_array, '-o');
% xlabel('Normalized l_A');
% ylabel('REE');
% title('REE vs Normalized l_A');
% grid on;

%% Save the data for manipulation for further manipulation 
save(sprintf('DMRG_data_%d.mat', Nsites));

%% Load the data 
REE_array_64 = load("DMRG_data_64.mat").REE_array;


%% Plot the figure (after loading)
Nsites = 64;
normalized_lA_array=(0:Nsites)/Nsites;

figure;
plot(normalized_lA_array, REE_array, '-o');
xlabel('Normalized l_A');
ylabel('REE');
title('REE vs Normalized l_A');
grid on;







