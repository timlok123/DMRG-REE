% mainDMRG_MPO
% ------------------------ 
% Script file for initializing the Hamiltonian of a 1D spin chain as an MPO
% before passing to the DMRG routine.
%
% by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 21/1/2019

%%%%% Example 1: XX model %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Set simulation options
chi = 16;    % maximum bond dimension
Nsites = 64; % number of lattice sites

% Define the setting of DMRG 
OPTS.numsweeps = 10; % number of DMRG sweeps
OPTS.display = 2;   % level of output display
OPTS.updateon = 1;  % update MPS tensors
OPTS.maxit = 20;     % iterations of Lanczos method
OPTS.krydim = 10;    % dimension of Krylov subspace

%%%% Define Hamiltonian MPO (quantum XX model)
% i.e. Writing the Hamitonian in MPO form => contract and it will recover 
% chid = 2;
% sP = sqrt(2)*[0,0;1,0]; % S^+
% sM = sqrt(2)*[0,1;0,0]; % S^-
% sX = [0,1;1,0]; 
% sY = [0,-1i;1i,0]; 
% sZ = [1,0;0,-1]; 
% sI = eye(2);
% 
% M = zeros(4,4,2,2);
% M(1,1,:,:) = sI; M(4,4,:,:) = sI;
% M(1,2,:,:) = sM; M(2,4,:,:) = sP;
% M(1,3,:,:) = sP; M(3,4,:,:) = sM;
% 
% ML = reshape([1;0;0;0],[4,1,1]); %left MPO boundary
% MR = reshape([0;0;0;1],[4,1,1]); %right MPO boundary

%%%% Define Hamiltonian MPO (transverse field Ising model)
% H = - J \sum Sz_{i} Sz_{i+1} - h \sum Sx_{i}

chid = 2; % Physical dimension
J = 1;    % Coupling strength
h = 1;    % Transverse field strength

sX = [0, 1; 1, 0];
sZ = [1, 0; 0, -1];
sI = eye(2);

% Define bulk MPO tensor
M = zeros(4, 4, 2, 2);
M(1, 1, :, :) = sI; 
M(1, 2, :, :) = -J* sZ; 
M(1, 3, :, :) = -h* sX; 
M(2, 4, :, :) = sZ;         
M(3, 4, :, :) = sI;    
M(4, 4, :, :) = sI;         

% Define left and right boundary tensors
ML = reshape([1; 0; 0; 0], [4, 1, 1]); % Left boundary tensor
MR = reshape([0; 0; 1; 1], [4, 1, 1]); % Right boundary tensor 


%%%% Initialize MPS tensors
A = {};
A{1} = rand(1,chid,min(chi,chid));
for k = 2:Nsites
    A{k} = rand(size(A{k-1},3),chid,min(min(chi,size(A{k-1},3)*chid),chid^(Nsites-k)));
end

%%%% Do DMRG sweeps 
[A,~,~,Ekeep1] = doDMRG_MPO(A,ML,M,MR,chi,OPTS);

%%%% Do DMRG sweeps 
chi=64;
[A,~,~,Ekeep2] = doDMRG_MPO(A,ML,M,MR,chi,OPTS);

%%%% Show the ground state energy 
E_gs = Ekeep1(end);
disp(['Normalized ground State Energy (16): ', num2str(E_gs/Nsites)]);

E_gs = Ekeep2(end);
disp(['Normalized ground State Energy (64): ', num2str(E_gs/Nsites)]);
normalized_ground_state_energy = E_gs/Nsites; 

%%%% Showing the convergence of ground state energy 

figure;
hold off;
semilogy(abs(Ekeep1 - min(Ekeep1)), 'LineWidth', 1.5); % Energy error for chi = initial bond dimension
hold on;
semilogy(abs(Ekeep2 - min(Ekeep2)), 'LineWidth', 1.5); % Energy error for chi = 32
xlabel('Update Step');
ylabel('Energy Error');
legend(['\chi = ', num2str(min(chi))], ['\chi = ', num2str(chi)]);
title('DMRG Energy Convergence');
grid on;

