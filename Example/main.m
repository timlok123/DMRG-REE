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
Nsites = 50; % number of lattice sites

% Define the setting of DMRG 
OPTS.numsweeps = 4; % number of DMRG sweeps
OPTS.display = 2;   % level of output display
OPTS.updateon = 1;  % update MPS tensors
OPTS.maxit = 2;     % iterations of Lanczos method
OPTS.krydim = 4;    % dimension of Krylov subspace

%%%% Define Hamiltonian MPO (quantum XX model)
% i.e. Writing the Hamitonian in MPO form => contract and it will recover 
chid = 2;
sP = sqrt(2)*[0,0;1,0]; % S^+
sM = sqrt(2)*[0,1;0,0]; % S^-
sX = [0,1;1,0]; 
sY = [0,-1i;1i,0]; 
sZ = [1,0;0,-1]; 
sI = eye(2);

M = zeros(4,4,2,2);
M(1,1,:,:) = sI; M(4,4,:,:) = sI;
M(1,2,:,:) = sM; M(2,4,:,:) = sP;
M(1,3,:,:) = sP; M(3,4,:,:) = sM;

ML = reshape([1;0;0;0],[4,1,1]); %left MPO boundary
MR = reshape([0;0;0;1],[4,1,1]); %right MPO boundary

%%%% Initialize MPS tensors
A = {};
A{1} = rand(1,chid,min(chi,chid));
for k = 2:Nsites
    A{k} = rand(size(A{k-1},3),chid,min(min(chi,size(A{k-1},3)*chid),chid^(Nsites-k)));
end

%%%% Do DMRG sweeps 
[A,~,~,Ekeep1] = doDMRG_MPO(A,ML,M,MR,chi,OPTS);

%%%% Increase bond dim and reconverge 
chi = 32;
[A,sWeight,B,Ekeep2] = doDMRG_MPO(A,ML,M,MR,chi,OPTS);

%%%% Compare with exact results (computed from free fermions)
H = zeros(Nsites,Nsites);
H(2:Nsites,1:Nsites-1) = H(2:Nsites,1:Nsites-1) + eye(Nsites-1);
H(1:Nsites-1,2:Nsites) = H(1:Nsites-1,2:Nsites) + eye(Nsites-1);
dtemp = eig(0.5*(H+H'));
EnExact = 2*sum(dtemp(dtemp<0));

figure(1);
hold off
semilogy(abs(Ekeep1-EnExact))
axis([1 2*OPTS.numsweeps*(Nsites-1) 10e-7 10e1])
hold on
semilogy(abs(Ekeep2-EnExact))
xlabel('Update step')
ylabel('Ground energy error')
legend('chi = 16', 'chi = 32')
title(['DMRG for XX model on ' num2str(Nsites) ' sites'])

%%%% Compute 2-site reduced density matrices, local energy density
rhotwo = {};
hamloc = reshape(kron(sP,sM) + kron(sM,sP),[2,2,2,2]);
for k = 1:Nsites-1
    rhotwo{k} = ncon({A{k},conj(A{k}),A{k+1},conj(A{k+1}),sWeight{k+2},sWeight{k+2}},...
        {[1,-3,2],[1,-1,3],[2,-4,4],[3,-2,5],[4,6],[5,6]});
    Enloc(k) = ncon({hamloc,rhotwo{k}},{[1:4],[1:4]});
end