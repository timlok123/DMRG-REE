% DMRG for Quantum Ising Model with Periodic Boundary Conditions
% Calculates ground state and Renyi entanglement entropy (2nd order)

clear all;
close all;

%% Parameters
N = 20;                  % System size
d = 2;                   % Local Hilbert space dimension (spin-1/2)
maxD = 20;               % Maximum bond dimension
maxiter = 20;            % Maximum DMRG iterations
tol = 1e-6;              % Convergence tolerance
h = 1.0;                 % Transverse field strength
J = 1.0;                 % Ising coupling strength

% Renyi entropy parameters
renyi_order = 2;         % 2nd Renyi entropy
measure_points = 1:N/2;  % Points to measure entanglement entropy

%% Define Pauli matrices
sX = [0 1; 1 0];
sZ = [1 0; 0 -1];
I = eye(2);

%% Construct MPO for Ising model with periodic boundary conditions
% Bulk MPO tensor
% W = zeros(4,4,d,d);
% W(1,1,:,:) = I;
% W(4,4,:,:) = I;
% W(1,2,:,:) = sZ;
% W(2,4,:,:) = J*sZ;
% W(1,3,:,:) = h*sX;
% W(3,4,:,:) = I;
% 
% % For periodic BC, we need to connect the MPO boundary
% W(1,4,:,:) = W(1,4,:,:) + J*sZ;  % Add the periodic coupling
% 
% % Left and right boundary tensors
% WL = zeros(1,4,d,d);
% WL(1,1,:,:) = I;
% WL(1,4,:,:) = I;  % For periodic BC
% 
% WR = zeros(4,1,d,d);
% WR(1,1,:,:) = I;
% WR(4,1,:,:) = I;  % For periodic BC

W = zeros(4,4,d,d);
W(1,1,:,:) = I;
W(4,4,:,:) = I;
W(1,2,:,:) = sZ;
W(2,4,:,:) = J*sZ;
W(1,3,:,:) = h*sX;
W(3,4,:,:) = I;

% For periodic BC, we need to connect the MPO boundary
% Note: This should actually be handled by the boundary tensors, not here
% W(1,4,:,:) = W(1,4,:,:) + J*sZ;  % This line should be removed

% Left and right boundary tensors
WL = zeros(1,4,d,d);
WL(1,1,:,:) = I;
% WL(1,4,:,:) = I;  % This should not be here for proper periodic BC
% Instead, the right boundary should connect to the left boundary

WR = zeros(4,1,d,d);
% WR(1,1,:,:) = I;  % This should not be here for proper periodic BC
WR(4,1,:,:) = I;  % This connects the "right" virtual index to the left
WR(2,1,:,:) = sZ; % This adds the periodic coupling term


%% Initialize MPS
A = cell(1,N);
A{1} = rand(1,d,min(maxD,d));
for k = 2:N
    A{k} = rand(size(A{k-1},3),d,min(min(maxD,size(A{k-1},3)*d),d^(N-k)));
end

%% DMRG options
OPTS.numsweeps = 4;     % Number of DMRG sweeps
OPTS.display = 1;       % Level of output display
OPTS.updateon = 1;      % Update MPS tensors
OPTS.maxit = 2;         % Iterations of Lanczos method
OPTS.krydim = 4;        % Dimension of Krylov subspace

%% Run DMRG
[A, sWeight, ~, Ekeep] = doDMRG_MPO_Ising(A, WL, W, WR, maxD, OPTS);

%% Calculate Renyi entanglement entropy
S_renyi = zeros(1, length(measure_points));

for k = 1:length(measure_points)
    cut_pos = measure_points(k);
    
    % Construct reduced density matrix for subsystem A (sites 1 to cut_pos)
    if cut_pos == N
        rhoA = 1;  % Full system has no entanglement
    else
        % Contract MPS up to cut_pos
        psiA = A{1};
        for site = 2:cut_pos
            psiA = ncon({psiA, A{site}}, {[1 -1 -2], [1 -3 -4]});
            psiA = reshape(psiA, size(psiA,1)*size(psiA,2), size(psiA,3), size(psiA,4));
        end
        
        % Contract with conjugate
        psiA_conj = conj(psiA);
        rhoA = ncon({psiA, psiA_conj}, {[1 2 -1], [1 2 -2]});
        
        % Trace over the remaining sites
        for site = cut_pos+1:N
            rhoA = ncon({rhoA, A{site}, conj(A{site})}, {[1 2], [1 3 -1], [2 3 -2]});
        end
    end
    
    % Calculate Renyi entropy
    if cut_pos < N
        rhoA = rhoA/trace(rhoA);  % Normalize
        S_renyi(k) = -log(trace(rhoA^renyi_order))/(renyi_order-1);
    else
        S_renyi(k) = 0;
    end
end

%% Plot results
figure;
plot(measure_points, S_renyi, '-o', 'LineWidth', 2);
xlabel('Subsystem size (L_A)');
ylabel(['Renyi-' num2str(renyi_order) ' Entropy']);
title(['Quantum Ising Model (N=' num2str(N) ', h=' num2str(h) ', J=' num2str(J) ')']);
grid on;

%% Modified DMRG function for Ising model
function [A,sWeight,B,Ekeep] = doDMRG_MPO_Ising(A,ML,M,MR,chimax,OPTS)
% Modified DMRG function for Ising model with periodic boundary conditions

%%%%% set options to defaults if not specified 
if ~isfield(OPTS,'numsweeps')
    OPTS.numsweeps = 10;
end
if ~isfield(OPTS,'display')
    OPTS.display = 1;
end
if ~isfield(OPTS,'updateon')
    OPTS.updateon = 1;
end
if ~isfield(OPTS,'maxit')
    OPTS.maxit = 2;
end
if ~isfield(OPTS,'krydim')
    OPTS.krydim = 4;
end
if OPTS.krydim < 2
    OPTS.krydim = 2;
end

%%%%% left-to-right 'warmup', put MPS in right orthogonal form
Nsites = length(A);
L{1} = ML; R{Nsites} = MR;
chid = size(M,3); 
for p = 1:Nsites - 1
    chil = size(A{p},1); chir = size(A{p},3);
    [qtemp,rtemp] = qr(reshape(A{p},[chil*chid,chir]),0);
    A{p} = reshape(qtemp,[chil,chid,chir]);
    A{p+1} = ncon({rtemp,A{p+1}},{[-1,1],[1,-2,-3]})/norm(rtemp(:));
    L{p+1} = ncon({L{p},M,A{p},conj(A{p})},{[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]});
end
chil = size(A{Nsites},1); chir = size(A{Nsites},3);
[qtemp,stemp] = qr(reshape(A{Nsites},[chil*chid,chir]),0);
A{Nsites} = reshape(qtemp,[chil,chid,chir]);
sWeight{Nsites+1} = stemp./sqrt(trace(stemp*stemp'));

Ekeep = [];
for k = 1:OPTS.numsweeps+1
    %%%%% final sweep is only for orthogonalization (disable updates)
    if k == OPTS.numsweeps+1
        OPTS.updateon = 0;
        OPTS.display = 0;
    end
    
    %%%%%% Optimization sweep: right-to-left 
    for p = Nsites-1:-1:1
        
        %%%%% two-site update
        chil = size(A{p},1); chir = size(A{p+1},3);
        psiGround = reshape(ncon({A{p},A{p+1},sWeight{p+2}},{[-1,-2,1],[1,-3,2],[2,-4]}),[chil*chid^2*chir,1]);
        if OPTS.updateon 
            [psiGround,Ekeep(end+1)] = eigLanczos(psiGround,OPTS,@doApplyMPO,{L{p},M,M,R{p+1}});
        end
        [utemp,stemp,vtemp] = svd(reshape(psiGround,[chil*chid,chid*chir]),'econ');
        chitemp = min(min(size(stemp)),chimax);
        A{p} = reshape(utemp(:,1:chitemp),[chil,chid,chitemp]);
        sWeight{p+1} = stemp(1:chitemp,1:chitemp)./sqrt(sum(diag(stemp(1:chitemp,1:chitemp)).^2));
        B{p+1} = reshape(vtemp(:,1:chitemp)',[chitemp,chid,chir]);
            
        %%%%% new block Hamiltonian MPO
        R{p} = ncon({M,R{p+1},B{p+1},conj(B{p+1})},{[-1,2,3,5],[2,1,4],[-3,5,4],[-2,3,1]});
        
       %%%%% display energy
        if OPTS.display == 2
            fprintf('Sweep: %2.1d of %2.1d, Loc: %2.1d, Energy: %12.12d\n',k,OPTS.numsweeps,p,Ekeep(end));
        end 
    end
    
    %%%%%% left boundary tensor
    chil = size(A{1},1); chir = size(A{1},3);
    [utemp,stemp,vtemp] = svd(reshape(ncon({A{1},sWeight{2}},{[-1,-2,1],[1,-3]}),[chil,chid*chir]),'econ');
    B{1} = reshape(vtemp',[chil,chid,chir]);
    sWeight{1} = utemp*stemp./sqrt(trace(stemp.^2));
    
    %%%%%% Optimization sweep: left-to-right
    for p = 1:Nsites-1
        
        %%%%% two-site update
        chil = size(B{p},1); chir = size(B{p+1},3);
        psiGround = reshape(ncon({sWeight{p},B{p},B{p+1}},{[-1,1],[1,-2,2],[2,-3,-4]}),[chil*chid^2*chir,1]);
        if OPTS.updateon 
            [psiGround,Ekeep(end+1)] = eigLanczos(psiGround,OPTS,@doApplyMPO,{L{p},M,M,R{p+1}});
        end
        [utemp,stemp,vtemp] = svd(reshape(psiGround,[chil*chid,chid*chir]),'econ');
        chitemp = min(min(size(stemp)),chimax);
        A{p} = reshape(utemp(:,1:chitemp),[chil,chid,chitemp]);
        sWeight{p+1} = stemp(1:chitemp,1:chitemp)./sqrt(sum(diag(stemp(1:chitemp,1:chitemp)).^2));
        B{p+1} = reshape(vtemp(:,1:chitemp)',[chitemp,chid,chir]);
            
        %%%%% new block Hamiltonian
        L{p+1} = ncon({L{p},M,A{p},conj(A{p})},{[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]});
        
        %%%%% display energy
        if OPTS.display == 2
            fprintf('Sweep: %2.1d of %2.1d, Loc: %2.1d, Energy: %12.12d\n',k,OPTS.numsweeps,p,Ekeep(end));
        end
    end
    
    %%%%%% right boundary tensor
    chil = size(B{Nsites},1); chir = size(B{Nsites},3);
    [utemp,stemp,vtemp] = svd(reshape(ncon({B{Nsites},sWeight{Nsites}},{[1,-2,-3],[-1,1]}),[chil*chid,chir,1]),'econ');    
    A{Nsites} = reshape(utemp,[chil,chid,chir]);
    sWeight{Nsites+1} = (stemp./sqrt(sum(diag(stemp).^2)))*vtemp';
    
    if OPTS.display == 1
        fprintf('Sweep: %2.1d of %2.1d, Energy: %12.12d, Bond dim: %2.0d\n',k,OPTS.numsweeps,Ekeep(end),chimax);
    end
end
A{Nsites} = ncon({A{Nsites},sWeight{Nsites+1}},{[-1,-2,1],[1,-3]});
sWeight{Nsites+1} = eye(size(A{Nsites},3));
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function psi = doApplyMPO(psi,L,M1,M2,R)
% applies the superblock MPO to the state

psi = reshape(ncon({reshape(psi,[size(L,3),size(M1,4),size(M2,4),size(R,3)]),L,M1,M2,R},...
    {[1,3,5,7],[2,-1,1],[2,4,-2,3],[4,6,-3,5],[6,-4,7]}),[size(L,3)*size(M1,4)*size(M2,4)*size(R,3),1]);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [psivec,dval] = eigLanczos(psivec,OPTS,linFunct,functArgs)
% function for computing the smallest algebraic eigenvalue and eigenvector
% of the linear function 'linFunct' using a Lanczos method. Maximum
% iterations are specified by 'OPTS.maxit' and the dimension of Krylov
% space is specified by 'OPTS.krydim'. Input 'functArgs' is an array of
% optional arguments passed to 'linFunct'.

if norm(psivec) == 0
    psivec = rand(length(psivec),1);
end
psi = zeros(numel(psivec),OPTS.krydim+1);
A = zeros(OPTS.krydim,OPTS.krydim);
for k = 1:OPTS.maxit
    
    psi(:,1) = psivec(:)/norm(psivec);
    for p = 2:OPTS.krydim+1
        psi(:,p) = linFunct(psi(:,p-1),functArgs{(1:length(functArgs))});
        for g = 1:1:p-1
            A(p-1,g) = dot(psi(:,p),psi(:,g));
            A(g,p-1) = conj(A(p-1,g));
        end
        for g = 1:1:p-1
            psi(:,p) = psi(:,p) - dot(psi(:,g),psi(:,p))*psi(:,g);
            psi(:,p) = psi(:,p)/max(norm(psi(:,p)),1e-16);
        end
    end
    
    [utemp,dtemp] = eig(0.5*(A+A'));
    xloc = find(diag(dtemp) == min(diag(dtemp)));
    psivec = psi(:,1:OPTS.krydim)*utemp(:,xloc(1));
end
psivec = psivec/norm(psivec);
dval = dtemp(xloc(1),xloc(1));
end