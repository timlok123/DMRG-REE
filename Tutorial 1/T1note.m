%% 1.1 Basic tensor initialization 

RandomTensor = rand(2,3); % (no of rows, no of cols)
ComplexTensor = rand(2,3,4) + 1i*rand(2,3,4); % (no of rows, no of cols, no of slices)

%% 1.2a) Permute and reshape 
APermute = permute(RandomTensor, [2,1]); 
CPermute = permute(ComplexTensor, [2,1,3]);

CReshape = reshape(ComplexTensor, [2, 3*4]); 

%% 1.3 Binary tensor contractions 
% This section reveals the need of permutation & reshape 

d=4;
A=rand(d,d,d,d); B=rand(d,d,d,d);

Aper=permute(A, [1,3,2,4]); Bper=permute(B, [1,4,2,3]);

AReshape=reshape(Aper,[d^2, d^2]); 
BReshape=reshape(Bper,[d^2, d^2]);

CNew=AReshape*BReshape; % MATLAB use vectorization behind the scene 
C = reshape(CNew, [d,d,d,d]); 
