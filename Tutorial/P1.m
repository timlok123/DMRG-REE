%% Problem 1b) 

d=20;
A=rand(d,d,d);B=rand(d,d,d);C=rand(d,d,d);

% ncon routine 
TensorArray={A,B,C};
IndexArray={[1,-2,2], [-1,1,3], [3,2,-3]};
ContOrder=[1,2,3];
D = ncon(TensorArray,IndexArray,ContOrder);