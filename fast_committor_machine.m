%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Fast Committor Machine Matlab implementation %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% choose parameters for FFM

Ns = input('vector of training set sizes = ');
r = input('number of columns = ');
gam = input('regularization parameter = ');
epsil = input('bandwidth = ');
iters = input('number of iterations = ');
repeats = input('number of repeats = ');

% %here are the parameters that we used:
% Ns = floor(10.^linspace(3,6,10));
% r = 1000; 800; 600; 400; 200;
% gam = 10^(-6);
% h = 1;
% iters = 5;
% repeats = 10;

%% define fixed parameters for FFM

%define block size for RPCholesky
T = floor(r/10);

%define machine tolerance
tol = 10^(-10);

%% load full dataset

%load reference, input/output pairs, weights, and indicators of B and I
load committor_data.mat Xref qref X_ Y_ w_ ... 
                        XinB_ YinB_ XinI_ YinI_

%get data dimension
[~,d] = size(X_);

%the data arrays are named as follows:
%Xref = reference sample points (not used during training)
%qref = reference committor values (not used during training)
%X_ = training sample inputs
%Y_ = training sample outputs
%w_ = training sample (square roots of) weights
%XinB_ = vector of membership of X in B (1 = member, 0 = not a member)
%YinB_ = vector of membership of Y in B (1 = member, 0 = not a member)
%XinI_ = vector of membership of X in I (1 = member, 0 = not a member)
%YinI_ = vector of membership of Y in I (1 = member, 0 = not a member)
%N = number of samples
%d = dimension of model
%M = initial (square root of) scaling matrix

%notes: 
%B is the target state, I is the interior (complement of A and B).
%Membership vectors are logical format (1 = "true", 0 = "false").
%Matrix M represents the square root of the scaling matrix.
%weights w represent the square roots of the importance weights.

%% define data matrix

FCM_data = zeros(iters,2,length(Ns),repeats);

%notes: second dimension gives mean squared errors and run times

%% begin simulation

for training_set = 1:length(Ns)

%define sample size and number of columns for RPCholesky
N = Ns(training_set);

%define number of samples for Mahalanobis matrix estimation
samples = floor(5*sqrt(N));

%display sample size
disp(['sample size ',num2str(N)]);

for repeat = 1:repeats   %repeat the experiments
    
    %display repeat count
    disp(['repeat ',num2str(repeat)]);

    %sample data
    ind = randsample(10^6,N,'false');
    X = X_(ind,:); Y = Y_(ind,:); w = w_(ind,:);
    XinB = XinB_(ind); YinB = YinB_(ind); 
    XinI = XinI_(ind); YinI = YinI_(ind);

    %define initial (square root of) scaling matrix
    M = eye(d);

    %run FCM for sample size N
    for iter = 1:iters
    
        tic
    
        %update bandwidth (absorbed into M) using sample variance
        M = M/sqrt(sum(var(X*M)));
    
        %get Mahalanobis kernel function and its derivative
        [k,dk] = get_kernel_functions(M,epsil);
    
        %define terms for linear system
        [K,diagK,b] = define_system(X,Y,k,epsil,XinI,YinI,XinB,YinB,w);
    
        %sample the columns using RPCholesky
        S = sample_columns(K,diagK,r,N,T,tol);
    
        %get the committor function by solving the reduced linear system
        [q,dq] = get_committor(X,Y,w,k,dk,K,S,b,XinI,YinI,N,gam,tol);
    
        %update the Mahalanobis matrix
        M = get_scaling_matrix(X,dq,N,d,samples);
        
        %evaluate committor at test points
        qtest = q(Xref);
    
        %compute iteration runtime
        runtime = toc;    

        %update data and save to workspace
        FCM_data(iter,1,training_set,repeat) = ... 
            norm(qref-qtest)^2/length(qref);
        FCM_data(iter,2,training_set,repeat) = runtime;
    
    end
end

%save data from simulation
save(['FCM_toy_Ns',num2str(length(Ns)), ...
    '_r',num2str(r), ...
    '_gam',num2str(gam), ...
    '_epsil',num2str(epsil), ...
    '_iters',num2str(iters), ...
    '_repeats',num2str(repeats)], ...
    "Ns","r","gam","epsil","iters","repeats","FCM_data");

end

%% end simulation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define kernel function and its derivative

function [k,dk] = get_kernel_functions(M,epsil)

%define kernel function (X = matrix, Y = matrix)
k = @(X,Y) exp(-pdist2(X*M,Y*M)/epsil);

%define derivative of kernel function (x = row vector, Y = matrix)
dk = @(x,Y) (1/epsil)*exp(-pdist2(x*M,Y*M)/epsil) ...
           .*((x-Y)*M^2 ... 
           ./(vecnorm((x-Y)*M,2,2) + (vecnorm(x-Y,2,2)==0)))';
%the last term here is included so that dk(x,x) = 0

end

%% define linear system ingredients

function [K,diagK,b] = define_system(X,Y,k,epsil,XinI,YinI,XinB,YinB,w)

%function for pulling a column of kernel matrix A
K = @(S) w.*(XinI.*k(X,X(S,:)).*XinI(S)' ...
           -XinI.*k(X,Y(S,:)).*YinI(S)' ...
           -YinI.*k(Y,X(S,:)).*XinI(S)' ...
           +YinI.*k(Y,Y(S,:)).*YinI(S)').*(w(S)');

%form diagonal of A
diagK = w.*(XinI ...
        -XinI.*exp(-vecnorm(X-Y,2,2)/epsil).*YinI ...
        -YinI.*exp(-vecnorm(Y-X,2,2)/epsil).*XinI ... 
        +YinI);

%form boundary vector
b = w.*(YinB-XinB);

end

%% column-sampling function (RPCholesky)

function S = sample_columns(K,diagK,r,N,T,tol)

%initialize factorized Nystrom approximation
F = zeros(N,r); d = diagK; S = zeros(r,1);

%select column set, S
col = 1;   %current total number of columns
for i=1:r/T
    S_ = randsample(N,T,true,d);       %sample columns according to d
    S_ = unique(S_); l = length(S_);   %get only unique samples
    S(col:col+l-1) = S_;               %update column list
    G = K(S_);                         %form matrix of selected columns
    G = G - F*F(S_,:)';                %remove effect from previous columns
    R = chol(G(S_,:) + tol*trace(G(S_,:))*eye(l));
    GRinv = G/R; F(:,col:col+l-1) = GRinv;   %update approximation
    d = max(0,d - vecnorm(GRinv,2,2).^2);    %update residual diagonal
    d(S_) = 0;                               %prevent double sampling
    col = col + l;                           %update current column
end

%trim away extra columns
S = S(S~=0);

end

%% get committor and its derivative

function [q,dq] = get_committor(X,Y,w,k,dk,K,S,b,XinI,YinI,N,gam,tol)

%create nugget and select columns
I = speye(N); K_S = K(S);

%add nugget to selected columns
K_S = K_S + tol*trace(K_S(S,:))*I(:,S); 

%solve for coefficients
eta = (K_S'*K_S + gam*N*K_S(S,:))\(K_S'*b);

%reweight coefficients by multiplying by (square roots of) weights
theta = eta.*w(S);

%construct committor (Z = matrix)
q = @(Z) k(Z,X(S,:))*(XinI(S).*theta) ...
            -k(Z,Y(S,:))*(YinI(S).*theta);

%construct gradient of committor (z = row vector)
dq = @(z) dk(z,X(S,:))*(XinI(S).*theta) ...
            -dk(z,Y(S,:))*(YinI(S).*theta);

end

%% get scaling matrix

function M = get_scaling_matrix(X,dq,N,d,samples)

%get samples
ind = randi(N,[samples 1]);

%initialize scaling matrix
M = zeros(d,d);

%compute gradient outerproduct
for n = 1:length(ind)
    z = X(ind(n),:); M = M + dq(z)*dq(z)';
end

%get square root
M = real(sqrtm(real(M)));

end
