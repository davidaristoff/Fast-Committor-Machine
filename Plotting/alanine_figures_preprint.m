Ns =  [1000; 1439; 2070; 2977; 4282; 6159; 8859; 12743; 
      18330; 26367; 37927; 54556; 78476; 112884; 162378; 
      233573; 335982; 483294; 695193; 1000000];

%set plotting preferences
close all; 
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',22);
set(groot,'DefaultTextFontSize',22);

%% figure 1(a)

%define potential energy and plot it
U = @(x,y) 3*exp(-x.^2-(y-1/3).^2) ...
        -3*exp(-x.^2-(y-5/3).^2) ...
        -5*exp(-(x-1).^2-y.^2) ...
        -5*exp(-(x+1).^2-y.^2) ...
        +0.2*x.^4 ...
        +0.2*(y-1/3).^4;

figure('position',[20 20 350 300]);
fcontour(U,'levellist',-20:.1:0); colorbar;
axis([-2 2 -1.5 2.5]); title('$V_0$')

%% figure 1(b)

%load largest sample size data
load FFM_committor_toy_iter_5_N1000000_cols1000_h1.mat

%plot solution
figure('position',[20 20 350 300]);
scatter(Xref(:,1),Xref(:,2),10,qref,'filled');
axis([-2 2 -1.5 2.5]);
clim([-.1 1.1]); colorbar; title('reference committor')

%% figure 4(a)

%load data
load FFM_committor_toy_iter_1_N1000000_cols1000_h1.mat

%plot scaling matrix
iter = 1;
figure('position',[20 20 350 300]);
imagesc(M);  title(['\mbox{{\boldmath $M$}$^{1/2}$}, iteration ',num2str(iter)]);
colorbar; pause(.1);

%% figure 4(b)

%load data
load FFM_committor_toy_iter_5_N1000000_cols1000_h1.mat

%plot scaling matrix
iter = 5;
figure('position',[20 20 350 300]);
imagesc(M);  title(['\mbox{{\boldmath $M$}$^{1/2}$}, iteration ',num2str(iter)]);
colorbar;

%% figure 5(a)
%plot alanine free energy

load free_energy.mat
load alanine_reference.mat
load alanine_data_combined.mat
figure('position',[20 20 350 300]); 
contourf(grd(1:end/2),grd(1:end/2),-(1/(kT))*log(H(1:end/2,1:end/2)'),...
    'levellist',-30:0.5:-15); colorbar;
xlabel('$\phi$'); ylabel('$\psi$'); title('free energy');
hold on;
S = alphaShape(anglesref(:,1),anglesref(:,2));
tricontour(S.alphaTriangulation,anglesref(:,1),anglesref(:,2),qtest,1);
colorbar; clim([-28 -14.5]);

%% plot figure 5(b)

%plot estimated committor
centerA = [-82 62]*pi/180; 
centerB = [62 -45]*pi/180;
r = 10*pi/180;
figure('position',[20 20 350 300]);
scatter(anglesref(:,1),anglesref(:,2),10,qref,'filled'); colorbar; 
hold on; viscircles([centerA' centerB'],[r r]);
xlabel('$\phi$'); ylabel('$\psi$'); title('reference committor');
axis([-pi pi -pi pi])

%% plot figure 7(a)

%plot Mahalanobis matrix at iteration 5
load FFM_committor_alanine_iter_5_N1000000_cols1000_h1.mat

figure('position',[20 20 350 300]); 
imagesc(M); colorbar;
[~,lam] = eig(M,'vector'); lam = sort(lam,'descend');
title('\mbox{{\boldmath $M$}$^{1/2}$}, 5th iteration')

%% plot figure 7(b)

%get 3d representation of committor
[V,lam] = eig(M,'vector'); [lam,ind] = sort(lam,'descend');
V = V(:,ind);
Xref2d = Xref*V(:,1:2).*lam(1:2).';

%plot committor in 2d
figure('position',[20 20 350 300]); 
scatter(Xref2d(:,1),Xref2d(:,2),10,qtest,'filled');
xlabel('eigenvector 1'); ylabel('eigenvector 2');
title('committor');
hold on;
S = alphaShape(Xref2d(:,1),Xref2d(:,2));
tricontour(S.alphaTriangulation,Xref2d(:,1),Xref2d(:,2),qtest,1);
colorbar;