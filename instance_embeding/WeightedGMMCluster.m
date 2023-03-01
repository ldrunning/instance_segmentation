function [Center, Sigma] = WeightedGMMCluster(Map, W, X, mkplt, Mu0)
%程序中Psi对应二维高斯函数，其中的(x-μ)与其转置的顺序与上述提到的高斯函数不同，这样是为了保证矩阵可乘性，不影响结果.
%Gamma 为隐变量的值，Gamma(i,j)代表第i个样本属于第j个模型的概率。
%Mu为期望，Sigma为协方差矩阵%Pi为各模型的权值系数%产生2个二维正态数据
global sparseflag;
Center = Mu0;
scatter(X(:,1),X(:,2),10,'.');
K=size(Mu0, 1);
[N,D]=size(X);
Gamma=zeros(N,K);
Psi=zeros(N,K);
Mu=zeros(K,D);
LM=zeros(K,D);
Sigma =zeros(D, D, K);
Pi=zeros(1,K);
%选择初始椭圆中心作为期望迭代初值
% Mu(1,:)=X(floor(N/4), :);
% Mu(2,:)=X(floor(3*N/4) ,:);
% Mu(3,:)=X(floor(N/2), :);
% Mu(4,:)=X(floor(N), :);
Mu = Mu0;
R1 = 0;
R2 = 0;
%所有数据的协方差作为协方差初值
for k=1:K
    Pi(k)=1/K;
    Sigma(:, :, k)=cov(X)/5;
end

LMu=Mu;
LSigma=Sigma;
LPi=Pi;
i = 1;
hold on
plot(Mu(:, 1), Mu(:, 2), 'r*', 'MarkerSize', 10)
while true
    i = i+1;
    %Estimation Step
    for k = 1:K
        Y = X - repmat(Mu(k,:),N,1);
        Psi(:,k) = (2*pi)^(-D/2)*det(Sigma(:,:,k))^(-1/2)*diag(exp(-1/2*Y/(Sigma(:,:,k))*(Y')));      %Psi每列代表每个高斯分布对所有数据的取值
    end
    Gamma_SUM=zeros(D,D);
    for j = 1:N
        SparPen = min(0.000000002, min(Pi.* Psi(j, :))); %guarantee each component staying positive
        SparPwn = 0;
%         fprintf('SparPen=0')
        for k=1:K
            %Gamma(j,k) = (Pi(1,k)*Psi(j,k)-0.00000000001)/sum(Psi(j,:)*Pi'-0.000000000001*K);                                               %Psi的第一行分别代表两个高斯分布对第一个数据的取值
            %Gamma(j,k) = (Pi(1,k)*Psi(j,k))/sum(Psi(j,:)*Pi')*W(j);
            Gamma(j,k) = (Pi(1,k)*Psi(j,k)-SparPen)/sum(Psi(j,:)*Pi'-SparPen*K);
        end 
    end
    Gamma = diag(W) * Gamma; 
    %Maximization Step
    for k = 1:K
        %update Mu
        Mu_SUM= zeros(1,D);
        for j=1:N
            Mu_SUM=Mu_SUM+Gamma(j,k)*X(j,:);
        end
        Mu(k,:)= Mu_SUM/sum(Gamma(:,k))
        %update Sigma
        Sigma_SUM= zeros(D,D);
        for j = 1:N
            Sigma_SUM = Sigma_SUM+ Gamma(j,k)*(X(j,:)-Mu(k,:))'*(X(j,:)-Mu(k,:));
        end
        %fprintf('Sigma is non sparse')
        Sigma(:,:,k)= Sigma_SUM/sum(Gamma(:,k))
        -min(0.000080, min(diag(Sigma_SUM/sum(Gamma(:,k))))/1.5)*[1, 0; 0, 1]
        %Sigma(:,:,k)= Sigma_SUM/sum(Gamma(:,k))-0.0000055*[1, 0; 0, 1];
        %-0.0000055*[1, 0; 0, 1];%trace
        %0.0000055*Sigma(:,:,k)/(det(Sigma(:,:,k))*sum(Gamma(:,k)));%norm
      
        %update Pi
        Pi_SUM=0;
        for j=1:N
            Pi_SUM=Pi_SUM+Gamma(j,k);
        end
        Pi(1,k)=Pi_SUM/N;
    end
    Pi
    R_Mu=sum(sum(abs(LMu- Mu)));
    R_Sigma=sum(sum(sum(sum(abs(LSigma- Sigma)))));
    R_Pi=sum(sum(abs(LPi- Pi)));
    R1=R2;
    R2=R_Mu+R_Sigma+R_Pi;
    tag = zeros([size(X, 1), 1]);
    for t=1:size(X, 1)
       [~, tag(t)] = max(Gamma(t, :));
    end
    
    figure(1)
    for k=1:K
        PP = X(find(tag == k), :);
        hold on 
        plot(PP(:,1),PP(:,2),'.');
    end
    hold off
    if i > 2000 || isnan(R1)
        break
    end
    if (abs(R1-R2)<1e-7)
        figure
        PP = X(find(tag == 1), :);
        hold on 
        plot(PP(:,1),PP(:,2),'g.');
        PP = X(find(tag == 2), :);
        hold on 
        plot(PP(:,1),PP(:,2),'ro');
        PP = X(find(tag == 3), :);
        hold on 
        plot(PP(:,1),PP(:,2),'kx');
        PP = X(find(tag == 4), :);
        hold on 
        plot(PP(:,1),PP(:,2),'b*');
        disp('期望');
        disp(Mu);
        disp('协方差矩阵');
        disp(Sigma);
        disp('权值系数');
        disp(Pi);
        obj=gmdistribution(Mu,Sigma);
        %figure,h = ezmesh(@(x,y)pdf(obj,[x,y]),[-10 10], [-10 10]);
        for i=1:K
            if (Sigma(1, 1, i) < 0.0001) || (Sigma(2, 2, i) < 0.0001)
                return
            end
        end
        x = 1:1001;
        y = 1:1001;
        m = length(x);
        [XX YY] = meshgrid(x, y);
        XX = XX(:);
        YY = YY(:);
        pp = pdf(obj, [XX, YY]);
        if mkplt
            %figure()
            %plot3(XX, YY, pp);
            figure,h = ezmesh(@(x,y)pdf(obj,[x,y]),[min(X(:, 1)) max(X(:, 1))], [min(X(:, 2)) max(X(:, 2))]);
            
        end
        pp = reshape(pp, [m, m]);
        [Cx, Cy] = find(imregionalmax(pp, 8) ~= 0);
        Center = [Cy, Cx];
        newobj = GMMPeak(obj);
        Center = newobj.mu;
        Sigma = newobj.Sigma;
        break;
    end
    LMu=Mu;
    LSigma=Sigma;
    LPi=Pi;
end
hold off
scatter(X(:,1),X(:,2),10,'.');
hold on
plot(Center(:, 1), Center(:, 2), 'r*')
hold off