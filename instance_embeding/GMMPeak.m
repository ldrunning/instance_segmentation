function newobj = GMMPeak(obj)
N = obj.NumComponents;
S = zeros([1, N]);
for i = 1:N
    S(i) = (obj.Sigma(1, 1, i) * obj.Sigma(2, 2, i));
end
[~, Seq1] = sort(S);
[~, Seq2] = sort(Seq1);
Smu = zeros([N, 2]);
SSigma = zeros([2, 2, N]);
for i = 1:N
    Smu(i, :) = obj.mu(Seq2(i), :);
    SSigma(:, :, i) = obj.Sigma(:, :, Seq2(i));
end
D1 = squareform(pdist(Smu));
D1 = triu(D1, 1);
D1 = D1 + eye(N);
% Ang represented by costheta ^ 2
Ang = zeros(N);
V = zeros([2, 2, N]);
W = V;
for i = 1:N
    %[V(:, :, i), W(:, :, i)] = eig(inv(SSigma(:, :, i)));
    [V(:, :, i), W(:, :, i)] = eig(SSigma(:, :, i));
end
%W = sqrt(W).^(-1);
for i = 1:N
    for j = 1:N
        Ang(i, j) = (dot(Smu(i, :) - Smu(j, :), V(:, 1, i)) / norm(Smu(i, :) - Smu(j, :))) ^ 2;
    end
end
% dist is expansion from i to j, non-symmetric, represented by polar coordinates of ellipse in
% which main direction determins Angle
dist = zeros(N); 
for i = 1:N
    for j = 1:N
        e = sqrt(abs(W(1, 1, i) ^ 2 - W(2, 2, i) ^ 2)) / max(W(1, 1, i), W(2, 2, i));
        if W(1, 1, i) > W(2, 2, i)
            dist(i, j) = W(2, 2, i) / sqrt(1 - e ^ 2 * Ang(i, j));
        else
            dist(i, j) = W(1, 1, i) / sqrt(1 - e ^ 2 * (1 - Ang(i, j)));
        end
        %dist(i, j) = W(1, 1, i) * Ang(i, j) + W(2, 2, i) - W(2, 2, i) * Ang(i, j);
    end
end
D2 = D1;
for i = 1:N-1
    for j = i+1:N
        %D2(i, j) = max(dist(i, j), dist(j, i)) - min(dist(i, j), dist(j, i))/2;
        D2(i, j) = max(dist(i, j), dist(j, i)) - 0.5 * min(dist(i, j), dist(j, i));
    end
end
dele = [1 1];
while ~isempty(dele)
    dele = [];
    Delta = D1 - D2;
    for i = 1:N-1
        for j = i+1:N
           if Delta(i, j) < 0 
              dele = [dele; j]; 
           end
        end
        if ~isempty(dele)
           D1(dele, :) = 0;
           D1(:, dele) = 0;
           D2(dele, :) = 0;
           D2(:, dele) = 0;
           break
        end
    end
end
dele = [];
for i = 1:N
    if ~(any(D1(i, :)+D2(i, :)) || any(D1(:, i)+D2(:, i)))
        dele = [dele; i];
    end
end
Mu = Smu;
Sigma = SSigma;
Mu(dele, :) = [];
Sigma(:, :, dele) = [];
newobj = gmdistribution(Mu, Sigma);
close all
%%
% [x, y] = find(D < dist);
% dele = [];
% for i = 1:length(x)
%     if x(i) < y(i)
%         if Map(Peak(x(i), 1), Peak(x(i), 2)) > Map(Peak(y(i), 1), Peak(y(i), 2))
%             dele = [dele; y(i)];
%         else
%             dele = [dele; x(i)];
%         end
%     end
% end
% dele = unique(dele);
% Peak(dele, :) = [];
% close all
