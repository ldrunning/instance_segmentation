function Peak = LocalPeak(Map, Area, CandiArea, dist)
[m n] = size(Map);
Peak = [];
getDist = dist;
D  = squareform(pdist(CandiArea));
D = triu(D, 1);
dist = zeros(size(CandiArea, 1));
for j=1:size(CandiArea, 1)
    for k=1:size(CandiArea, 1)
        if k > j
            %dist(j, k) = min(Map(CandiArea(j, 1), CandiArea(j, 2)), Map(CandiArea(k, 1), CandiArea(k, 2)));
            %dist(j, k) = 1/(1/Map(CandiArea(j, 1), CandiArea(j, 2))+1/Map(CandiArea(k, 1), CandiArea(k, 2)));
            %dist(j, k) = (Map(CandiArea(j, 1), CandiArea(j, 2)) + Map(CandiArea(k, 1), CandiArea(k, 2))) * 0.5;
            %dist(j, k) = max(Map(CandiArea(j, 1), CandiArea(j, 2)), Map(CandiArea(k, 1), CandiArea(k, 2)));
            dist(j, k) = sqrt(Map(CandiArea(j, 1), CandiArea(j, 2)) ^ 2 + Map(CandiArea(k, 1), CandiArea(k, 2)) ^ 2) ;
           
        end
    end
end
dist = triu(dist, 1);


%dist(dist ~= 0) = getDist;


[x, y] = find((D - dist) < 0);
dele = [];
for i = 1:length(x)
    if x(i) < y(i)
        if Map(CandiArea(x(i), 1), CandiArea(x(i), 2)) > Map(CandiArea(y(i), 1), CandiArea(y(i), 2))
            dele = [dele; y(i)];
        else
            dele = [dele; x(i)];
        end
    end
end
dele = unique(dele);
CandiArea(dele, :) = [];
Peak = CandiArea;
close all

