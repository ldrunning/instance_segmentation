function DistProb = getProb(this, E)
Dist = zeros(size(E, 1), 1);
for i=1:size(E, 1)
    Dist(i) = sqrt((this(1) - E(i, 1))^2 + (this(2) - E(i, 2))^2);
    if Dist(i) == 0
        Dist(i) = Dist(i) + 0.1;
    end
end
% Dist
% hold on
% plot(this(:, 1), this(:, 2), 'bo')

DistProb = min(Dist);