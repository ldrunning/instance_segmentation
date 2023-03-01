clear all
close all
clc
FilePath = 'D:\nuclei seg data\0126\1\color_union';
FileMask = '*.png';
imgDataPath = 'D:\nuclei seg data\0126\1\sem\';
imgDir = dir([imgDataPath  '\*.png']);
%c = 0;
for kkk =1:length(imgDir)                 % 遍历所有图片
    %c= c+1;
    c = 0;
    I = imread([imgDataPath  imgDir(kkk).name]);
    global gmap;
    global sparseflag;
    %%
    tic
%     I = imread(strcat(FilePath, PicList(pic_num).name));
%     I = imread(('C:\Users\admin\Desktop\tupian\0.615734_mass16_3_0.8876_49280_9520_700_700.png'));
    I = imfill(I, 'holes');
    I(:, 1) = 0;
    I(:, end) = 0;
    I(1, :) = 0;
    I(end, :) = 0;
    R1=0;
    R2=0;
    [xxx, yyy] = size(I);
    gmap = zeros(xxx);
    ORI = I;
    I = bwareaopen(I, 40, 8);
    L = bwlabel(I, 8);
    O1 = [];
    O2 = [];
    img = I*255;
    %%
    iii = 0;
    Output = [];
    LLL = zeros(xxx);
    WhMap = zeros(xxx);
    for num = 1:max(max(L))
        toc
        tic
        iii=iii+1;
        [Ex, Ey] = find(L == num);
        E = [Ex, Ey];
        E = sortrows(E, 1);
        
        %%
        EdgePoi = [];
        for i = 1:size(E, 1)
            this = E(i, :);
            count = 0;
            %八邻域
            if (ismember([this(1), this(2)+1], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1)+1, this(2)+1], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1), this(2)-1], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1)-1, this(2)], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1)-1, this(2)+1], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1)+1, this(2)-1], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1)+1, this(2)], E, 'rows'))
                count = count + 1;
            end
            if (ismember([this(1)-1, this(2)-1], E, 'rows'))
                count = count + 1;
            end
            if count ~= 8
                EdgePoi = [EdgePoi; i];
            end
        end
        EdgePoi = E(EdgePoi, :);
        %%
        ProbList = zeros(size(E, 1), 1);
        x= 1:xxx;
        y= 1:yyy;
        ProbMap = zeros(size(x, 2));
        %plot(EdgePoi(:, 1), EdgePoi(:, 2), 'r*')
        for i = 1:size(E, 1)
            ProbList(i, 1) = getProb(E(i, :), EdgePoi);
        end
        for i = 1:size(E, 1)
            ProbMap(E(i, 1), E(i, 2)) = ProbList(i);
        end
        ProbMap=reshape(ProbMap, [size(x, 2), size(y, 2)]);
        [X, Y] = meshgrid(x, y);
        X=X(:);
        Y=Y(:);
        X=reshape(X, [size(x, 2), size(y, 2)]);
        Y=reshape(Y, [size(y, 2), size(x, 2)]);
        Map = ProbMap;
        ProbPeak = max(max(ProbMap));
        ProbMap = ProbMap/ProbPeak*5;
        ProbMap = exp(ProbMap);
        ProbMap(ProbMap == min(min(ProbMap))) = 0;
        ProbSum = sum(sum(ProbMap));
        ProbMap = ProbMap/ProbSum;
        %     figure(10)
        %     plot3(X, Y, ProbMap)
        %     hold off

        %%
        [x, y] = find (ProbMap > 0);
        if length(x) < 10
            continue
        end
        ProbArea = [x, y];
        [Mux, Muy] = find(imregionalmax(ProbMap, 8)~=0);%手写 峰的距离 峰的高度
        Mu0 = LocalPeak(Map, ProbArea, [Mux, Muy], 10);%手写 峰的距离 峰的高度
        figure(1)
        scatter(x,y,10,'.');
        hold on
        scatter(E(:, 1),E(:, 2),10,'k<');
        hold on
        scatter(Mu0(1), Mu0(2), 'o')
        hold off
        O1 = [O1; Mu0];
        if size(Mu0, 1) > 1
            W = ProbMap(x, y);
            W = diag(W);
            X = ProbArea;
            [Center, Sigma0] = WeightedGMMCluster(Map, W, X, 1, Mu0);
        else
            Center = Mu0;
        end
        O2 = [O2; Center];
        %%
        %     if size(Center, 1) == 1
        %         continue
        %     end
        WhMap = WhMap + ProbMap;
        close all
    end

    %%
    BW = zeros(xxx);
    for i=1:size(O2, 1)
        BW(round(O2(i, 1)), round(O2(i, 2))) = 1;
    end
    background = (WhMap == 0);
    WhMap1 = imimposemin(-WhMap, BW);
    imshow(WhMap1)
    LL = double(watershed(WhMap1));
    LL(background) = 0;
    Lrgb = label2rgb(LL, 'jet', 'k', 'shuffle');
    imshow(Lrgb)
    %%
    SavePath = 'D:\nuclei seg data\0126\1\test_inst\';
%     imwrite(Lrgb, strcat(SavePath, erase(PicList(pic_num).name, '.png'), '.png'))
    imwrite(Lrgb,strcat(SavePath,imgDir(kkk).name))
%     imwrite(Lrgb,strcat(SavePath,char(PicList(pic_num).name)))
    
 end
