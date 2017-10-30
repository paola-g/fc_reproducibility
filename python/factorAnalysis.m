% factor analysis or pca on item scores from NEO-FFI

X = csvread('itemScores.csv',1,1);

Y       = csvread('NEOFACscores.csv',1,1);
% read headers
fid     = fopen('NEOFACscores.csv','r');
l       = fgetl(fid);
fclose(fid);
factors = strsplit(l,',');factors=factors(2:end);


factorL = zeros(size(X,2),1);
% openness
factorL([13,43,53,58,28])          =  1;
factorL([23,48,03,08,18,38,33])    = -1;
% conscientiousness
factorL([05,10,25,35,60,20,40,50]) =  2;
factorL([15,30,55,45])             = -2;
% extroversion
factorL([07,37,02,17,22,32,47,52]) =  3;
factorL([12,42,27,57])             = -3;
% agreeableness
factorL([19,04,34,49])             =  4;
factorL([09,14,24,29,44,54,59,39]) = -4;
% neuroticism
factorL([11,06,21,26,36,41,51,56]) =  5;
factorL([01,16,31,46])             = -5;

methods = {'FA'};%'factoran'};%,'FA','pca'

colors = distinguishable_colors(length(factors));

for iMethod = 1:length(methods)
    method = methods{iMethod};
    switch method
        case 'factoran'
            [Lambda,Psi,T,stats,F] = factoran(X,2);
        case 'FA' % factor analysis with varimax rotation
            [B,Lambda,var,F,E] = FA(X,2);
            fprintf('\n\n\n\n');
            fid = fopen('higherOrder.csv','w');
            for i =1:size(F,1)
                fprintf(fid,'%.6f,%.6f\n',F(i,1),F(i,2));
            end
            fclose(fid);
        case 'pca' % pca
            [Lambda, F, ~, tsquared, explained] = pca(X);
    end
    % item loading
    figure(1);subplot(1,length(methods),iMethod);hold on;
    switch method
        case {'FA','factoran'}
            axis(.75*[-1 1 -1 1]);axis square
        case 'pca'
            axis(.42*[-1 1 -1 1]);axis square
    end
    for f = 1:length(factors)
        scatter(Lambda(abs(factorL)==f,1),Lambda(abs(factorL)==f,2),20,colors(f,:),'filled'); 
    end
    if iMethod==1,legend(cellfun(@(x) strrep(x,'_','-'),factors','UniformOutput',false));end
    for f = 1:length(factors)
        scatter(...
            mean(Lambda(abs(factorL)==f,1)),...
            mean(Lambda(abs(factorL)==f,2)),...
            100,colors(f,:),'filled');
    end
    title(method);
    line([-1 1],[0 0],'Color','k','LineStyle','--');
    line([0 0],[-1 1],'Color','k','LineStyle','--');
    
    % subjects
    figure(2);
    for f = 1:length(factors)
        subplot(length(methods),length(factors),(iMethod-1)*length(factors)+f);hold on;
        if iMethod == 1,title(strrep(factors{f},'_','-'));end
        if f == 1,ylabel(method);end
        scatter(F(:,1),F(:,2),[],Y(:,f),'filled');
        switch method
            case {'FA','factoran'}
                axis(5*[-1 1 -1 1]);
            case 'pca'
                axis(10*[-1 1 -1 1]);
        end
        axis square;
        
    end
end

return
for f = 1:length(factors)
    figure;
    subplot(3,3,[2 3 5 6]);hold on;title(strrep(factors{f},'_','-'));
    scatter(F(:,1),F(:,2),[],Y(:,f),'filled');
    axis([-5 5 -5 5]);axis square

    % look at distribution of each factor on the new 2d projection
    binCs = linspace(-5,5,20);
    binEs = [(binCs - diff(binCs(1:2))) binCs(end)+diff(binCs(1:2))];
    xBins = discretize(F(:,1),binEs);
    yBins = discretize(F(:,2),binEs);
    xMean = zeros(1,length(binCs));xSte = zeros(1,length(binCs));
    yMean = zeros(1,length(binCs));ySte = zeros(1,length(binCs));
    for i = 1:length(binCs)
        if sum(xBins==i)>0
            xMean(i) = mean(Y(xBins==i,f));
            xSte(i)  = std(Y(xBins==i,f))/sqrt(sum(xBins==i));
        end
        if sum(yBins==i)>0
            yMean(i) = mean(Y(yBins==i,f));
            ySte(i)  = std(Y(yBins==i,f))/sqrt(sum(yBins==i));
        end
    end
    
    subplot(3,3,[8 9]);
    bar(binCs,xMean,'FaceColor','none','EdgeColor','k');
    xlim([-5 5]);
    
    subplot(3,3,[1 4]);
    barh(binCs,yMean,'FaceColor','none','EdgeColor','k');
    ylim([-5 5]);
end
