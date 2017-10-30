% plot behavior

scores        = {'NEOFAC_O','NEOFAC_C','NEOFAC_E','NEOFAC_A_corr','NEOFAC_N','Alpha','Beta'};
confounds     = {'Gender','Age_in_Yrs','FS_BrainSeg_Vol','FDsum_REST1','FDsum_REST2','PMAT24_A_CR','fMRI_3T_ReconVrs'};

% load variables of interest
csvFile   = fullfile('df.csv');
% read headers to know how many columns
fid = fopen(csvFile,'r');
tmp = fgetl(fid);
fclose(fid);
headers = strsplit(tmp,',');
% read entire file
fid = fopen(csvFile,'r');
C = textscan(fid,repmat('%s',1,length(headers)),'delimiter',',','headerlines',1);
fclose(fid);
subjects      = C{strcmp(headers,'Subject')};
for iScore = 1:length(scores)
    eval(sprintf('%s = cellfun(@str2double,C{strcmp(headers,''%s'')});',scores{iScore},scores{iScore}));
end
for iCon = 1:length(confounds)
    eval(sprintf('%s = cellfun(@str2double,C{strcmp(headers,''%s'')});',confounds{iCon},confounds{iCon}));
end

% distributions of the scores
figure;
for iScore = 1:length(scores)
    score = scores{iScore};
    subplot(1,length(scores),iScore);hold on;
    ylabel(strrep(score,'_','-'));
    if iScore<=5
        xs = 0:48;
    else
        xs = linspace(-4,4,49);
    end
    N = hist(eval(score),xs);
    barh(xs,N,'EdgeColor','k','FaceColor','b','FaceAlpha',.2);
    if iScore<=5
        axis([0 max(N) 0 48]);
    else
        axis([0 max(N) -4 4]);
    end
end

nSub = length(NEOFAC_O);
% correlations
variables = {'NEOFAC_O','NEOFAC_C','NEOFAC_E','NEOFAC_A_corr','NEOFAC_N','Alpha','Beta',...
    'Gender','Age_in_Yrs','FS_BrainSeg_Vol','FDsum_REST1','FDsum_REST2','PMAT24_A_CR','fMRI_3T_ReconVrs'}; ...
bigMat = [];
for i = 1:length(variables)
    bigMat = [bigMat eval(variables{i})];
end
corrMat  = corr(bigMat);
pcorrMat = partialcorr(bigMat);
tmp = corrMat.*triu(ones(size(corrMat)),1);
% + pcorrMat.*tril(ones(size(pcorrMat)),-1);
figure;imagesc(tmp);axis square;caxis([-.5 .5]);colormap(bluewhitered);
set(gca,'ytick',1:length(variables),'yticklabel',cellfun(@(x) strrep(x,'_','-'),variables,'UniformOutput',false),...
    'xtick',1:length(variables),'xticklabel',cellfun(@(x) strrep(x,'_','-'),variables,'UniformOutput',false),'xticklabelrotation',90);
hold on;
for i = 1:size(tmp,1)
    for j = 1:size(tmp,2)
        t  = tmp(i,j)*sqrt((nSub-2)/(1-tmp(i,j)^2));
        p2 = 2*(1-tcdf(abs(t),nSub-2));
        if p2<0.001
            addStr='^{***}';
        elseif p2<0.01
            addStr='^{**}';
        elseif p2<.05
            addStr='^{*}';
        else
            addStr = '';
        end
        text(j,i,sprintf(['%.2f',addStr],tmp(i,j)),'HorizontalAlignment','center','VerticalAlignment','middle');
    end
end


% plot factor scores on the higher order factors axes
figure;
for iScore = 1:5
    score = scores{iScore};
    subplot(1,5,iScore);hold on;title(strrep(score,'_','-'));
    scatter(Alpha,Beta,[],eval(score),'filled');
    axis([-4 4 -4 4]);axis square;
end
    
keyboard