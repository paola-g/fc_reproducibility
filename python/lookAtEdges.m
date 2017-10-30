% look at weights
% try this in Python
% https://martinos.org/mne/stable/auto_examples/connectivity/plot_mne_inverse_label_connectivity.html
dbstop if error

ROOTDIR = 'Z:\LabUsers\duboisjx';
WBCOMMAND = '!"C:\\Program Files\\workbench\\bin_windows64\\wb_command"';
% ROOTDIR = '/home/duboisjx/mnt10/LabUsers/duboisjx';

parcellations = {'Glasser'};%'shen2013',
predAlgs      = {'elnet'};%'Finn',
preprocs      = {'Finn','Ciric6','SiegelB'};
sessions      = {'REST1','REST2'};

release       = 'all+MEG2';
thr           = 0.01;
SMs           = {'PMAT24_A_CR','Beta'};%'NEOFAC_O','NEOFAC_C','NEOFAC_E','NEOFAC_A_corr','NEOFAC_N','Alpha',
cols          = distinguishable_colors(length(SMs));

figure;
adjMats    = cell(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));
adjFrac    = cell(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));
adjFracAll = cell(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));
adjBinoCDF = cell(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));
rhos    = nan(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));

for iParcel = 1:length(parcellations)
    parcellation = parcellations{iParcel};
    switch parcellation
        case 'shen2013'
            nNodes        = 268;
            indsL         = 135:268;
            indsR         = 1:134;
            networkID     = csvread(fullfile(ROOTDIR,'data','parcellations','shenetal_neuroimage2013_new','shen_268_parcellation_networklabels.csv'),1);
            networkLabels = {'MedialFrontal','Frontoparietal','DefaultMode','SubcortCerebellum','Motor','VisualI','VisualII','VisualAssoc'};
        case 'Glasser'
            nNodes     = 360;
            indsL      = 1:180;
            indsR      = 181:360;
            % match to shen
            load('distGlasserShen.mat'); % this comes from a script that matches the centroids of the parcels in MNI space
            [mini,indMini]   = min(D,[],2);
            % indMini is the corresponding shen parcels
            % [sorted,indSort] = sort(indMini,'ascend');
            networkID_    = csvread(fullfile(ROOTDIR,'data','parcellations','shenetal_neuroimage2013_new','shen_268_parcellation_networklabels.csv'),1);
            networkLabels = {'MedialFrontal','Frontoparietal','DefaultMode','SubcortCerebellum','Motor','VisualI','VisualII','VisualAssoc'};
            networkID  = [(1:nNodes)' zeros(nNodes,1)];
            for iNode = 1:length(indMini)
                networkID(iNode,2) = networkID_(indMini(iNode),2);
            end
    end
    nEdges     = sum(sum(triu(ones(nNodes),1)>0));
    
%     %% reorder nodes
%     % order by L- ascending network, R- descending network
    networks   = unique(networkID(:,2))';
%     colors     = distinguishable_colors(length(networks));
%     % left 1,2,3,4,5... right 1,2,3,4,5,...
%     nodeOrderM  = [];
%     networkBdrs = zeros(1,2*length(networks)+1);
%     % Left
%     for iN = 1:length(networks)
%         N = networks(iN);
%         inds = find(networkID(:,2)==N & ismember(networkID(:,1),indsL));
%         nodeOrderM = [nodeOrderM inds'];
%         networkBdrs(1+iN) = networkBdrs(1+iN-1)+length(inds);
%     end
%     % Right
%     for iN = 1: length(networks)
%         N = networks(iN);
%         inds = find(networkID(:,2)==N & ismember(networkID(:,1),indsR));
%         nodeOrderM = [nodeOrderM inds'];
%         networkBdrs(1+length(networks)+iN) = networkBdrs(1+length(networks)+iN-1)+length(inds);
%     end
%     networkBdrs(end) = size(networkID,1);
%     networkLabelPos = conv(networkBdrs,[.5 .5],'same'); networkLabelPos = networkLabelPos(1:end-1);
    for iPredAlg = 1:length(predAlgs)
        predAlg = predAlgs{iPredAlg};
        for iPrep = 1:length(preprocs)
            preproc = preprocs{iPrep};
            inDir = fullfile(ROOTDIR,'data','HCP','MRI','Results','FINAL',preproc,parcellation);
            for iSM = 1:length(SMs)
                SM = SMs{iSM};
                fprintf('%s\n',SM);
                for iSess = 1:length(sessions)
                    session = sessions{iSess};
                    fprintf('\t%s\n',session);
                    
                    %% performance
                    switch SM
                        case 'PMAT24_A_CR'
                            decon = 'decon';
                        otherwise
                            decon = 'decon+IQ';
                    end
                    inDir_ = fullfile(inDir,...
                        sprintf('%s_%s_%s_%s_%s_%s_%s_thr%.2f',...
                        SM,preproc,parcellation,session,decon,predAlg,release,thr),...
                        '0000');
                    % load scores
                    try
                        switch predAlg
                            case 'Finn'
                                res0 = load(fullfile(inDir_,'result.mat'),'rho_pos','rho_neg','rho_posneg','pred_pos');
                                rhos(iSM,iParcel,iPrep,iPredAlg,iSess) = res0.rho_pos;
                                if ~exist('nSub','var')
                                    nSub = length(res0.pred_pos);
                                end
                            otherwise
                                res0 = load(fullfile(inDir_,'result.mat'),'rho','pred');
                                rhos(iSM,iParcel,iPrep,iPredAlg,iSess) = res0.rho;
                                if ~exist('nSub','var')
                                    nSub = length(res0.pred);
                                end
                        end
                    end
                    
                    files = dir(fullfile(inDir_,'*.mat'));
                    files = {files(:).name};
                    files = files(~ismember(files,{'result.mat','_result.mat'}));
                    
                    adjAll     = zeros(nNodes,nNodes,length(files));
                    for i = 1:length(files)
                        adj     = zeros(nNodes);
                        adjTriu = zeros(nEdges,1);
                        res = load(fullfile(inDir_, files{i}));
                        switch predAlg
                            case 'Finn'
                                adjTriu(res.idx_filtered_pos+1) = 1;
                            case 'elnet'
                                adjTriu(res.idx_filtered+1)     = 1; %res.coef';
                        end
                        adj(triu(ones(nNodes),1)>0) = adjTriu;
                        adjM = adj + adj';
%                         adjM = adjM(nodeOrderM,:);
%                         adjM = adjM(:,nodeOrderM);
                        adjAll(:,:,i) = adjM;
                    end
                    tmp            = mean(adjAll,3);
                    adjMats{iSM,iParcel,iPrep,iPredAlg,iSess} = tmp(triu(ones(nNodes),1)>0);
                    
%                     % NETWORK-BASED
%                     adjFrac{iSM,iParcel,iPrep,iPredAlg,iSess}     = zeros((length(networkBdrs)-1),(length(networkBdrs)-1),size(adjAll,3));
%                     adjBinoCDF{iSM,iParcel,iPrep,iPredAlg,iSess}  = zeros((length(networkBdrs)-1),(length(networkBdrs)-1),size(adjAll,3));
%                     adjFracAll{iSM,iParcel,iPrep,iPredAlg,iSess}  = zeros(1,size(adjAll,3));
%                     for i = 1:size(adjAll,3)
%                         adjFracAll{iSM,iParcel,iPrep,iPredAlg,iSess}(i) = mean(mean(adjAll(:,:,i)~=0));
%                         % count proportion of edges in each network x network
%                         for iN = 1:(length(networkBdrs)-1)
%                             for jN = 1:(length(networkBdrs)-1)
%                                 tmp = adjAll((networkBdrs(iN)+1):networkBdrs(iN+1),(networkBdrs(jN)+1):networkBdrs(jN+1),i)~=0;
%                                 if iN==jN
%                                     tmp = tmp(triu(ones(length(tmp)),1)>0);
%                                 else
%                                     tmp = tmp(:);
%                                 end
%                                 adjFrac{iSM,iParcel,iPrep,iPredAlg,iSess}(iN,jN,i)    = mean(tmp~=0);
%                                 adjBinoCDF{iSM,iParcel,iPrep,iPredAlg,iSess}(iN,jN,i) = binocdf(sum(tmp~=0),length(tmp),adjFracAll{iSM,iParcel,iPrep,iPredAlg,iSess}(i));
%                             end
%                         end
%                     end
%                     adjFracMedian    = zeros(nNodes);
%                     adjBinoCDFMedian = zeros(nNodes);
%                     for iN = 1:(length(networkBdrs)-1)
%                         for jN = 1:(length(networkBdrs)-1)
%                             adjFracMedian((networkBdrs(iN)+1):networkBdrs(iN+1),(networkBdrs(jN)+1):networkBdrs(jN+1)) = ...
%                                 median(adjFrac(iN,jN,:),3);
%                             adjBinoCDFMedian((networkBdrs(iN)+1):networkBdrs(iN+1),(networkBdrs(jN)+1):networkBdrs(jN+1)) = ...
%                                 median(adjBinoCDF(iN,jN,:),3);
%                         end
%                     end
                    
%                     %% WHICH EDGES -- use RSN networks
%                     subplot(2,ceil(length(SMs)/2),iSM);
%                     imagesc(adjFracMedian);axis square;
%                     caxis(mean(adjFracAll)+.25*(max(adjFracMedian(:))-min(adjFracMedian(:)))*[-1 1]);colorbar
%                     hold on;
%                     title(strrep(SM,'_','-'));
%                     line(repmat(networkBdrs+.5,2,1),repmat([0.5;nNodes+.5],1,length(networkBdrs)),'Color','k','LineWidth',1);
%                     line(repmat([0.5;nNodes+.5],1,length(networkBdrs)),repmat(networkBdrs+.5,2,1),'Color','k','LineWidth',1);
%                     for iN = 1:length(networks)
%                         % LEFT
%                         line([networkBdrs(iN) networkBdrs(iN+1)]+.5,[0 0]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line([networkBdrs(iN) networkBdrs(iN+1)]+.5,nNodes/2*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line([networkBdrs(iN) networkBdrs(iN+1)]+.5,nNodes*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line([0 0]+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line(nNodes/2*ones(1,2)+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line(nNodes*ones(1,2)+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         % RIGHT
%                         line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,[0 0]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,nNodes/2*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,nNodes*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line([0 0]+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line(nNodes/2*ones(1,2)+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
%                         line(nNodes*ones(1,2)+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
%                     end
%                     for iN = 1:(length(networkBdrs)-1)
%                         for jN = 1:(length(networkBdrs)-1)
%                             if median(adjBinoCDF(iN,jN,:),3)>.975
%                                 plot(networkLabelPos(jN),networkLabelPos(iN),'r*','Markersize',10);
%                             elseif median(adjBinoCDF(iN,jN,:),3)<.025
%                                 plot(networkLabelPos(jN),networkLabelPos(iN),'b*','Markersize',10);
%                             end
%                         end
%                     end
%                     set(gca,...
%                         'xtick',networkLabelPos,'xticklabel',networkLabels,'xticklabelrotation',45,...
%                         'ytick',networkLabelPos,'yticklabel',networkLabels);
                end
            end
        end
    end
end

%% count edges in each network (within and between)
% figure;
% parcellation = 'Glasser';iParcel = find(strcmp(parcellations,parcellation));
% preproc      = 'Ciric6';iPreproc  = find(strcmp(preprocs,preproc));
% predAlg      = 'Finn';iPredAlg  = find(strcmp(predAlgs,predAlg));
% for iSess = 1:length(sessions)
%     for iSM = 1:length(SMs)
%         adj = zeros(nNodes);
%         adj(triu(ones(nNodes),1)>0) = abs(adjMats{iSM,iParcel,iPreproc,iPredAlg,iSess});
%         adjM = adj + adj';
%         % use weights
%         sumWeightsTot  = sum(sum(adjM));
%         sumWeightsTotN = nNodes^2;
%         sumWeightsNet  = zeros(2,length(networks));
%         sumWeightsNetN  = zeros(2,length(networks));
%         for iN = 1:length(networks)
%             for jN = iN:length(networks)
%                 if jN==iN
%                     sumWeightsNet(1,iN)  = sumWeightsNet(1,iN) +  sum(sum(adjM(networkID(:,2)==iN,networkID(:,2)==jN)));
%                     sumWeightsNetN(1,iN) = sumWeightsNetN(1,iN) +  sum(networkID(:,2)==iN)*sum(networkID(:,2)==jN);
%                 else
%                     sumWeightsNet(2,iN)  = sumWeightsNet(2,iN) +  sum(sum(adjM(networkID(:,2)==iN,networkID(:,2)==jN),1));
%                     sumWeightsNetN(2,iN) = sumWeightsNetN(2,iN) +  sum(networkID(:,2)==iN)*sum(networkID(:,2)==jN);
%                 end
%             end
%         end
%         subplot(2,2,(iSM-1)*length(sessions)+iSess);bar(sumWeightsNet./sumWeightsNetN*sumWeightsTotN/sumWeightsTot);
%     end
% end

%% OVERLAP
figure;
parcellation = 'Glasser';iParcel = find(strcmp(parcellations,parcellation));
predAlg      = 'elnet';iPredAlg  = find(strcmp(predAlgs,predAlg));
adj = zeros(length(SMs)*length(preprocs)*length(sessions),nEdges);
for iSM = 1:length(SMs)
    for iSess = 1:length(sessions)
        for iPreproc=1:length(preprocs)
            adj((iSM-1)*length(sessions)*length(preprocs)+(iSess-1)*length(preprocs)+iPreproc,:) = adjMats{iSM,iParcel,iPreproc,iPredAlg,iSess};
        end
    end
end
overlap = zeros(size(adj,1));
for i = 1:size(adj,1)
    for j = 1:size(adj,1)
        overlap(i,j)    = sum(adj(i,:)~=0 & adj(j,:)~=0)...
            /(.5*(sum(adj(i,:)~=0)+sum(adj(j,:)~=0)));
    end
end
figure;imagesc(overlap);axis square;caxis([0 0.5]);colormap(bluewhitered);


% make .ptseries if doesn't exist
if ~exist(fullfile(ROOTDIR,'data','parcellations','example.ptseries.txt'),'file')
    eval(sprintf('%s -cifti-parcellate %s %s COLUMN %s',...
        WBCOMMAND,...
        fullfile(ROOTDIR,'data','parcellations','example.dtseries.nii'),...
        fullfile(ROOTDIR,'data','parcellations','Glasser2016','Parcels.dlabel.nii'),...
        fullfile(ROOTDIR,'data','parcellations','example.ptseries.nii')));
    eval(sprintf('%s -cifti-convert -to-text %s %s',...
        WBCOMMAND,...
        fullfile(ROOTDIR,'data','parcellations','example.ptseries.nii'),...
        fullfile(ROOTDIR,'data','parcellations','example.ptseries.txt')));
end

figure;
adjM = cell(1,length(SMs));
for iSM = 1:length(SMs)
    adj = zeros(nNodes);
    for iPreproc = 1:length(preprocs)
        for iSess = 1:length(sessions)
            if iSess==1 && iPreproc==1
            adj(triu(ones(nNodes),1)>0) = adjMats{iSM,iParcel,iPreproc,iPredAlg,iSess}~=0;
            else
            adj(triu(ones(nNodes),1)>0) = adj(triu(ones(nNodes),1)>0) & adjMats{iSM,iParcel,iPreproc,iPredAlg,iSess}~=0; 
            end
        end
    end
    adjM{iSM} = adj + adj';
    subplot(2,length(SMs),iSM);imagesc(adjM{iSM});axis square;caxis([-1 1]);colormap(bluewhitered);
    subplot(2,length(SMs),length(SMs)+iSM);bar(mean(adjM{iSM},2));
    % make a cifti (node degree map)
    dlmwrite('tmp.txt',100*mean(adjM{iSM},2));
%     keyboard
    % use WB_COMMAND
    eval(sprintf('%s -cifti-convert -from-text %s %s %s',...
        WBCOMMAND,...
        'tmp.txt',...
        fullfile(ROOTDIR,'data','parcellations','example.ptseries.nii'),...
        sprintf('%s_degree.ptseries.nii',SMs{iSM})));
end

sum(adjM{1}(triu(ones(nNodes),1)>0)~=0 & adjM{2}(triu(ones(nNodes),1)>0)~=0)...
    /(.5*(sum(adjM{1}(triu(ones(nNodes),1)>0)~=0)+sum(adjM{2}(triu(ones(nNodes),1)>0)~=0)))

keyboard

% show theoretical chance level as gray area
r      = linspace(-1,1,2000);
t      = r.*sqrt((nSub-2)./(1-r.^2));
p2     = 2*(1-tcdf(abs(t),nSub-2));
thThr  = [r([find(p2>=0.025,1,'first') find(p2>=0.025,1,'last')]);...
    r([find(p2>=0.005,1,'first') find(p2>=0.005,1,'last')]);...
    r([find(p2>=0.0005,1,'first') find(p2>=0.0005,1,'last')])];
figure;
for iSM = 1:length(SMs)
    SM = SMs{iSM};
    subplot(2,length(SMs)/2,iSM);hold on;
    title(sprintf('%s',SM));
    lim = .3;
    axis(lim*[-1 1 -1 1]);
    % 99.9% theoretical threshold
    fill([thThr(3,1) thThr(3,2) thThr(3,2) thThr(3,1)],[thThr(3,1) thThr(3,1) thThr(3,2) thThr(3,2)],'k','FaceColor','k','EdgeColor','none','FaceAlpha',.1);
    line(repmat([-lim lim]',1,2),repmat([thThr(3,1) thThr(3,2)],2,1),'Color','k','LineStyle',':');
    line(repmat([thThr(3,1) thThr(3,2)],2,1),repmat([-lim lim]',1,2),'Color','k','LineStyle',':');
    % 99% theoretical threshold
    fill([thThr(2,1) thThr(2,2) thThr(2,2) thThr(2,1)],[thThr(2,1) thThr(2,1) thThr(2,2) thThr(2,2)],'k','FaceColor','k','EdgeColor','none','FaceAlpha',.1);
    line(repmat([-lim lim]',1,2),repmat([thThr(2,1) thThr(2,2)],2,1),'Color','k','LineStyle','--');
    line(repmat([thThr(2,1) thThr(2,2)],2,1),repmat([-lim lim]',1,2),'Color','k','LineStyle','--');
    % 95% theoretical threshold
    fill([thThr(1,1) thThr(1,2) thThr(1,2) thThr(1,1)],[thThr(1,1) thThr(1,1) thThr(1,2) thThr(1,2)],'k','FaceColor','k','EdgeColor','none','FaceAlpha',.1);
    line(repmat([-lim lim]',1,2),repmat([thThr(1,1) thThr(1,2)],2,1),'Color','k','LineStyle','-');
    line(repmat([thThr(1,1) thThr(1,2)],2,1),repmat([-lim lim]',1,2),'Color','k','LineStyle','-');
    % show diagonal
    line([-lim lim],[-lim lim],'Color','k','LineStyle','--');
    % show data
    % iSM,iParcel,iPrep,iPredAlg,iSess
    inLegend = [];
    legendEntries ={};
    colors   = parula(2);
    symbols  = {'s','^','v'};
    for iParcel = 1:length(parcellations)
        parcellation = parcellations{iParcel};
        for iPrep = 1:length(preprocs)
            preproc = preprocs{iPrep};
            for iPredAlg = 1:length(predAlgs)
                predAlg = predAlgs{iPredAlg};
                switch predAlg
                    case 'Finn'
                        h = scatter(rhos(iSM,iParcel,iPrep,iPredAlg,1),rhos(iSM,iParcel,iPrep,iPredAlg,2),75,cols(iParcel,:),symbols{iPrep},'filled');
                    case 'elnet'
                        h = scatter(rhos(iSM,iParcel,iPrep,iPredAlg,1),rhos(iSM,iParcel,iPrep,iPredAlg,2),75,cols(iParcel,:),symbols{iPrep});
                end
                inLegend      = [inLegend h];
                legendEntries = [legendEntries {sprintf('%s/%s/%s',parcellation,preproc,predAlg)}];
            end
        end
    end
    if iSM==1
        legend(inLegend,legendEntries,'Location','SouthWest');
    end
    xlabel('correlation predicted vs. measured, REST1');
    ylabel('correlation predicted vs. measured, REST2');
    axis square
end

parcellation = 'Glasser';iParcel = find(strcmp(parcellations,parcellation));
preproc      = 'Ciric6';iPreproc  = find(strcmp(preprocs,preproc));
predAlg      = 'elnet';iPredAlg  = find(strcmp(predAlgs,predAlg));
figure;hold on;
subplot(1,2,1);
imagesc(median(adjBinoCDF{1,iParcel,iPreproc,iPredAlg,1},3));axis square;
subplot(1,2,2);
imagesc(median(adjBinoCDF{1,iParcel,iPreproc,iPredAlg,2},3));axis square;


keyboard
% % compute Dice Similarity
% thisThr = .1;
% overlap = nan(length(SMs));
% % cell(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));
% parcellation = 'Glasser';iParcel = find(strcmp(parcellations,parcellation));
% preproc      = 'Ciric6';iPreproc  = find(strcmp(preprocs,preproc));
% predAlg      = 'Finn';iPredAlg  = find(strcmp(predAlgs,predAlg));
% for i = 1:length(SMs)
%     for j = 1:length(SMs)
%         if j>i      % REST2 vs REST2
%             overlap(i,j)    = sum(adjMats{i,iParcel,iPreproc,iPredAlg,2}>=thisThr & adjMats{j,iParcel,iPreproc,iPredAlg,2}>=thisThr)...
%                 /(.5*(sum(adjMats{i,iParcel,iPreproc,iPredAlg,2}>=thisThr)+sum(adjMats{j,iParcel,iPreproc,iPredAlg,2}>=thisThr)));
%         elseif j==i % REST1 vs REST2
%             overlap(i,j)    = sum(adjMats{i,iParcel,iPreproc,iPredAlg,1}>=thisThr & adjMats{j,iParcel,iPreproc,iPredAlg,2}>=thisThr)...
%                 /(.5*(sum(adjMats{i,iParcel,iPreproc,iPredAlg,1}>=thisThr)+sum(adjMats{j,iParcel,iPreproc,iPredAlg,2}>=thisThr)));
%         elseif j<i  % REST1 vs REST1
%             overlap(i,j)    = sum(adjMats{i,iParcel,iPreproc,iPredAlg,1}>=thisThr & adjMats{j,iParcel,iPreproc,iPredAlg,1}>=thisThr)...
%                 /(.5*(sum(adjMats{i,iParcel,iPreproc,iPredAlg,1}>=thisThr)+sum(adjMats{j,iParcel,iPreproc,iPredAlg,1}>=thisThr)));
%         end
%     end
% end
% figure;
% imagesc([overlap.*tril(ones(length(SMs)),-1),...
%     overlap(eye(length(SMs))>0),...
%     overlap.*triu(ones(length(SMs)),1)]);
% axis equal;axis tight;caxis([0 .2])
% set(gca,...
%     'xtick',[1:length(SMs) length(SMs)+1+(1:length(SMs))],...
%     'xticklabel',[cellfun(@(x) strrep(x,'_','-'),SMs,'UniformOutput',false),cellfun(@(x) strrep(x,'_','-'),SMs,'UniformOutput',false)],...
%     'xticklabelrotation',45,...
%     'ytick',1:length(SMs),...
%     'yticklabel',cellfun(@(x) strrep(x,'_','-'),SMs,'UniformOutput',false));
% nSub = length(preds);
% for iSM = 1:length(SMs)
%     t      = rhos_pos(iSM,1).*sqrt((nSub-2)./(1-rhos_pos(iSM,1).^2));p2     = 2*(1-tcdf(abs(t),nSub-2));
%     if p2<=0.0005,sigStr='***';elseif p2<=0.005,sigStr='**';elseif p2<=0.025,sigStr='*';else sigStr='';end
%     text(length(SMs),iSM,sprintf('%.3f^{%s}',rhos_pos(iSM,1),sigStr),...
%         'HorizontalAlignment','center','VerticalAlignment','middle',...
%         'FontSize',10,'Color','w');
%     t      = rhos_pos(iSM,2).*sqrt((nSub-2)./(1-rhos_pos(iSM,2).^2));p2     = 2*(1-tcdf(abs(t),nSub-2));
%     if p2<=0.0005,sigStr='***';elseif p2<=0.005,sigStr='**';elseif p2<=0.025,sigStr='*';else sigStr='';end
%     text(length(SMs)+2,iSM,sprintf('%.3f^{%s}',rhos_pos(iSM,2),sigStr),...
%         'HorizontalAlignment','center','VerticalAlignment','middle',...
%         'FontSize',10,'Color','w');
% end

keyboard

eCTAn = 100*adjTriuAll./repmat(sum(abs(adjTriuAll),1),nEdges,1);
adj(triu(ones(nNodes),1)>0) = mean(eCTAn,2);

%% compute whether average absolute weight of connections
% normalized by average weight in the entire matrix
% THIS COULD BE DONE ON EACH CV SEPARATELY
adjMmean = zeros(size(adjM));
for iN = 1:(length(networkBdrs)-1)
    for jN = 1:(length(networkBdrs)-1)
        adjMmean((networkBdrs(iN)+1):networkBdrs(iN+1),(networkBdrs(jN)+1):networkBdrs(jN+1)) = ...
            mean(mean(abs(adjM((networkBdrs(iN)+1):networkBdrs(iN+1),(networkBdrs(jN)+1):networkBdrs(jN+1)))));
    end
end
adjMmean =  (adjMmean - mean(mean(abs(adjM))))/mean(mean(abs(adjM)));


%% DISPLAY RESULTS IN MATRICES, SHOW NETWORK LABELS
figure;
subplot(1,2,1);
imagesc(adjM);axis square;
caxis([-0.2 0.2]);
line(repmat(networkBdrs+.5,2,1),repmat([0.5;nNodes+.5],1,length(networkBdrs)),'Color','k','LineWidth',1);
line(repmat([0.5;nNodes+.5],1,length(networkBdrs)),repmat(networkBdrs+.5,2,1),'Color','k','LineWidth',1);
for iN = 1:length(networks)
    % LEFT
    line([networkBdrs(iN) networkBdrs(iN+1)]+.5,[0 0]+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN) networkBdrs(iN+1)]+.5,nNodes/2*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN) networkBdrs(iN+1)]+.5,nNodes*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([0 0]+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes/2*ones(1,2)+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes*ones(1,2)+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
    % RIGHT
    line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,[0 0]+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,nNodes/2*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,nNodes*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([0 0]+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes/2*ones(1,2)+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes*ones(1,2)+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
end
colormap(bluewhitered);
set(gca,...
    'xtick',networkLabelPos,'xticklabel',networkLabels,'xticklabelrotation',45,...
    'ytick',networkLabelPos,'yticklabel',networkLabels);


%
subplot(1,2,2);
imagesc(adjMmean);axis square;
caxis([-3 3]);
line(repmat(networkBdrs+.5,2,1),repmat([0.5;nNodes+.5],1,length(networkBdrs)),'Color','k','LineWidth',1);
line(repmat([0.5;nNodes+.5],1,length(networkBdrs)),repmat(networkBdrs+.5,2,1),'Color','k','LineWidth',1);
for iN = 1:length(networks)
    % LEFT
    line([networkBdrs(iN) networkBdrs(iN+1)]+.5,[0 0]+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN) networkBdrs(iN+1)]+.5,nNodes/2*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN) networkBdrs(iN+1)]+.5,nNodes*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([0 0]+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes/2*ones(1,2)+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes*ones(1,2)+.5,[networkBdrs(iN) networkBdrs(iN+1)]+.5,'Color',colors(iN,:),'LineWidth',5);
    % RIGHT
    line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,[0 0]+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,nNodes/2*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,nNodes*ones(1,2)+.5,'Color',colors(iN,:),'LineWidth',5);
    line([0 0]+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes/2*ones(1,2)+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
    line(nNodes*ones(1,2)+.5,[networkBdrs(iN+length(networks)) networkBdrs(iN+1+length(networks))]+.5,'Color',colors(iN,:),'LineWidth',5);
end
set(gca,...
    'xtick',networkLabelPos,'xticklabel',networkLabels,'xticklabelrotation',45,...
    'ytick',networkLabelPos,'yticklabel',networkLabels);


return

% circular order
nodeOrderC  = [];
cmapC       = [];
% Left
for iN = 1:length(networks)
    N = networks(iN);
    inds = find(networkID(:,2)==N & ismember(networkID(:,1),indsL));
    nodeOrderC = [nodeOrderC inds'];
    cmapC      = [cmapC;repmat(colors(iN,:),length(inds),1)];
end
% Right
for iN = length(networks):-1:1
    N = networks(iN);
    inds = find(networkID(:,2)==N & ismember(networkID(:,1),indsR));
    nodeOrderC = [nodeOrderC inds(end:-1:1)'];
    cmapC      = [cmapC;repmat(colors(iN,:),length(inds),1)];
end
% rotate by 1/4 turn
shift      = ceil(nNodes/4);
nodeOrderC = [nodeOrderC((shift+1):end) nodeOrderC(1:shift)];
cmapC      = [cmapC((shift+1):end,:);cmap(1:shift,:)];
edgeCountC = edgeCount(nodeOrderC,:);
edgeCountC = edgeCountC(:,nodeOrderC);





% Call CIRCULARGRAPH with only the adjacency matrix as an argument.
addpath('circularGraph');
% subplot(1,2,2);
figure;circularGraph(double(edgeCount>.995*max(edgeCount(:))),...
    'Colormap',cmap);
