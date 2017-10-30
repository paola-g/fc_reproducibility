% look at weights
% try this in Python
% https://martinos.org/mne/stable/auto_examples/connectivity/plot_mne_inverse_label_connectivity.html

parcellations = {'shen2013','Glasser'};
predAlgs      = {'Finn','elnet'};
preprocs      = {'Finn','Ciric6','SiegelB'};
sessions      = {'REST1','REST2'};

release       = 'all+MEG2';
thr           = 0.01;
SMs   = {'PMAT24_A_CR','NEOFAC_O','NEOFAC_C','NEOFAC_E','NEOFAC_A_corr','NEOFAC_N','Alpha','Beta'};
cols  = distinguishable_colors(length(SMs));

figure;

%(iSM,iParcel,iPrep,iPredAlg,iSess)
adjMats = cell(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));
rhos    = nan(length(SMs),length(parcellations),length(preprocs),length(predAlgs),length(sessions));

for iParcel = 1:length(parcellations)
    parcellation = parcellations{iParcel};
    switch parcellation
        case 'shen2013'
            nNodes        = 268;
            indsL         = 135:268;
            indsR         = 1:134;
            networkID     = csvread('Z:\LabUsers\duboisjx\data\parcellations\shenetal_neuroimage2013_new\shen_268_parcellation_networklabels.csv',1);
            networkLabels = {'MedialFrontal','Frontoparietal','DefaultMode','SubcortCerebellum','Motor','VisualI','VisualII','VisualAssoc'};
        case 'Glasser'
            nNodes     = 360;
            indsL      = 1:180;
            indsR      = 181:360;
            % match to shen
            load('distGlasserShen.mat');
            [mini,indMini]   = min(D,[],2);
            % indMini is the corresponding shen parcels
            % [sorted,indSort] = sort(indMini,'ascend');
            networkID_    = csvread('Z:\LabUsers\duboisjx\data\parcellations\shenetal_neuroimage2013_new\shen_268_parcellation_networklabels.csv',1);
            networkLabels = {'MedialFrontal','Frontoparietal','DefaultMode','SubcortCerebellum','Motor','VisualI','VisualII','VisualAssoc'};
            networkID  = [(1:nNodes)' zeros(nNodes,1)];
            for iNode = 1:length(indMini)
                networkID(iNode,2) = networkID_(indMini(iNode),2);
            end
    end
    nEdges     = sum(sum(triu(ones(nNodes),1)>0));
    %% reorder nodes
    % order by L- ascending network, R- descending network
    networks   = unique(networkID(:,2))';
    colors     = distinguishable_colors(length(networks));
    % left 1,2,3,4,5... right 1,2,3,4,5,...
    nodeOrderM  = [];
    % Left
    for iN = 1:length(networks)
        N = networks(iN);
        inds = find(networkID(:,2)==N & ismember(networkID(:,1),indsL));
        nodeOrderM = [nodeOrderM inds'];
    end
    % Right
    for iN = 1: length(networks)
        N = networks(iN);
        inds = find(networkID(:,2)==N & ismember(networkID(:,1),indsR));
        nodeOrderM = [nodeOrderM inds'];
    end
    networkBdrs     = [0 find(diff(networkID(nodeOrderM,2))~=0)' nNodes];
    networkLabelPos = conv(networkBdrs,[.5 .5],'same'); networkLabelPos = networkLabelPos(1:end-1);
    
    for iPredAlg = 1:length(predAlgs)
        predAlg = predAlgs{iPredAlg};
        
        for iPrep = 1:length(preprocs)
            preproc = preprocs{iPrep};
            inDir = fullfile('Z:\LabUsers\duboisjx\data\HCP\MRI\Results\FINAL',preproc,parcellation);
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
                                %                                 if iParcel==2 && iPredAlg==1 && iPrep==2
                                %                                     keyboard
                                %                                 end
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
                end
            end
        end
    end
end

% show theoretical chance level as gray area
r      = linspace(-1,1,2000);
t      = r.*sqrt((nSub-2)./(1-r.^2));
p2     = 2*(1-tcdf(abs(t),nSub-2));
thThr  = [r([find(p2>=0.025,1,'first') find(p2>=0.025,1,'last')]);...
    r([find(p2>=0.005,1,'first') find(p2>=0.005,1,'last')]);...
    r([find(p2>=0.0005,1,'first') find(p2>=0.0005,1,'last')])];
figure;
fid = fopen('results.csv','w');
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
                fprintf(fid,'%s,%s,%s,%s,%.3f,%.3f\n',...
                    SM,parcellation,preproc,predAlg,rhos(iSM,iParcel,iPrep,iPredAlg,1),rhos(iSM,iParcel,iPrep,iPredAlg,2));
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
fclose(fid);
