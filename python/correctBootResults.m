% recompute Pearson for all result files and save
% issue is that the Python code correlates the predicted scores with the
% non-deconfounded variables
clear all

preprocs      = {'Finn'};%,'Ciric6','SiegelB'}
parcellations = {'shen2013'};%'Glasser',
scores        = {'PMAT24_A_CR'};%,'NEOFAC_O','NEOFAC_C','NEOFAC_E','NEOFAC_A_corr','NEOFAC_N','Alpha','Beta'};
predAlgs      = {'Finn'};%,'elnet'};
sessions      = {'REST1'};%,'REST2'};
release       = 'all+MEG2';

confounds     = {'Gender','Age_in_Yrs','FS_BrainSeg_Vol','FDsum_REST1','FDsum_REST2','fMRI_3T_ReconVrs'};

for iPreproc = 1:length(preprocs)
    preproc = preprocs{iPreproc};
    for iParcel = 1:length(parcellations)
        parcellation = parcellations{iParcel};
        inDir = fullfile('Z:\LabUsers\duboisjx\data\HCP\MRI\Results\FINAL',preproc,parcellation);
        % load permutation indices
        permInds = load(fullfile(inDir,'permInds.txt'));
        % load variables of interest
        csvFile   = fullfile(inDir, 'df.csv');
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
        keyboard
        for iScore = 1:length(scores)
            eval(sprintf('%s = cellfun(@str2double,C{strcmp(headers,''%s'')});',scores{iScore},scores{iScore}));
        end
        for iCon = 1:length(confounds)
            eval(sprintf('%s = cellfun(@str2double,C{strcmp(headers,''%s'')});',confounds{iCon},confounds{iCon}));
        end
        for iScore = 1:length(scores)
            score = scores{iScore};
            if strcmp(score,'PMAT24_A_CR')
                decon = 'decon';
            else
                decon = 'decon+IQ';
            end
            for iSess = 1:length(sessions)
                session = sessions{iSess};
                switch decon
                    case 'decon'
                        %confounds=['gender','age','brainsize','motion','recon']
                        conMat = [Gender Age_in_Yrs FS_BrainSeg_Vol eval(sprintf('FDsum_%s',session)) fMRI_3T_ReconVrs];
                    case 'decon+IQ'
                        %confounds=['gender','age','brainsize','motion','recon','PMAT24_A_CR']
                        conMat = [Gender Age_in_Yrs FS_BrainSeg_Vol eval(sprintf('FDsum_%s',session)) fMRI_3T_ReconVrs PMAT24_A_CR];
                end
                % deconfound 
                X = [ones(length(subjects),1) conMat];
                b = regress(eval(score),X);
                resid_ = eval(score) - X*b;
                for iPredAlg = 1:length(predAlgs)
                    predAlg = predAlgs{iPredAlg};
                    for iPerm = 0:999
                        resFile = fullfile(inDir,...
                            sprintf('%s_%s_%s_%s_%s_%s_%s_thr0.01',...
                            score,preproc,parcellation,session,decon,predAlg,release),...
                            sprintf('%04d',iPerm),'result.mat');
                        if ~exist(resFile,'file')
                            continue
                        end
                        if ~exist(fullfile(fileparts(resFile),'_result.mat'),'file')
                            copyfile(resFile,fullfile(fileparts(resFile),'_result.mat'));
                        else
                            continue
                        end
                        result = load(resFile);
                        % need to shuffle score if iPerm>0
                        if iPerm>0
                            resid    = resid_(permInds(iPerm+1,:)+1); % note indexing issue
                        else
                            resid     = resid_;
                        end
                        switch predAlg
                            case 'Finn'
                                % pos
                                [RHO,PVAL]        = corr(resid,result.pred_pos);
                                result.rho_pos    = RHO;
                                result.p_pos      = PVAL;
                                % neg
                                [RHO,PVAL]        = corr(resid,result.pred_neg);
                                result.rho_neg    = RHO;
                                result.p_neg      = PVAL;
                                % posneg
                                [RHO,PVAL]        = corr(resid,result.pred_posneg);
                                result.rho_posneg = RHO;
                                result.p_posneg   = PVAL;
                                %                                 if strcmp(score,'NEOFAC_O') && strcmp(preproc,'Finn') && strcmp(parcellation,'Glasser') && ...
                                %                                         strcmp(session,'REST1')
                                %                                     figure;
                                %                                     stats = regstats(result.pred_pos,resid,'linear','beta');
                                %                                     regression_line_ci(.05,stats.beta,resid,result.pred_pos);
                                %                                     axis([-20 20 -20 20]);axis square;
                                %                                     hold on;
                                %                                     line([-20 20],[-20 20],'LineStyle','--','Color','k');
                                %                                     line([-20 20],[0 0],'LineStyle','--','Color','k');
                                %                                     title(sprintf('rho=%.3f, p=%.2E',result.rho_pos,result.p_pos));
                                %                                     return
                                %                                 end
                            otherwise
                                [RHO,PVAL]        = corr(resid,result.pred);
                                result.rho        = RHO;
                                result.p          = PVAL;
                        end
                        f = fields(result);
                        for iVar = 1:length(f)
                            eval(sprintf('%s=result.%s;',f{iVar},f{iVar}));
                            save(resFile,f{iVar},'-append');
                        end
                        fprintf('corrected %s\n',resFile);
                    end
                end
            end
        end
    end
end
        
    