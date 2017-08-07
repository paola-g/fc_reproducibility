from helpers import *
config.DATADIR      = op.join('Results','ciric7')
doPrediction = True
path = 'Results/'
ciric7_rest1 = sio.loadmat(path+'fcMats_Ciric7_shen2013_REST1.mat')
fcMatFile = path+'fcMats_Ciric7_shen2013_REST1.mat'
subjects = ciric7_rest1['subjects']
#config.outScore  = 'NEOFAC_A'
newdf = df[df['Subject'].isin([int(s) for s in subjects])]
config.parcellationName = 'shen2013'

# fMRI runs
session = 'REST1'
idcode = session
fmriRuns      = ['rfMRI_{}_LR'.format(session),'rfMRI_{}_RL'.format(session)]
# use volume or surface data
config.isCifti      = False

config.overwrite    = False

config.pipelineName = 'Ciric7'

config.queue        = False
# make sure to set memory requirements according to data size
config.sgeopts='-l h_vmem=30G -l h="node1|node2"'
# whether to use memmapping (which involves unzipping)
config.useMemMap    = False



# subject selection parameters
config.behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
config.release   = 'Q2'


#predict = 'motion'
predict = 'IQ'
config.queue        = False
launchSubproc = True
config.sgeopts      = '-l mem_free=2G' 
motFile = 'RMS_{}.txt'.format(session)
# run the IQ prediction for each subject
regression='Finn'
for regression in ['lasso', 'svm', 'elnet']:
    for config.outScore in ['NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E']: 
        score = np.ravel(newdf[config.outScore])
        for model in ['IQ']:
            runPredictionPar(fcMatFile,thresh=0.01, model=model,predict=predict, motFile=motFile,idcode=idcode, regression=regression)
            checkProgress(pause=5)
            config.joblist = []
            # merge cross-validation folds, save results
            n_subs          = fcMats_dn.shape[-1]
            predictions_pos = np.zeros([n_subs,1])
            predictions_neg = np.zeros([n_subs,1])
            
            iSub = 0
            for subject in subjects:
                results = sio.loadmat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}_{}_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, subject, session, regression))) 
                if regression=='Finn':
                    predictions_neg[iSub] = results['pred_neg']
                    predictions_pos[iSub] = results['pred_pos']
                else:
                    predictions_pos[iSub] = results['pred']
                iSub = iSub + 1
                
            if predict=='IQ':
                rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(score)))
            elif predict=='motion':
                rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(RelRMSMean)))
            if regression=='Finn':
                results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho_pos':rho, 'p_pos': p}
            else:
                results = {'pred':predictions_pos, 'rho':rho, 'p': p}
            print ('Correlation score (positive) {}: rho {} p {} ({})'.format(model, rho,p,config.pipelineName))
            if regression=='Finn':
                if predict=='IQ':
                    rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(score)))
                elif predict=='motion':
                    rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(RelRMSMean)))
                    print ('Correlation score (negative) {}: rho {} p {} ({})'.format(model, rho,p,config.pipelineName))
                    results['rho_neg'] = rho
                    results['p_neg'] = p
            # save result
            sio.savemat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}_{}.mat'.format(model,config.outScore,config.pipelineName, config.parcellationName, session, regression)),results)
