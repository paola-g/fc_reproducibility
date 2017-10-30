# coding: utf-8

# # Load libraries & helper functions

# In[1]:

from helpers import *

config.DATADIR      = '/data2/jdubois2/data/HCP/MRI'
doPrediction = True
# fMRI runs
session = 'REST1'
#idcode = 'Q2_{}'.format(session)
idcode = session
fmriRuns      = ['rfMRI_{}_LR'.format(session),'rfMRI_{}_RL'.format(session)]
# use volume or surface data
config.isCifti      = True

config.overwrite    = False

config.pipelineName = 'Ciric7'
# use ICA-FIX input
config.useFIX       = False
config.preWhitening = False
config.Operations   = config.operationDict['Ciric6']
config.melodicFolder = op.join('#fMRIrun#_hp2000.ica','filtered_func_data.ica') #the code #fMRIrun# will be replaced
config.movementRegressorsFile = 'Movement_Regressors_dt.txt'
config.movementRelativeRMSFile = 'Movement_RelativeRMS.txt'
# submit jobs with sge
config.queue        = True
# make sure to set memory requirements according to data size
config.sgeopts='-l h_vmem=35G -l h="node1|node2"'
# whether to use memmapping (which involves unzipping)
config.useMemMap    = False

# parcellation for FC matrix
config.parcellationName = 'glasser'
config.parcellationFile = '/data/jdubois/data/parcellations/Glasser_Aseg_Suit/Parcels.dlabel.nii'
config.nParcels         = 405

# subject selection parameters
config.behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
config.release   = 'Q2'
config.outScore  = 'PMAT24_A_CR'


# # Subject selection

# In[3]:
data = sio.loadmat('flatsample_{}'.format(session))
subjects   = [str(el) for el in np.ravel(data['subjects']).astype(int)]
RelRMSMean = np.ravel(data['mot_score'])
score      = np.ravel(data['pmat_score'])


# # Do work

# ### preprocess everybody

# In[4]:

keepSub = np.zeros((len(subjects)),dtype=np.bool_)
iSub=0
for config.subject in subjects:
    iRun = 0
    for config.fmriRun in fmriRuns:
        keepSub[iSub] = runPipelinePar(launchSubproc=False)
        if not keepSub[iSub]:
            break
        iRun+=1
    iSub+=1
print 'Keeping {}/{} subjects'.format(np.sum(keepSub),len(subjects))
score    = score[keepSub]
RelRMSMean = RelRMSMean[keepSub]
if not op.isfile('RMS_flat_{}.txt'.format(session)):
    np.savetxt('RMS_flat_{}.txt'.format(session), RelRMSMean)
subjects   = [subject for (subject, keep) in zip(subjects, keepSub) if keep]

if config.queue:
    if len(config.joblist) != 0:
        while True:
            nleft = len(config.joblist)
            for i in range(nleft):
                myCmd = "qstat | grep ' {} '".format(config.joblist[i])
                isEmpty = False
                try:
                    cmdOut = check_output(myCmd, shell=True)
                except CalledProcessError as e:
                    isEmpty = True
                finally:
                    if isEmpty:
                        nleft = nleft-1
            if nleft == 0:
                break
            else:
                print 'Waiting for {} subjects to complete...'.format(nleft)
            sleep(10)
    print 'All done!!'

print 'Computing FC...'
fcMatFile = 'fcMats_{}_{}_flat_{}'.format(config.pipelineName, config.parcellationName, session)
if op.isfile('{}.mat'.format(fcMatFile)) and not config.overwrite:
    fcMats_dn = sio.loadmat(fcMatFile)['fcMats']
else:
    fcMats_dn    = np.zeros((config.nParcels,config.nParcels,len(subjects),len(fmriRuns)),dtype=np.float32)
    config.queue =False
    config.overwrite = False
    iSub= 0
    for config.subject in subjects:
        if keepSub[iSub]:
            iRun = 0
            for config.fmriRun in fmriRuns:
                runPipelinePar(launchSubproc=False)
                tsDir = op.join(buildpath(),config.parcellationName,config.fmriRun+config.ext)
                rstring = get_rcode(config.fmriFile_dn)
                fcFile_dn = op.join(tsDir,'allParcels_{}_Pearson.txt'.format(rstring))
                if not op.isfile(fcFile_dn) or config.overwrite: computeFC(True)
                fcMats_dn[:,:,iSub,iRun] = np.genfromtxt(fcFile_dn,delimiter=",")
                iRun = iRun+1
        iSub = iSub + 1

    fcMats_dn = np.squeeze(np.mean(np.arctanh(fcMats_dn),3))
    results = {}
    results['subjects']      = subjects
    results['fcMats'] = fcMats_dn
    results[config.outScore] = np.asarray(score)
    sio.savemat(fcMatFile, results)

# submit jobs with sge
#predict = 'motion'
predict = 'IQ'
config.queue        = True
launchSubproc = False
config.sgeopts      = '-l mem_free=2G' 
motFile = 'RMS_flat_{}.txt'.format(session)
# run the IQ prediction for each subject
#for model in ['IQ-mot', 'IQ+mot', 'mot-IQ']:
#for model in ['IQ-mot', 'IQ+mot', 'mot-IQ', 'parIQ', 'parmot']:
#for model in ['parIQ', 'parmot']:
for model in ['IQ', 'mot']:
    runPredictionPar(fcMatFile,thresh=0.01, model=model,predict=predict, motFile=motFile, idcode=idcode)
    checkProgress(pause=5)
    config.joblist = []
    # merge cross-validation folds, save results
    n_subs          = fcMats_dn.shape[-1]
    predictions_pos = np.zeros([n_subs,1])
    predictions_neg = np.zeros([n_subs,1])
    iSub = 0
    for subject in subjects:
       results = sio.loadmat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, subject, session))) 
       predictions_pos[iSub] = results['pred_pos']
       predictions_neg[iSub] = results['pred_neg']
       iSub = iSub + 1
    if predict=='IQ':
        rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(score)))
    elif predict=='motion':
        rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(RelRMSMean)))
        
    results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho_pos':rho, 'p_pos': p}
    print 'Correlation score (positive) {}: rho {} p {} ({})'.format(model, rho,p,config.pipelineName)
    if predict=='IQ':
        rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(score)))
    elif predict=='motion':
        rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(RelRMSMean)))
    print 'Correlation score (negative) {}: rho {} p {} ({})'.format(model, rho,p,config.pipelineName)
    results['rho_neg'] = rho
    results['p_neg'] = p
    # save result
    sio.savemat(op.join(config.DATADIR, '{}_{}pred_{}_{}_flat_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, session)),results)
