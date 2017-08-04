# coding: utf-8

# # Load libraries & helper functions

# In[1]:

from helpers import *

config.DATADIR      = '/data2/jdubois2/data/HCP/MRI'
doPrediction = True
# fMRI runs
session = 'REST1'
fmriRuns      = ['rfMRI_{}_LR'.format(session),'rfMRI_{}_RL'.format(session)]
# use volume or surface data
config.isCifti      = False

config.overwrite    = False

config.pipelineName = 'SiegelA'
# use ICA-FIX input
config.useFIX       = False
config.preWhitening = False
config.Operations   = config.operationDict['SiegelA']
config.melodicFolder = op.join('#fMRIrun#_hp2000.ica','filtered_func_data.ica') #the code #fMRIrun# will be replaced
config.movementRegressorsFile = 'Movement_Regressors_dt.txt'
config.movementRelativeRMSFile = 'Movement_RelativeRMS.txt'
# make sure to set memory requirements according to data size
config.sgeopts='-l h_vmem=40G -l h="node1|node2"'
# whether to use memmapping (which involves unzipping)
config.useMemMap    = False

# parcellation for FC matrix
config.parcellationName = 'shen2013'
config.parcellationFile = '/data/jdubois/data/parcellations/shenetal_neuroimage2013_new/shen_2mm_268_parcellation.nii.gz'
config.nParcels         = 268

# subject selection parameters
config.behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
config.release   = 'Q2'
config.outScore  = 'PMAT24_A_CR'


# # Subject selection

# In[3]:

df = pd.read_csv(config.behavFile)
# select subjects that have completed all fMRI
ind = ((df['fMRI_WM_Compl']== True) & (df['fMRI_Mot_Compl']==True) 
             & (df['fMRI_Lang_Compl']==True) & (df['fMRI_Emo_Compl']==True)         
             & (df['RS-fMRI_Count']==4) & df[config.outScore].notnull())
df = df[ind]  
# check if either of the two subjects recommended for exclusion by HCP are still present
df = df[~df['Subject'].isin(['209733','528446'])]
df.index = range(df.shape[0])
tmpAgeRanges = sorted(df['Age'].unique())
# list of all selected subjects
subjects = df['Subject']
# pull their IQ, Age, Gender
age = df['Age']
gender = df['Gender']
score = df[config.outScore]

print 'Selected', str(df.shape[0]), 'from the release',config.release
print 'Number of males is:', df[df['Gender']=='M'].shape[0]
print 'Age range is', tmpAgeRanges

# Exclusion of high-motion subjects
# exclude subjects with >0.14 frame-to-frame head motion estimate averged across both rest runs (arbitrary threshold as in Finn et al 2015)
subjects   = [str(subject) for subject in subjects]
RelRMSMean = np.zeros([len(subjects), len(fmriRuns)],dtype=np.float32)
compSub    = np.zeros((len(subjects)),dtype=np.bool_)
keepSub    = np.zeros((len(subjects)),dtype=np.bool_)
iSub=0
for config.subject in subjects:
    i=0
    for config.fmriRun in fmriRuns:
        RelRMSMeanFile = op.join(buildpath(), 'Movement_RelativeRMS_mean.txt')
        if op.isfile(RelRMSMeanFile):
            with open(RelRMSMeanFile,'r') as tmp:
                RelRMSMean[iSub,i] = float(tmp.read())
        else:
            print RelRMSMeanFile+' File missing'
            break
        i=i+1
    if i==len(fmriRuns): # all RelRMSMeanFile exist
        compSub[iSub]=True
        if np.mean(RelRMSMean[iSub,:]) > 0.14:
            print config.subject, ': too much motion, exclude'
        else:
            keepSub[iSub]=True
    iSub=iSub+1

rho1,p1 = stats.pearsonr(score[keepSub],np.mean(RelRMSMean[keepSub,:],axis=1))
rho2,p2 = stats.pearsonr(score[compSub],np.mean(RelRMSMean[compSub,:],axis=1))            
print 'With all subjects: corr(IQ,motion) = {:.3f} (p = {:.3f})'.format(rho2,p2)
print 'After discarding high movers: corr(IQ,motion) = {:.3f} (p = {:.3f})'.format(rho1,p1)
    
subjects   = np.array([subject for (subject, keep) in zip(subjects, keepSub) if keep])
age        = age[keepSub]
gender     = gender[keepSub]
score      = score[keepSub]
RelRMSMean = RelRMSMean[keepSub,:]
print 'Keeping {} subjects [{} M]'.format(len(subjects),sum([g=='M' for g in gender]))
score = np.asarray(score)
# Generate random sample
sample_size = 120
repetitions = 100
if not op.isfile('samples.txt') or config.overwrite:
    samples = np.vstack([random.sample(range(len(subjects)),sample_size) for i in range(repetitions)])
    np.savetxt('samples.txt', samples)
else:
    samples = np.loadtxt('samples.txt',dtype=np.int16)
for iSample in range(repetitions):
    # submit jobs with sge
    config.queue        = True
    idx_smpl = samples[iSample,:]
    smpl_subjects = subjects[idx_smpl]
    smpl_score = score[idx_smpl]
    smpl_motscore = np.mean(RelRMSMean[idx_smpl,:], axis=1)
    np.savetxt('RMS_sample{}.txt'.format(iSample), smpl_motscore)
    # # Do work
    
    # ### preprocess everybody
    
    # In[4]:
    
    keepSub = np.zeros((len(smpl_subjects)),dtype=np.bool_)
    iSub=0
    for config.subject in smpl_subjects:
        iRun = 0
        for config.fmriRun in fmriRuns:
            keepSub[iSub] = runPipelinePar(launchSubproc=False)
            if not keepSub[iSub]:
                break
            iRun+=1
        iSub+=1
    print 'Keeping {}/{} subjects'.format(np.sum(keepSub),len(smpl_subjects))
    smpl_score    = smpl_score[keepSub]
    smpl_motscore    = smpl_motscore[keepSub]

    checkProgress(pause=20)

    print 'Computing FC...'
    fcMatFile = 'fcMats_{}_{}_{}'.format(config.pipelineName, config.parcellationName, 'sample{}'.format(iSample))
    config.overwrite = True
    if op.isfile('{}.mat'.format(fcMatFile)) and not config.overwrite:
        fcMats_dn = sio.loadmat(fcMatFile)['fcMats']
    else:
        fcMats_dn    = np.zeros((config.nParcels,config.nParcels,len(smpl_subjects),len(fmriRuns)),dtype=np.float32)
        config.queue =False
        config.overwrite = False
        iSub= 0
        for config.subject in smpl_subjects:
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
        results['subjects']      = smpl_subjects
        results['fcMats'] = fcMats_dn
        results[config.outScore] = np.asarray(smpl_score)
        sio.savemat(fcMatFile, results)
    
    # submit jobs with sge
    predict = 'IQ'
    #predict = 'motion'
    config.queue        = False
    launchSubproc = False
    config.sgeopts      = '-l mem_free=2G' 
    motFile = 'RMS_sample{}.txt'.format(iSample)
    # run the prediction for each subject
    #for model in ['IQ','IQ-mot', 'IQ+mot', 'mot-IQ', 'parIQ', 'parmot']:
    for model in ['IQ', 'mot']:
        runPredictionPar(fcMatFile,thresh=0.01, model=model,predict=predict, motFile=motFile, idcode='sample{}'.format(iSample))
        if config.queue: 
            checkProgress(pause=5)
            config.joblist = []
        # merge cross-validation folds, save results
        n_subs          = fcMats_dn.shape[-1]
        predictions_pos = np.zeros([n_subs,1])
        predictions_neg = np.zeros([n_subs,1])
        iSub = 0
        for subject in smpl_subjects:
           results = sio.loadmat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, subject, 'sample{}'.format(iSample)))) 
           predictions_pos[iSub] = results['pred_pos']
           predictions_neg[iSub] = results['pred_neg']
           iSub = iSub + 1
        if predict=='IQ':
            rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(smpl_score)))
        elif predict=='motion':
            rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(smpl_motscore)))
        results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho_pos':rho, 'p_pos': p}
        print 'Correlation score (positive) {}: rho {} p {} ({})'.format(model, rho,p,config.pipelineName)
        if predict=='IQ':
            rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(smpl_score)))
        elif predict=='motion':
            rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(smpl_motscore)))
        print 'Correlation score (negative) {}: rho {} p {} ({})'.format(model, rho,p,config.pipelineName)
        results['rho_neg'] = rho
        results['p_neg'] = p
        # save result
        sio.savemat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, 'sample{}'.format(iSample))),results)
