# coding: utf-8

# # Load libraries & helper functions

# In[1]:

from helpers import *
from makePrediction import *

config.DATADIR      = '/data2/jdubois2/data/HCP/MRI'
doPrediction = True
# fMRI runs
train_session = 'REST1'
test_session = 'REST2'
idcode = '{}to{}'.format(train_session, test_session)
fmriRuns      = ['rfMRI_{}_LR'.format(train_session),'rfMRI_{}_RL'.format(train_session)]
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

df = pd.read_csv(config.behavFile)
# select subjects according to release
if config.release == 'Q2':
    ind = (df['Release'] == 'Q2') | (df['Release'] == 'Q1')
elif config.release == 'S500':
    ind = (df['Release'] != 'Q2') & (df['Release'] != 'Q1')
else:
    sys.exit("Invalid release code")
# select subjects that have completed all fMRI
ind = ind & ((df['fMRI_WM_Compl']== True) & (df['fMRI_Mot_Compl']==True) 
             & (df['fMRI_Lang_Compl']==True) & (df['fMRI_Emo_Compl']==True)         
             & (df['RS-fMRI_Count']==4))
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
print 'Age range is', tmpAgeRanges[0].split('-')[0], '-', tmpAgeRanges[-1].split('-')[1]

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
    
subjects   = [subject for (subject, keep) in zip(subjects, keepSub) if keep]
age        = age[keepSub]
gender     = gender[keepSub]
score      = score[keepSub]
RelRMSMean = RelRMSMean[keepSub,:]
print 'Keeping {} subjects [{} M]'.format(len(subjects),sum([g=='M' for g in gender]))


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
RelRMSMean = np.mean(RelRMSMean[keepSub,:],axis=1)
if not op.isfile('RMS_{}.txt'.format(train_session)):
    np.savetxt('RMS_{}.txt'.format(train_session), RelRMSMean)

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
fcMatFile = 'fcMats_{}_{}_{}'.format(config.pipelineName, config.parcellationName, train_session)
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
motFile = 'RMS_{}.txt'.format(train_session)
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
       results = sio.loadmat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, subject, idcode))) 
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
    sio.savemat(op.join(config.DATADIR, '{}_{}pred_{}_{}_{}.mat'.format(model,predict,config.pipelineName, config.parcellationName, idcode)),results)
