# # Load libraries & helper functions
from helpers import *

# # Declare parameters
config.DATADIR      = '/data2/jdubois2/data/HCP/MRI'

# fMRI runs
session = 'REST1'
idcode = 'Q2_{}'.format(session)
fmriRuns      = ['rfMRI_{}_LR'.format(session),'rfMRI_{}_RL'.format(session)]
# use volume or surface data
config.isCifti      = False

config.overwrite    = False



config.pipelineName = 'Ciric5'
# use ICA-FIX input
config.useFIX       = False
config.preWhitening = False

config.Operations   = config.operationDict['Ciric5']
config.melodicFolder = op.join('#fMRIrun#_hp2000.ica','filtered_func_data.ica') #the code #fMRIrun# will be replaced
config.movementRegressorsFile = 'Movement_Regressors_dt.txt'
config.movementRelativeRMSFile = 'Movement_RelativeRMS.txt'
# submit jobs with sge
config.queue        = True
launchSubproc       = False
# make sure to set memory requirements according to data size
# 15G for HCP data!
config.sgeopts='-l h_vmem=30G -pe openmp 6'
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
        #print RelRMSMeanFile
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


 # Do work

 ### preprocess everybody

keepSub = np.zeros((len(subjects)),dtype=np.bool_)
iSub=0
#for config.subject in ['109325']:
for config.subject in subjects:
    iRun = 0
    for config.fmriRun in fmriRuns:
        if not config.queue:
            print 'SUB {}/{} [{}]: run {}/{} [{}]'.format(iSub+1,len(subjects),config.subject,iRun+1,len(fmriRuns),config.fmriRun)
        keepSub[iSub] = runPipelinePar(launchSubproc=False)
        if not keepSub[iSub]:
            break
        iRun+=1
    iSub+=1
print 'Keeping {}/{} subjects'.format(np.sum(keepSub),len(subjects))
score    = score[keepSub]



if len(config.scriptlist)>0:
    config.sgeopts      = '-l mem_free=25G' 
    JobID = fnSubmitJobArrayFromJobList()
    print 'Running array job {} ({} sub jobs)'.format(JobID.split('.')[0],JobID.split('.')[1].split('-')[1].split(':')[0])
    config.joblist.append(JobID.split('.')[0])
    checkProgress()

print 'Computing FC...'
fcMatFile = 'fcMats_{}_{}_{}'.format(config.pipelineName, config.parcellationName, session)

if op.isfile('{}.mat'.format(fcMatFile)) and not config.overwrite:
    fcMats_dn = sio.loadmat(fcMatFile)['fcMats']
else:
    fcMats_dn    = np.zeros((config.nParcels,config.nParcels,len(subjects),len(fmriRuns)),dtype=np.float32)
    config.queue =False
    config.overwrite = False
    iSub= 0
    for config.subject in subjects:
        if keepSub[iSub]:
            print 'SUB {}/{}: {}'.format(iSub+1,len(subjects),config.subject)
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






print "Starting IQ prediction..."
# submit jobs with sge
config.queue        = True
config.overwrite = True
launchSubproc = False
config.sgeopts      = '-l mem_free=8G -pe openmp 6' 
motFile = np.loadtxt('RMS_{}.txt'.format(session))
# run the IQ prediction for each subject
#for mode in ['IQ-mot', 'IQ+mot', 'mot-IQ']:
family = pd.read_csv('HCPfamily.csv')
newfamily = family[family['Subject'].isin([int(s) for s in subjects])]
newdf       = df[df['Subject'].isin([int(s) for s in subjects])]
for regression in ['svm', 'lasso', 'elnet']:
    runPredictionParFamily(fcMatFile,thresh=0.01, model='IQ', motFile='RMS_{}.txt'.format(session), idcode=idcode, regression=regression)
    checkProgress()
    # merge cross-validation folds, save results
    n_subs          = len(subjects)
    predictions_pos = np.zeros([n_subs,1])
    predictions_neg = np.zeros([n_subs,1])
    for el in np.unique(newfamily['Family_ID']):
       idx = np.array(newfamily.ix[newfamily['Family_ID']==el]['Subject'])
       sidx = np.array(newdf.ix[newdf['Subject'].isin(idx)]['Subject'])
       test_index = [np.where(np.in1d(subjects,str(elem)))[0][0] for elem in sidx]
       results = sio.loadmat(op.join('', '{}_{}pred_{}_{}_{}_{}_{}_{}.mat'.format('IQ',config.outScore,config.pipelineName, config.parcellationName, '_'.join(['%s' % el for el in sidx]), idcode, regression, config.release))) 
       if regression=='Finn':
           predictions_neg[test_index] = results['pred_neg'].T
           predictions_pos[test_index] = results['pred_pos'].T
       else:
           predictions_pos[test_index] = results['pred'].T
    rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(score)))
    print 'Correlation score {}: rho {} p {}'.format(config.outScore, rho,p)
    # save result
    if regression=='Finn':
        results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho_pos':rho, 'p_pos': p}
        rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(score)))
        results['rho_neg'] = rho
        results['p_neg'] = p
    else:
        results = {'pred':predictions_pos, 'rho_pos':rho, 'p_pos': p}   
    sio.savemat(op.join('{}_{}pred_{}_{}_{}_{}_3.mat'.format('IQ',config.outScore,config.pipelineName, config.parcellationName, idcode, regression)),results)
