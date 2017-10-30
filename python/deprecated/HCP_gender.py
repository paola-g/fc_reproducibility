# # Load libraries & helper functions
from helpers import *

# # Declare parameters
config.DATADIR      = '/data2/jdubois2/data/HCP/MRI'

# fMRI runs
session = 'REST1'
idcode = '{}'.format(session)
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
config.sgeopts='-l h_vmem=30G'
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
# select subjects that have completed all rsfMRI
ind = ind  & (df['RS-fMRI_Count']==4)
df = df[ind]  
# check if either of the two subjects recommended for exclusion by HCP are still present
df = df[~df['Subject'].isin(['209733','528446'])]
# removing subjects with a MMSE score < 26
df = df[df['MMSE_Score']>25]
# behavioural criterion for exclusion (stable outliers)
outliers = np.loadtxt('outliers.txt', dtype=np.int32)
df = df[~df['Subject'].isin(outliers)]
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


print 'Selected', str(df.shape[0]), 'from the release',config.release
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
    
subjects   = np.array([subject for (subject, keep) in zip(subjects, keepSub) if keep])
age        = age[keepSub]
gender     = gender[keepSub]
score      = np.array(score[keepSub])
RelRMSMean = RelRMSMean[keepSub,:]
print 'Keeping {} subjects [{} M]'.format(len(subjects),sum([g=='M' for g in gender]))

f_idx = np.where(gender=='F')[0]
m_idx = np.where(gender=='M')[0]
equateGenders = False
if equateGenders:
    max_size = min(f_idx.shape, m_idx.shape)[0]
    f_score = score[f_idx[:max_size]]
    m_score = score[m_idx[:max_size]]
    f_motscore = np.mean(RelRMSMean[f_idx[:max_size],:], axis=1)
    m_motscore = np.mean(RelRMSMean[m_idx[:max_size],:], axis=1)
    np.savetxt('RMS_{}_{}_F.txt'.format(session, config.release), f_motscore)
    np.savetxt('RMS_{}_{}_M.txt'.format(session, config.release), m_motscore)
    f_subjects = subjects[f_idx[:max_size]]
    m_subjects = subjects[m_idx[:max_size]]
    suffix='eq' 
else:
    f_score = score[f_idx]
    m_score = score[m_idx]
    f_motscore = np.mean(RelRMSMean[f_idx,:], axis=1)
    m_motscore = np.mean(RelRMSMean[m_idx,:], axis=1)
    np.savetxt('RMS_{}_{}_F.txt'.format(session, config.release), f_motscore)
    np.savetxt('RMS_{}_{}_M.txt'.format(session, config.release), m_motscore)
    f_subjects = subjects[f_idx]
    m_subjects = subjects[m_idx]
    suffix=''

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
if len(config.joblist)>0:
    checkProgress(pause=5, verbose=True)

if len(config.scriptlist)>0:
    config.sgeopts      = '-l mem_free=25G' 
    JobID = fnSubmitJobArrayFromJobList()
    print 'Running array job {} ({} sub jobs)'.format(JobID.split('.')[0],JobID.split('.')[1].split('-')[1].split(':')[0])
    config.joblist.append(JobID.split('.')[0])
    checkProgress()

print 'Computing FC...'
fcMatFile = 'fcMats_{}_{}_{}_{}_gender'.format(config.pipelineName, config.parcellationName, session, config.release)

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
#config.sgeopts      = '-l mem_free=8G -pe openmp 6' 
config.sgeopts      = '-l mem_free=8G' 
motFile = 'RMS_{}_{}.txt'.format(session, config.release)
if not op.isfile(motFile):
    np.savetxt(motFile, np.mean(RelRMSMean, axis=1))
# run the IQ prediction for each subject
#for mode in ['IQ-mot', 'IQ+mot', 'mot-IQ']:
family = pd.read_csv('HCPfamily.csv')
newfamily_f = family[family['Subject'].isin([int(s) for s in f_subjects])]
newdf_f       = df[df['Subject'].isin([int(s) for s in f_subjects])]
newfamily_m = family[family['Subject'].isin([int(s) for s in m_subjects])]
newdf_m       = df[df['Subject'].isin([int(s) for s in m_subjects])]
fcMats_f = fcMats_dn[:, :, [np.where(np.in1d(subjects,str(elem)))[0][0] for elem in f_subjects]]
fcMats_m = fcMats_dn[:, :, [np.where(np.in1d(subjects,str(elem)))[0][0] for elem in m_subjects]]
results = {}
results['subjects']      = f_subjects
results['fcMats'] = fcMats_f
results[config.outScore] = f_score
results['motscore'] = f_motscore
fcMatFile_F = '{}_{}{}'.format(fcMatFile, 'F', suffix)
sio.savemat(fcMatFile_F, results)
results = {}
results['subjects']      = m_subjects
results['fcMats'] = fcMats_m
results[config.outScore] = m_score
results['motscore'] = m_motscore
fcMatFile_M = '{}_{}{}'.format(fcMatFile, 'M', suffix)
sio.savemat(fcMatFile_M, results)
#### LADIES FIRST
#for regression in ['lasso','mlr','elnet','svm']:
regression = 'Finn'
#for config.outScore in ['PMAT24_A_CR', 'NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']:
for config.outScore in ['PMAT24_A_CR']:
    runPredictionParFamily(fcMatFile_F,thresh=0.01, model='IQ', motFile='RMS_{}_{}_F.txt'.format(session, config.release), idcode=idcode, regression=regression, gender='F')
    checkProgress(pause=5, verbose=False)
    # merge cross-validation folds, save results
    n_subs          = len(f_subjects)
    predictions_pos = np.zeros([n_subs,1])
    predictions_neg = np.zeros([n_subs,1])
    for el in np.unique(newfamily_f['Family_ID']):
       idx = np.array(newfamily_f.ix[newfamily_f['Family_ID']==el]['Subject'])
       sidx = np.array(newdf_f.ix[newdf_f['Subject'].isin(idx)]['Subject'])
       test_index = [np.where(np.in1d(f_subjects,str(elem)))[0][0] for elem in sidx]
       results = sio.loadmat(op.join('', '{}_{}pred_{}_{}_{}_{}_{}_{}.mat'.format('IQ',config.outScore,config.pipelineName, config.parcellationName, '_'.join(['%s' % el for el in sidx]), idcode+'_F', regression, config.release))) 
       if regression=='Finn':
           predictions_neg[test_index] = results['pred_neg'].T
           predictions_pos[test_index] = results['pred_pos'].T
       else:
           predictions_pos[test_index] = results['pred'].T
    rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(f_score)))
    print 'Correlation score {} ({}): rho {} p {}'.format(config.outScore, regression, rho,p)
    # save result
    if regression=='Finn':
        results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho_pos':rho, 'p_pos': p}
        rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(f_score)))
        print 'Correlation score {} ({}): rho {} p {} (neg)'.format(config.outScore, regression, rho,p)
        results['rho_neg'] = rho
        results['p_neg'] = p
    else:
        results = {'pred':predictions_pos, 'rho_pos':rho, 'p_pos': p}   
    sio.savemat(op.join('{}_{}pred_{}_{}_{}_{}_{}_F.mat'.format('IQ',config.outScore,config.pipelineName, config.parcellationName, regression, config.release, suffix)),results)
#### GENTLEMEN
#for regression in ['lasso','mlr','elnet','svm']:
regression = 'Finn'
#for config.outScore in ['PMAT24_A_CR', 'NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']:
for config.outScore in ['PMAT24_A_CR']:
    runPredictionParFamily(fcMatFile_M,thresh=0.01,model='IQ',motFile='RMS_{}_{}_M.txt'.format(session, config.release),idcode=idcode,regression=regression,gender='M')
    checkProgress(pause=5, verbose=False)
    # merge cross-validation folds, save results
    n_subs          = len(m_subjects)
    predictions_pos = np.zeros([n_subs,1])
    predictions_neg = np.zeros([n_subs,1])
    for el in np.unique(newfamily_m['Family_ID']):
       idx = np.array(newfamily_m.ix[newfamily_m['Family_ID']==el]['Subject'])
       sidx = np.array(newdf_m.ix[newdf_m['Subject'].isin(idx)]['Subject'])
       test_index = [np.where(np.in1d(m_subjects,str(elem)))[0][0] for elem in sidx]
       results = sio.loadmat(op.join('', '{}_{}pred_{}_{}_{}_{}_{}_{}.mat'.format('IQ',config.outScore,config.pipelineName, config.parcellationName, '_'.join(['%s' % el for el in sidx]), idcode+'_M', regression, config.release))) 
       if regression=='Finn':
           predictions_neg[test_index] = results['pred_neg'].T
           predictions_pos[test_index] = results['pred_pos'].T
       else:
           predictions_pos[test_index] = results['pred'].T
    rho,p   = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(m_score)))
    print 'Correlation score {} ({}): rho {} p {}'.format(config.outScore, regression, rho,p)
    # save result
    if regression=='Finn':
        results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho_pos':rho, 'p_pos': p}
        rho,p   = stats.pearsonr(np.ravel(predictions_neg),np.squeeze(np.ravel(m_score)))
        print 'Correlation score {} ({}): rho {} p {} (neg)'.format(config.outScore, regression, rho,p)
        results['rho_neg'] = rho
        results['p_neg'] = p
    else:
        results = {'pred':predictions_pos, 'rho_pos':rho, 'p_pos': p}   
    sio.savemat(op.join('{}_{}pred_{}_{}_{}_{}_{}_M.mat'.format('IQ',config.outScore,config.pipelineName, config.parcellationName, regression, config.release, suffix)),results)
