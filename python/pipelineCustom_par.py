from runPipeline import *

# ### Get subjects

df = pd.read_csv(config.behavFile)

# select subjects according to release
if config.release == 'Q2':
    ind = (df['Release'] == 'Q2')     | (df['Release'] == 'Q1')
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
print 'Selected', str(df.shape[0]), 'from the release',config.release
print 'Number of males is:', df[df['Gender']=='M'].shape[0]
tmpAgeRanges = sorted(df['Age'].unique())
print 'Age range is', tmpAgeRanges[0].split('-')[0], '-', tmpAgeRanges[-1].split('-')[1]

# list of all selected subjects
subjects = df['Subject']
# pull their IQ, Age, Gender
age = df['Age']
gender = df['Gender']
score = df[config.outScore]


# ### Exclusion of high-motion subjects
# Further exclude subjects with >0.14 frame-to-frame head motion estimate averged across both rest runs (arbitrary threshold as in Finn et al 2015)

# In[ ]:

ResultsDir = op.join(DATADIR,'Results')
if not op.isdir(ResultsDir): mkdir(ResultsDir)
ResultsDir = op.join(ResultsDir, config.pipelineName)
if not op.isdir(ResultsDir): mkdir(ResultsDir)
ResultsDir = op.join(ResultsDir, config.parcellation)
if not op.isdir(ResultsDir): mkdir(ResultsDir)

PEdirs = ['LR', 'RL']
RelRMSMean = np.zeros([len(subjects), 2])
excludeSub = list()
joblist = []
config.logfile = op.join(ResultsDir,'log_{}_HCP_{}.txt'.format(config.thisRun,config.release))
for iSub in range(len(subjects)):
    subject = str(subjects[iSub])
    RelRMSMeanFile = op.join(buildpath(subject, config.thisRun+'_zz'), 'Movement_RelativeRMS_mean.txt')
    fLR = RelRMSMeanFile.replace('zz','LR');
    fRL = RelRMSMeanFile.replace('zz','RL');
    
    if op.isfile(fLR) & op.isfile(fRL):
        with open(fLR,'r') as tmp:
            RelRMSMean[iSub,0] = float(tmp.read())
        with open(fRL,'r') as tmp:
            RelRMSMean[iSub,1] = float(tmp.read())
        print '{} {:.3f} {:.3f}'.format(subjects[iSub], RelRMSMean[iSub,0], RelRMSMean[iSub,1])
        if np.mean(RelRMSMean[iSub,:]) > 0.14:
            print subjects[iSub], ': too much motion, exclude'
            excludeSub.append(iSub)
            continue
    
            
  
    for iPEdir in range(len(PEdirs)):
        PEdir=PEdirs[iPEdir]
        fmriRun = config.thisRun+'_'+PEdir
        if config.parcellation=='shenetal_neuroimage2013':
            fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+suffix+'.nii.gz')
            config.isCifti=0
        elif config.parcellation=='shenetal_neuroimage2013_new':
            fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+suffix+'.nii.gz')
            config.isCifti=0
        elif config.parcellation=='Glasser_Aseg_Suit':
            fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+'_Atlas'+suffix+'.dtseries.nii')
            config.isCifti=1
        elif config.parcellation=='Glasser_CIT168Amy_Aseg_Suit':
            fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+'_Atlas'+suffix+'.dtseries.nii')
            config.isCifti=1
        else:
            print 'Wrong parcellation code'
            exit()
        if not op.isfile(fmriFile):
            print str(subjects[iSub]), 'missing', fmriFile, ', exclude'
            excludeSub.append(iSub)
            continue
        
        if not (op.isfile(op.join(ResultsDir, str(subjects[iSub])+'_'+config.thisRun+'_'+PEdir+'.txt'))) or config.overwrite:
            if config.queue:
                # make a script to load and preprocess that file, then save as .mat
                jobDir = op.join(buildpath(str(subjects[iSub]),config.thisRun+'_'+PEdir),'jobs')
                if not op.isdir(jobDir): mkdir(jobDir)
                thispythonfn = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(getcwd())
                thispythonfn += 'from runPipeline import *\n'
                thispythonfn += 'config.subject = "{}"\n'.format(subject)
                thispythonfn += 'config.fmriRun = "{}"\n'.format(fmriRun)
		thispythonfn += 'runPipeline("{}","{}","{}")\nEND\n'.format(subject,fmriRun,fmriFile)
                jobName = 's{}_{}_{}_{}'.format(subjects[iSub],config.thisRun,PEdir, config.pipelineName)
                # prepare a script
                thisScript=op.join(jobDir,jobName+'.sh')
                with open(thisScript,'w') as fidw:
                    fidw.write('#!/bin/bash\n')
                    fidw.write('echo ${FSLSUBALREADYRUN}\n')
                    fidw.write('python {}'.format(thispythonfn))
                cmd='chmod 774 '+thisScript
                call(cmd,shell=True)
                # call to fnSubmitToCluster
                JobID = fnSubmitToCluster(thisScript,jobDir, jobName, '-p {} -l h_vmem=19G'.format(priority))
                joblist.append(JobID)
            else:
		config.subject = subject
		config.fmriRun = fmriRun
                runPipeline(subject, fmriRun, fmriFile)
        else:
            print subjects[iSub], ' : ', PEdir, 'results already computed; skipping'

indkeep = np.setdiff1d(range(len(subjects)),excludeSub, assume_unique=True)

if config.queue:
    if len(joblist) != 0:
        print 'Waiting for jobs...'
        while True:
            nleft = len(joblist)
            for i in range(nleft):
                myCmd = "qstat | grep ' {} '".format(joblist[i])
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
            sleep(10) 
		
rho1,p1 = stats.pearsonr(score[indkeep],np.mean(RelRMSMean[indkeep,:],axis=1))
rho2,p2 = stats.pearsonr(score,np.mean(RelRMSMean,axis=1))                         
print 'With all subjects: corr(IQ,motion) = {:.3f} (p = {:.3f})'.format(rho2,p2)
print 'After discarding high movers: corr(IQ,motion) = {:.3f} (p = {:.3f})'.format(rho1,p1)

print 'Computing correlation matrices...'
if config.parcellation=='shenetal_neuroimage2013':
    nParcels = 268
if config.parcellation=='shenetal_neuroimage2013_new':
    nParcels = 268
elif config.parcellation=='Glasser_Aseg_Suit':
    nParcels = 405
elif config.parcellation=='Glasser_CIT168Amy_Aseg_Suit':
    nParcels = 423
for iSub in range(len(subjects)):
    if iSub not in excludeSub:
        tsFile_LR=op.join(ResultsDir,str(subjects[iSub])+'_'+config.thisRun+'_LR.txt')
        tsFile_RL=op.join(ResultsDir,str(subjects[iSub])+'_'+config.thisRun+'_RL.txt')
        if not op.isfile(tsFile_LR) or not op.isfile(tsFile_RL):
            excludeSub.append(iSub)
            print ('Warning! Missing output file for subject',subjects[iSub])
indkeep = np.setdiff1d(range(len(subjects)),excludeSub, assume_unique=True)
corrmats = np.zeros([nParcels,nParcels,len(indkeep)])
scores = np.zeros([len(indkeep)])
index = 0
for iSub in range(len(subjects)):
    if iSub not in excludeSub:
        PEdir=PEdirs[iPEdir] 
        tsFile_LR=op.join(ResultsDir,str(subjects[iSub])+'_'+config.thisRun+'_LR.txt')
        tsFile_RL=op.join(ResultsDir,str(subjects[iSub])+'_'+config.thisRun+'_RL.txt')
        ts_LR = np.loadtxt(tsFile_LR)
        ts_RL = np.loadtxt(tsFile_RL)
        # Fisher z transform of correlation coefficients
        corrMat_LR = np.arctanh(np.corrcoef(ts_LR,rowvar=0))
        corrMat_RL = np.arctanh(np.corrcoef(ts_RL,rowvar=0))
        np.fill_diagonal(corrMat_LR,1)
        np.fill_diagonal(corrMat_RL,1)
        corrmats[:,:,index] = (corrMat_LR + corrMat_RL)/2
        scores[index] = score[iSub]
	index = index + 1
        
results = {}
results[outMat] = corrmats
results[config.outScore] = scores
sio.savemat(op.join(ResultsDir,'{}_HCP_{}.mat'.format(config.thisRun,config.release)),results)
