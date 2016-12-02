# ### Get subjects

# In[16]:

df = pd.read_csv(behavFile)

# select subjects according to release
if release == 'Q2':
    ind = (df['Release'] == 'Q2')     | (df['Release'] == 'Q1')
elif release == 'S500':
    ind = (df['Release'] != 'Q2') & (df['Release'] != 'Q1')
else:
    sys.exit("Invalid release code")
    
# select subjects that have completed all fMRI
ind = ind & ((df['fMRI_WM_Compl']== True) & (df['fMRI_Mot_Compl']==True)         & (df['fMRI_Lang_Compl']==True) & (df['fMRI_Emo_Compl']==True)         & (df['RS-fMRI_Count']==4))
                
df = df[ind]  

# check if either of the two subjects recommended for exclusion by HCP are still present
df = df[~df['Subject'].isin(['209733','528446'])]
df.index = range(df.shape[0])
print 'Selected', str(df.shape[0]), 'from the release',release
print 'Number of males is:', df[df['Gender']=='M'].shape[0]
tmpAgeRanges = sorted(df['Age'].unique())
print 'Age range is', tmpAgeRanges[0].split('-')[0], '-', tmpAgeRanges[-1].split('-')[1]

# list of all selected subjects
subjects = df['Subject']
# pull their IQ, Age, Gender
age = df['Age']
gender = df['Gender']
score = df[outScore]


# ### Exclusion of high-motion subjects
# Further exclude subjects with >0.14 frame-to-frame head motion estimate averged across both rest runs (arbitrary threshold as in Finn et al 2015)

# In[ ]:

ResultsDir = op.join(DATADIR, 'Testing','Results')
if not op.isdir(ResultsDir): makedirs(ResultsDir)
#ResultsDir = op.join(DATADIR,'Results')
#if not op.isdir(ResultsDir): mkdir(ResultsDir)
ResultsDir = op.join(ResultsDir, 'Finn')
if not op.isdir(ResultsDir): mkdir(ResultsDir)
ResultsDir = op.join(ResultsDir, parcellation)
if not op.isdir(ResultsDir): mkdir(ResultsDir)

PEdirs = ['LR', 'RL']
RelRMSMean = np.zeros([len(subjects), 2])
excludeSub = list()
joblist = []
for iSub in [6,8]:
    RelRMSMeanFile = op.join(buildpath(str(subjects[iSub]), thisRun+'_zz'), 'Movement_RelativeRMS_mean.txt')
    fLR = RelRMSMeanFile.replace('zz','LR');
    fRL = RelRMSMeanFile.replace('zz','RL');
    
    if op.isfile(fLR) & op.isfile(fRL):
        with open(fLR,'r') as tmp:
            RelRMSMean[iSub,0] = float(tmp.read())
        with open(fRL,'r') as tmp:
            RelRMSMean[iSub,1] = float(tmp.read())
        print '{} {:.3f} {:.3f}'.format(subjects[iSub], RelRMSMean[iSub,0], RelRMSMean[iSub,1])
        if np.mean(RelRMSMean[iSub,:]) > 0.141:
            print subjects[iSub], ': too much motion, exclude'
            excludeSub.append(iSub)
            continue
     
    for iPEdir in range(len(PEdirs)):
        PEdir=PEdirs[iPEdir]
        fmriFile = op.join(buildpath(str(subjects[iSub]),thisRun+'_'+PEdir),                           thisRun+'_'+PEdir+suffix+'.nii.gz')
        if not op.isfile(fmriFile):
            print str(subjects[iSub]), 'missing', fmriFile, ', exclude'
            excludeSub.append(iSub)
            continue
        
        if not (op.isfile(op.join(ResultsDir, str(subjects[iSub])+'_'+thisRun+'_'+PEdir+'.txt')))         or not (op.isfile(op.join(ResultsDir, str(subjects[iSub])+'_'+thisRun+'_'+PEdir+'._GM.txt')))         or overwrite:
	    
            print 'load and preprocess'
            if isTest and not op.isfile(testpath(str(subjects[iSub]),thisRun+'_'+PEdir)) and not op.isfile(testpath(str(subjects[iSub]),thisRun+'_'+PEdir)):
                try:
                    mktestdirs(str(subjects[iSub]),thisRun)
		except Exception as e:
		    print 'isTest:',isTest
	    if queue:
		# make a script to load and preprocess that file, then save as .mat
		thispythonfn = 'import Finn_loadandpreprocess\nFinn_loadandpreprocess("{}","{}",{})'.format(fmriFile,parcellation,str(overwrite))
		jobDir = op.join(testpath(str(subjects[iSub]),thisRun+'_'+PEdir),'jobs')	
		if not op.isdir(jobDir): mkdir(jobDir)
		jobName = 's{}_{}_{}_makeFCmat'.format(subjects[iSub],thisRun,PEdir)
		# prepare a script
		thisScript=op.join(jobDir,jobName+'.sh')
		with open(thisScript,'w') as fidw:
			fidw.write('#!/bin/bash\n')
			fidw.write('echo ${FSLSUBALREADYRUN}\n')
			fidw.write('python {}'.format(thispythonfn))
		cmd='chmod 774 '+thisScript
		call(cmd,shell=True)
		# call to fnSubmitToCluster		
		JobID = fnSubmitToCluster(thisScript,jobDir, jobName, '-p {} -l h_vmem=15G'.format(priority))
		joblist.append(JobID)
	    else:
            	Finn_loadandpreprocess(fmriFile, parcellation, overwrite)
        else:
            print subject[iSub], ' : ', PEdir, 'results already computed; skipping'

indkeep = np.setdiff1d(range(len(subjects)),excludeSub, assume_unique=True)



