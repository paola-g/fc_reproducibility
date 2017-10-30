# # Load libraries & helper functions
from helpers import *

# # Declare parameters
config.DATADIR      = '/data2/jdubois2/data/HCP/MRI'

# fMRI runs
session = 'REST2'
fmriRuns      = ['rfMRI_{}_LR'.format(session),'rfMRI_{}_RL'.format(session)]
# use volume or surface data
config.isCifti      = False

config.overwrite    = False

config.pipelineName = 'Finn'
# use ICA-FIX input
config.useFIX       = False
config.preWhitening = False
config.Operations   = config.operationDict['Finn']
config.melodicFolder = op.join('#fMRIrun#_hp2000.ica','filtered_func_data.ica') #the code #fMRIrun# will be replaced
config.movementRegressorsFile = 'Movement_Regressors_dt.txt'
config.movementRelativeRMSFile = 'Movement_RelativeRMS.txt'
# submit jobs with sge
config.queue        = False
launchSubproc       = False
# make sure to set memory requirements according to data size
# 15G for HCP data!
config.sgeopts='-l h_vmem=35G -l h="node1|node2"'
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
fcMatFile = 'fcMats_{}_{}_{}'.format(config.pipelineName, config.parcellationName, session)
if op.isfile('{}.mat'.format(fcMatFile)):
    fcMats_dn = sio.loadmat(fcMatFile)['fcMats_{}_{}_{}'.format(config.pipelineName, config.parcellationName, session)]
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
                if not op.isfile(fcFile_dn): computeFC(True)
                fcMats_dn[:,:,iSub,iRun] = np.genfromtxt(fcFile_dn,delimiter=",")
                iRun = iRun+1
        iSub = iSub + 1

    fcMats_dn = np.squeeze(np.mean(np.arctanh(fcMats_dn),3))
    results = {}
    results['fcMats_{}_{}_{}'.format(config.pipelineName, config.parcellationName, session)] = fcMats_dn
    results[config.outScore] = np.asarray(score)
    sio.savemat(fcMatFile, results)

print "Starting IQ prediction..."
thresh = 0.01

config.queue        = True
config.sgeopts='-l h_vmem=2G -l h="node1|node2"'

iSub = 0
for config.subject in subjects:
        jobDir = op.join(config.DATADIR, config.subject,'MNINonLinear', 'Results','jobs')
        if not op.isdir(jobDir): 
	        mkdir(jobDir)
        jobName = 's{}_{}_iqpred'.format(config.subject,config.pipelineName)
	# make a script
	thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(getcwd())
	thispythonfn += 'from helpers import *\n'
	thispythonfn += 'logFid                  = open("{}","a+")\n'.format(op.join(jobDir,jobName+'.log'))
	thispythonfn += 'sys.stdout              = logFid\n'
	thispythonfn += 'sys.stderr              = logFid\n'
	# print date and time stamp
	thispythonfn += 'print "========================="\n'
	thispythonfn += 'print strftime("%Y-%m-%d %H:%M:%S", localtime())\n'
	thispythonfn += 'print "========================="\n'
	thispythonfn += 'config.subject          = "{}"\n'.format(config.subject)
	thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
	thispythonfn += 'config.pipelineName     = "{}"\n'.format(config.pipelineName)
	thispythonfn += 'config.parcellationName = "{}"\n'.format(config.parcellationName)
	thispythonfn += 'config.outScore = "{}"\n'.format(config.outScore)
	thispythonfn += 'makePrediction("{}","{}", "{}", {}, thresh={})\n'.format(config.subject, session, fcMatFile, iSub, thresh)
	thispythonfn += 'logFid.close()\n'
	thispythonfn += 'END'

	# prepare a script
	thisScript=op.join(jobDir,jobName+'.sh')
	while True:
		if op.isfile(thisScript) and (not config.overwrite):
			thisScript=thisScript.replace('.sh','+.sh') # use fsl feat convention
		else:
			break

	with open(thisScript,'w') as fidw:
		fidw.write('#!/bin/bash\n')
		fidw.write('echo ${FSLSUBALREADYRUN}\n')
		fidw.write('python {}\n'.format(thispythonfn))
	cmd='chmod 774 '+thisScript
	call(cmd,shell=True)

	#this is a "hack" to make sure the .sh script exists before it is called... 
	while not op.isfile(thisScript):
		sleep(.05)

	if config.queue:
		# call to fnSubmitToCluster
		JobID = fnSubmitToCluster(thisScript, jobDir, jobName, '-p {} {}'.format(-100,config.sgeopts))
		config.joblist.append(JobID)
		print 'submitted {} (SGE job #{})'.format(jobName,JobID)
		sys.stdout.flush()
	elif launchSubproc:
		#print 'spawned python subprocess on local machine'
		sys.stdout.flush()
		call(thisScript,shell=True)
	else:
		makePrediction(config.subject,session,fcMatFile,iSub,thresh=0.01)
        iSub = iSub +1

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

n_subs  = fcMats_dn.shape[-1]
predictions_pos = np.zeros([n_subs,1])
predictions_neg = np.zeros([n_subs,1])
iSub = 0
for config.subject in subjects:
   results = sio.loadmat(op.join(config.DATADIR, 'IQpred_{}_{}_{}_{}.mat'.format(config.pipelineName, config.parcellationName, config.subject, session))) 
   predictions_pos[iSub] = results['pred_pos']
   predictions_neg[iSub] = results['pred_neg']
   iSub = iSub + 1

rho,p = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(np.ravel(score)))
results = {'pred_pos':predictions_pos, 'pred_neg':predictions_neg, 'rho':rho, 'p': p}
sio.savemat(op.join(config.DATADIR, 'IQpred_{}_{}_{}.mat'.format(config.pipelineName, config.parcellationName, session)),results)
print 'Correlation score: rho {} p {}'.format(rho,p)
