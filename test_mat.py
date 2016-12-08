from __future__ import division
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
import numpy as np
import os.path as op
from os import mkdir, makedirs
import scipy.stats as stats
import nipype.interfaces.fsl as fsl
from subprocess import call
import nibabel as nib
from shutil import copyfile
import pandas as pd
import scipy.io as sio
from sklearn import cross_validation
from sklearn import linear_model
from numpy.polynomial.legendre import Legendre


behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
release = 'Q2'
outScore = 'PMAT24_A_CR'
DATADIR = '/data/jdubois/data/HCP/MRI'
PARCELDIR = '/data/jdubois/data/parcellations'
parcellation = 'shenetal_neuroimage2013'
overwrite = False
thisRun = 'rfMRI_REST1'
isDataClean = True
doPlot = False
useLegendre = True
queue = False

if thisRun == 'rfMRI_REST1':
    outMat = 'rest_1_mat'
elif thisRun == 'rfMRI_REST2':
    outMat = 'rest_1_mat'
else:
    sys.exit("Invalid run code")  
    
suffix = '_hp2000_clean' if isDataClean else ''   

def buildpath(subject,fmriRun):
    return op.join(DATADIR, subject,'MNINonLinear','Results',fmriRun)

def testpath(subject,fmriRun):
    return op.join(DATADIR, 'Testing', subject,'Results',fmriRun)
df = pd.read_csv(behavFile)

# select subjects according to release
if release == 'Q2':
    ind = (df['Release'] == 'Q2') \
    | (df['Release'] == 'Q1')
elif release == 'S500':
    ind = (df['Release'] != 'Q2') & (df['Release'] != 'Q1')
else:
    sys.exit("Invalid release code")
    
# select subjects that have completed all fMRI
ind = ind & ((df['fMRI_WM_Compl']== True) & (df['fMRI_Mot_Compl']==True) \
        & (df['fMRI_Lang_Compl']==True) & (df['fMRI_Emo_Compl']==True) \
        & (df['RS-fMRI_Count']==4))
                
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



ResultsDir = op.join(DATADIR,'Results')
if not op.isdir(ResultsDir): mkdir(ResultsDir)
ResultsDir = op.join(ResultsDir, 'Finn')
if not op.isdir(ResultsDir): mkdir(ResultsDir)
ResultsDir = op.join(ResultsDir, parcellation)
if not op.isdir(ResultsDir): mkdir(ResultsDir)

PEdirs = ['LR', 'RL']
RelRMSMean = np.zeros([len(subjects), 2])
excludeSub = list()

for iSub in range(len(subjects)):
    subject = str(subjects[iSub])
    RelRMSMeanFile = op.join(buildpath(subject, thisRun+'_zz'), 'Movement_RelativeRMS_mean.txt')
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
        fmriRun = thisRun+'_'+PEdir 
        fmriFile = op.join(buildpath(subject,fmriRun),
                           fmriRun+suffix+'.nii.gz')
        if not op.isfile(fmriFile):
            print str(subjects[iSub]), 'missing', fmriFile, ', exclude'
            excludeSub.append(iSub)
            continue
        
        if not (op.isfile(op.join(ResultsDir, str(subjects[iSub])+'_'+thisRun+'_'+PEdir+'.txt'))) \
        or not (op.isfile(op.join(ResultsDir, str(subjects[iSub])+'_'+thisRun+'_'+PEdir+'._GM.txt'))) \
        or overwrite:
            print subject[iSub], ' : ', PEdir, 'load and preprocess'
        else:
            print subject[iSub], ' : ', PEdir, 'results already computed; skipping'

indkeep = np.setdiff1d(range(len(subjects)),excludeSub, assume_unique=True)

# Whole Parcels
corrmats = np.zeros([268,268,len(indkeep)])
scores = np.zeros([len(indkeep)])
index = 0
for iSub in range(len(subjects)):
    if iSub not in excludeSub:
        PEdir=PEdirs[iPEdir] 
        tsFile_LR=op.join(ResultsDir,str(subjects[iSub])+'_'+thisRun+'_LR.txt')
        tsFile_RL=op.join(ResultsDir,str(subjects[iSub])+'_'+thisRun+'_RL.txt')
        ts_LR = np.loadtxt(tsFile_LR)
        ts_RL = np.loadtxt(tsFile_RL)
        # Fisher z transform of correlation coefficients
        corrMat_LR = np.arctanh(np.corrcoef(ts_LR,rowvar=0))
        corrMat_RL = np.arctanh(np.corrcoef(ts_RL,rowvar=0))
        np.fill_diagonal(corrMat_LR,1)
        np.fill_diagonal(corrMat_RL,1)
        corrmats[:,:,index] = (corrMat_LR + corrMat_RL)/2
        scores[index] = score[iSub]
        
results = {}
results[outMat] = corrmats
results[outScore] = scores
sio.savemat('my_{}_HCP_{}.mat'.format(thisRun,release),results)

# GM Parcels        
index = 0
for iSub in range(len(subjects)):
    if iSub not in excludeSub:
        PEdir=PEdirs[iPEdir] 
        tsFile_LR=op.join(ResultsDir,str(subjects[iSub])+'_'+thisRun+'_LR_GM.txt')
        tsFile_RL=op.join(ResultsDir,str(subjects[iSub])+'_'+thisRun+'_RL_GM.txt')
        ts_LR = np.loadtxt(tsFile_LR)
        ts_RL = np.loadtxt(tsFile_RL)
        # Fisher z transform of correlation coefficients
        corrMat_LR = np.arctanh(np.corrcoef(ts_LR,rowvar=0))
        corrMat_RL = np.arctanh(np.corrcoef(ts_RL,rowvar=0))
        np.fill_diagonal(corrMat_LR,1)
        np.fill_diagonal(corrMat_RL,1)
        corrmats[:,:,index] = (corrMat_LR + corrMat_RL)/2
        scores[index] = score[iSub]
        
results = {}
results[outMat] = corrmats
results[outScore] = scores
sio.savemat('my_{}_HCP_{}_GM.mat'.format(thisRun,release),results)      

