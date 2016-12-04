
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

behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
release = 'Q2'
outScore = 'PMAT24_A_CR'
DATADIR = '/data/jdubois/data/HCP/MRI'
PARCELDIR = '/data/jdubois/data/HCP/MRI/parcellations'
parcellation = 'shenetal_neuroimage2013'
overwrite = True
thisRun = 'rfMRI_REST1'
isDataClean = False
doPlot = False
isTest = True

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

def mktestdirs(subject,thisRun):
    basepath = op.join(DATADIR,'Testing',subject,'Results')
    for fmriRun in ['LR', 'RL']:
        makedirs(op.join(basepath,thisRun+'_'+fmriRun))
	
# ### Functions

# In[13]:

def makeTissueMasks(subject,fmriRun,overwrite):
    fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+suffix+'.nii.gz')
    WMmaskFileout = op.join(testpath(subject,fmriRun), 'WMmask.nii.gz')
    CSFmaskFileout = op.join(testpath(subject,fmriRun), 'CSFmask.nii.gz')
    GMmaskFileout = op.join(testpath(subject,fmriRun), 'GMmask.nii.gz')
    WMCSFmaskFileout = op.join(testpath(subject,fmriRun), 'WMCSFmask.nii.gz')
    WMCSFGMmaskFileout = op.join(testpath(subject,fmriRun), 'WMCSFGMmask.nii.gz')
    
    if not op.isfile(WMCSFGMmaskFileout) or overwrite:
        # load ribbon.nii.gz and wmparc.nii.gz
        ribbonFilein = op.join(DATADIR, subject, 'MNINonLinear','ribbon.nii.gz')
        wmparcFilein = op.join(DATADIR, subject, 'MNINonLinear', 'wmparc.nii.gz')
        # make sure it is resampled to the same space as the functional run
        ribbonFileout = op.join(testpath(subject,fmriRun), 'ribbon.nii.gz')
        wmparcFileout = op.join(testpath(subject,fmriRun), 'wmparc.nii.gz')
        # make identity matrix to feed to flirt for resampling
        with open('eye.mat','w') as fid:
            fid.write('1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1')
        
        flirt_ribbon = fsl.FLIRT(in_file=ribbonFilein, out_file=ribbonFileout,            reference=fmriFile, apply_xfm=True,            in_matrix_file='eye.mat', interp='nearestneighbour')
        flirt_ribbon.run()

        flirt_wmparc = fsl.FLIRT(in_file=wmparcFilein, out_file=wmparcFileout,            reference=fmriFile, apply_xfm=True,            in_matrix_file='eye.mat', interp='nearestneighbour')
        flirt_wmparc.run()
        
        # load nii (ribbon & wmparc)
        ribbon = nib.load(ribbonFileout).get_data()
        wmparc = nib.load(wmparcFileout).get_data()
        
        # white & CSF matter mask
        # indices are from FreeSurferColorLUT.txt
        
        # Left-Cerebral-White-Matter, Right-Cerebral-White-Matter
        ribbonWMstructures = [2, 41]
        # Left-Cerebral-Cortex, Right-Cerebral-Cortex
        ribbonGMstrucures = [3, 42]
        # Cerebellar-White-Matter-Left, Brain-Stem, Cerebellar-White-Matter-Right
        wmparcWMstructures = [7, 16, 46]
        # Left-Cerebellar-Cortex, Right-Cerebellar-Cortex, Thalamus-Left, Caudate-Left
        # Putamen-Left, Pallidum-Left, Hippocampus-Left, Amygdala-Left, Accumbens-Left 
        # Diencephalon-Ventral-Left, Thalamus-Right, Caudate-Right, Putamen-Right
        # Pallidum-Right, Hippocampus-Right, Amygdala-Right, Accumbens-Right
        # Diencephalon-Ventral-Right
        wmparcGMstructures = [8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60]
        # Fornix, CC-Posterior, CC-Mid-Posterior, CC-Central, CC-Mid-Anterior, CC-Anterior
        wmparcCCstructures = [250, 251, 252, 253, 254, 255]
        # Left-Lateral-Ventricle, Left-Inf-Lat-Vent, 3rd-Ventricle, 4th-Ventricle, CSF
        # Left-Choroid-Plexus, Right-Lateral-Ventricle, Right-Inf-Lat-Vent, Right-Choroid-Plexus
        wmparcCSFstructures = [4, 5, 14, 15, 24, 31, 43, 44, 63]
        
        # make masks
	WMmask = np.double(np.logical_or(np.logical_or(np.in1d(ribbon, ribbonWMstructures),np.in1d(wmparc, wmparcWMstructures)),np.logical_and(np.logical_and(np.in1d(wmparc, wmparcCCstructures),np.logical_not(np.in1d(wmparc, wmparcCSFstructures))),np.logical_not(np.in1d(wmparc, wmparcGMstructures)))))
        CSFmask = np.double(np.in1d(wmparc, wmparcCSFstructures))
        WMCSFmask = np.double((WMmask > 0) | (CSFmask > 0))
        GMmask = np.double(np.logical_or(np.in1d(ribbon,ribbonGMstrucures), np.in1d(wmparc,wmparcGMstructures)))
        WMCSFGMmask = np.double((WMmask > 0) | (CSFmask > 0) | (GMmask > 0))
        
        # write masks
        ref = nib.load(wmparcFileout)
        WMmask = np.reshape(WMmask,ref.shape)
        img = nib.Nifti1Image(WMmask, ref.affine)
        nib.save(img, WMmaskFileout)
        
        CSFmask = np.reshape(CSFmask,ref.shape)
        img = nib.Nifti1Image(CSFmask, ref.affine)
        nib.save(img, CSFmaskFileout)
        
        GMmask = np.reshape(GMmask,ref.shape)
        img = nib.Nifti1Image(GMmask, ref.affine)
        nib.save(img, GMmaskFileout)
        
        WMCSFmask = np.reshape(WMCSFmask,ref.shape)
        img = nib.Nifti1Image(WMCSFmask, ref.affine)
        nib.save(img, WMCSFmaskFileout)
        
        WMCSFGMmask = np.reshape(WMCSFGMmask,ref.shape)
        img = nib.Nifti1Image(WMCSFGMmask, ref.affine)
        nib.save(img, WMCSFGMmaskFileout)

subject = '118730'
fmriRun = 'rfMRI_REST1_LR'


makeTissueMasks(subject,fmriRun,True)





















































