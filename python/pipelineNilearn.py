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
from subprocess import call, Popen, check_output, PIPE
import nibabel as nib
from shutil import copyfile, rmtree
import pandas as pd
import scipy.io as sio
from sklearn import cross_validation
from sklearn import linear_model
from numpy.polynomial.legendre import Legendre
import shlex
from scipy import signal
import gzip
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_roi
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img
nTRs=1200
def buildpath(subject,fmriRun):
    return 'test'
subject = '734045'
fmriRun = 'rfMRI_REST1_LR'
fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+'.nii.gz')


WMmaskFile = op.join(buildpath(subject,fmriRun),'WMmask.nii')
CSFmaskFile = op.join(buildpath(subject,fmriRun),'CSFmask.nii')
GMmaskFile = op.join(buildpath(subject,fmriRun),'GMmask.nii')

unionMask = intersect_masks([WMmaskFile, CSFmaskFile, GMmaskFile], threshold=0, connected=False)

maskWM_ = apply_mask(WMmaskFile, unionMask)
maskGM_ = apply_mask(GMmaskFile, unionMask)
maskCSF_ = apply_mask(CSFmaskFile, unionMask)

masker = NiftiMasker(mask_img=unionMask, standardize=True)
fmri_masked = masker.fit_transform(fmriFile)

order = 3
x = np.arange(nTRs)
x = x - x.max()/2
num_pol = range(order+1)
y = np.ones((len(num_pol),len(x)))   
coeff = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
# Print out text file for each polynomial to be used as a regressor
for i in num_pol:
    myleg = Legendre(coeff[i])
    y[i,:] = myleg(x) 
    if i>0:
	y[i,:] = y[i,:] - np.mean(y[i,:])
	y[i,:] = y[i,:]/np.max(y[i,:])
    np.savetxt(op.join(buildpath(subject,fmriRun),
		       'poly_detrend_legendre' + str(i) + '.txt'), y[i,:] ,fmt='%.2f')

invimg = masker.inverse_transform(fmri_masked)
invimg.to_filename('std_img.nii')
invmaskGM = masker.inverse_transform(maskGM_[np.newaxis,:])
invmaskCSF = masker.inverse_transform(maskCSF_[np.newaxis,:])
invmaskWM = masker.inverse_transform(maskWM_[np.newaxis,:])
X  = np.concatenate((np.ones([nTRs,1]), y[1:4,:].T), axis=1)
WMCSFmask = intersect_masks([invmaskCSF, invmaskWM], threshold=0, connected=False)
masker2 = NiftiMasker(mask_img=WMCSFmask, standardize=False, detrend=True)
fmri_masked2 = masker2.fit_transform(invimg, confounds=X)
fmri_masked[:,np.logical_or(maskWM_,maskCSF_)] = fmri_masked2
invimg2 = masker.inverse_transform(fmri_masked)
invimg2.to_filename('rwmcsf_img.nii')


masker3 = NiftiMasker(mask_img=invmaskWM)
fmri_masked3 = masker3.fit_transform(invimg2)
meanWM = np.mean(np.float64(fmri_masked3),axis=1)
meanWM = meanWM - np.mean(meanWM)
meanWM = meanWM/max(meanWM)

masker4 = NiftiMasker(mask_img=invmaskCSF)
fmri_masked4 = masker4.fit_transform(invimg2)
meanCSF = np.mean(np.float64(fmri_masked4),axis=1)
meanCSF = meanCSF - np.mean(meanCSF)
meanCSF = meanCSF/max(meanCSF)

X  = np.concatenate((np.ones([nTRs,1]), meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)
masker5 = NiftiMasker(mask_img=invmaskGM, detrend=True)
fmri_masked5 = masker5.fit_transform(invimg2, confounds=X)
fmri_masked[:,np.squeeze(np.where(maskGM_))] = fmri_masked5
invimg3 = masker.inverse_transform(fmri_masked)
invimg3.to_filename('rwmcsf_gm_img.nii')
