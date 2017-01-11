
# coding: utf-8

# ### Required libraries

# In[12]:

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


# ### Parameters

# In[2]:
myoffset=352
behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
release = 'Q2'
outScore = 'PMAT24_A_CR'
DATADIR = '/data/jdubois/data/HCP/MRI'
PARCELDIR = '/data/jdubois/data/HCP/MRI/parcellations'
parcellation = 'shenetal_neuroimage2013'
overwrite = False
thisRun = 'rfMRI_REST1'
isDataClean = False
doPlot = True
useLegendre = True
nPoly = 4
queue = False
useFeat = False
preproOnly = True
doTsmooth = True
normalize = 'pcSigCh'
isCifti = False
if thisRun == 'rfMRI_REST1':
    outMat = 'rest_1_mat'
elif thisRun == 'rfMRI_REST2':
    outMat = 'rest_1_mat'
else:
    sys.exit("Invalid run code")  
    
suffix = '_hp2000_clean' if isDataClean else ''   

def buildpath(subject,fmriRun):
    return 'test'

def testpath(subject,fmriRun):
    return op.join(DATADIR, 'Testing', subject,'Results',fmriRun)


# ### Functions

# In[3]:

def makeTissueMasks(subject,fmriRun,overwrite):
    fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+suffix+'.nii.gz')
    WMmaskFileout = op.join(buildpath(subject,fmriRun), 'WMmask.nii')
    CSFmaskFileout = op.join(buildpath(subject,fmriRun), 'CSFmask.nii')
    GMmaskFileout = op.join(buildpath(subject,fmriRun), 'GMmask.nii')
    WMCSFmaskFileout = op.join(buildpath(subject,fmriRun), 'WMCSFmask.nii')
    WMCSFGMmaskFileout = op.join(buildpath(subject,fmriRun), 'WMCSFGMmask.nii')
    
    if not op.isfile(WMCSFGMmaskFileout) or overwrite:
        # load ribbon.nii.gz and wmparc.nii.gz
        ribbonFilein = op.join(DATADIR, subject, 'MNINonLinear','ribbon.nii.gz')
        wmparcFilein = op.join(DATADIR, subject, 'MNINonLinear', 'wmparc.nii.gz')
        # make sure it is resampled to the same space as the functional run
        ribbonFileout = op.join(buildpath(subject,fmriRun), 'ribbon.nii.gz')
        wmparcFileout = op.join(buildpath(subject,fmriRun), 'wmparc.nii.gz')
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
        WMmask = np.double(np.logical_and(np.logical_and(np.logical_or(np.logical_or(np.in1d(ribbon, ribbonWMstructures),
                                                                              np.in1d(wmparc, wmparcWMstructures)),
                                                                np.in1d(wmparc, wmparcCCstructures)),
                                                  np.logical_not(np.in1d(wmparc, wmparcCSFstructures))),
                                   np.logical_not(np.in1d(wmparc, wmparcGMstructures))))
        CSFmask = np.double(np.in1d(wmparc, wmparcCSFstructures))
        WMCSFmask = np.double((WMmask > 0) | (CSFmask > 0))
        GMmask = np.double(np.logical_or(np.in1d(ribbon,ribbonGMstrucures),np.in1d(wmparc,wmparcGMstructures)))
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





# In[4]:


subject = '734045'
fmriRun = 'rfMRI_REST1_LR'



            


# In[20]:

# PREPROCESSING as in Finn et al Nat Neuro 2015
# 1. "Regress temporal drift from CSF and white matter (3rd order polynomial)"
# 2. Regress CSF/WM signal from gray matter voxels
# 3. Regress motion parameters (found in the Movement_Regressors_dt.txt file from HCP)
# 4. Temporal smoothing with Gaussian kernel (sigma = 1 TR)
# 5. Regress temporal drift from gray matter (3rd order polynomial)    
# 6. Regress global mean (mask includes all voxels in brain mask  
mysuffix = '_FinnPrepro_'+normalize
if not doTsmooth:
    mysuffix = mysuffix + '_noTsmooth'

if isCifti:
    fmriFile = op.join(buildpath(subject,fmriRun),fmriRun+suffix+'.dtseries.nii')
    outFile = op.join(buildpath(subject,fmriRun),fmriRun+suffix+mysuffix+'_FinnPrepro.dtseries.nii')
    prefix = 'GrayOrdStep'
else:
    fmriFile = op.join(buildpath(subject,fmriRun),fmriRun+suffix+'.nii.gz')
    outFile = op.join(buildpath(subject,fmriRun),fmriRun+suffix+mysuffix+'_FinnPrepro.nii.gz')
    prefix = 'VolumeStep'

WMmaskFile = op.join(buildpath(subject,fmriRun),'WMmask.nii')
CSFmaskFile = op.join(buildpath(subject,fmriRun),'CSFmask.nii')
GMmaskFile = op.join(buildpath(subject,fmriRun),'GMmask.nii')


# make WM, CSF, GM masks (if not already done)
if not op.isfile(GMmaskFile):
    makeTissueMasks(subject,fmriRun,overwrite)
    
tmpnii = nib.load(WMmaskFile)
data = np.memmap(WMmaskFile, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
		 offset=myoffset,shape=tmpnii.header.get_data_shape())
nRows, nCols, nSlices = tmpnii.header.get_data_shape()
maskWM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
del data
tmpnii = nib.load(CSFmaskFile)
data = np.memmap(CSFmaskFile, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
		 offset=myoffset,shape=tmpnii.header.get_data_shape())
maskCSF = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
del data
tmpnii = nib.load(GMmaskFile)
data = np.memmap(GMmaskFile, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
		 offset=myoffset,shape=tmpnii.header.get_data_shape())
maskGM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
del data
maskAll = np.logical_or(np.logical_or(maskWM, maskCSF), maskGM)
maskWM_ = maskWM[maskAll]
maskGM_ = maskGM[maskAll]
maskCSF_ = maskCSF[maskAll]


# get some info
if isCifti:
    cmd = 'cut -d \' \' -f 4 <<< $(wb_command -file-information {} -no-map-info|grep "Number of Columns")'.format(fmriFile)
    nTRs = long(check_output(cmd,shell=True))
    cmd = 'wb_command -cifti-convert -to-text {} {}'.format(fmriFile,op.join(buildpath(subject,fmriRun),'.tsv'))
    call(cmd,shell=True)
else:
    img = nib.load(fmriFile)
    hdr = img.header.structarr
    nTRs = long(hdr['dim'][4])
    
    
## DO PREPROCESSING:
## 1) Regress temporal drift from CSF and white matter (3rd order Legendre polynomial)
print 'Step 1 (detrend WMCSF voxels, polynomial order 3)'
# ** a) create polynomial regressor **
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
		       'poly_detrend_legendre' + str(i) + '.txt'), y[i,:] ,fmt='%.4f')

# keep only WM/CSF voxels to speed things up
if isCifti:
    volFile = fmriFileOrig.replace('_Atlas.dtseries.nii','.nii.gz')
else:
    outFilePath = op.join(buildpath(subject, fmriRun), fmriRun+'.nii')
    
    with gzip.open(fmriFile, 'rb') as fFile:
	outFilePath = op.join(buildpath(subject, fmriRun), fmriRun+'.nii')
	with open(outFilePath, 'wb') as outfile:
	    outfile.write(fFile.read())

    volFile = outFilePath
    
img = nib.load(volFile)
data = np.memmap(volFile, dtype=img.header.get_data_dtype(), mode='c', order='F',
		 offset=myoffset,shape=img.header.get_data_shape())

nRows, nCols, nSlices, nTRs = img.header.get_data_shape()
niiImg = data.reshape([nRows*nCols*nSlices, nTRs], order='F')
niiImg = niiImg[maskAll,:]

niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step0.nii')
del niiimg 

if normalize == 'zscore':
    niiImg = stats.zscore(niiImg, axis=1, ddof=1)
    print niiImg.shape
elif normalize == 'pcSigCh':
    niiImg = 100 * (niiImg - np.mean(niiImg,axis=0)) / np.mean(niiImg,axis=0)

niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step1.nii')
del niiimg 

niiImgWMCSF = niiImg[np.logical_or(maskWM_,maskCSF_),:]
X  = np.concatenate((np.ones([nTRs,1]), y[1:4,:].T), axis=1)
N = niiImgWMCSF.shape[0]
for i in range(N):
    fit = np.linalg.lstsq(X, niiImgWMCSF[i,:].T)[0]
    fittedvalues = np.dot(X, fit)
    resid = niiImgWMCSF[i,:] - np.ravel(fittedvalues)
    if normalize == 'keepMean':
	niiImgWMCSF[i,:] = X[:,0]*fit[0] + resid
    else:
	niiImgWMCSF[i,:] = resid
	
niiImg[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
del niiImgWMCSF
    
niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step2.nii')
del niiimg 


## 2. Regress CSF/WM signal from gray matter voxels
print 'Step 2 (regress WM/CSF signal from GM (two separate means))'
# ** a) extract the WM and the CSF data from the detrended volume
meanWM = np.mean(np.float64(niiImg[maskWM_,:]),axis=0)
meanWM = meanWM - np.mean(meanWM)
meanWM = meanWM/max(meanWM)
meanCSF = np.mean(np.float64(niiImg[maskCSF_,:]),axis=0)
meanCSF = meanCSF - np.mean(meanCSF)
meanCSF = meanCSF/max(meanCSF)
X  = np.concatenate((np.ones([nTRs,1]), meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)
if not isCifti:
    niiImgGM = niiImg[maskGM_,:]
else:
    niiImgGM = np.genfromtxt(op.join(buildpath(subject,fmriRun),'.tsv'))
    if normalize == 'zscore':
	niiImgGM = stats.zscore(niiImgGM, axis=1, ddof=1)
    elif normalize == 'pcSigCh':
	niiImgGM = 100 * (niiImgGM - np.mean(niiImgGM,axis=0)) / np.mean(niiImgGM,axis=0)
niiImgGM[np.isnan(niiImgGM)] = 0
N = niiImgGM.shape[0]
for i in range(N):
    fit = np.linalg.lstsq(X, niiImgGM[i,:].T)[0]
    fittedvalues = np.dot(X, fit)
    resid = niiImgGM[i,:] - np.ravel(fittedvalues)
    if normalize == 'keepMean':
	niiImgGM[i,:] = X[:,0]*fit[0] + resid
    else:
	niiImgGM[i,:] = resid
	
if not isCifti:
    niiImg[maskGM_,:] = niiImgGM
else:
    niiImg = niiImgGM
del niiImgGM  

niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step3.nii')
del niiimg      

## 3. Regress motion parameters (found in the Movement_Regressors_dt_txt
# file from HCP)    
print 'Step 3 (regress 12 motion parameters from whole brain)'
# ** a) load the detrended motion parameters
motionFile = op.join(buildpath(subject,fmriRun),                            'Movement_Regressors_dt.txt')
# this needs to be split into columns
colNames = ['mmx','mmy','mmz','degx','degy','degz','dmmx','dmmy','dmmz','ddegx','ddegy','ddegz']
df = pd.read_csv(motionFile,delim_whitespace=True,header=None)
df.columns = colNames
for iCol in range(len(colNames)):
    with open(motionFile.replace('.txt','_'+colNames[iCol]+'.txt'),'w') as tmp:
	df.to_csv(path_or_buf=tmp,sep='\n',columns=[colNames[iCol]],header=None,index=False)
     

X  = np.concatenate((np.ones([nTRs,1]),
		     np.loadtxt(motionFile.replace('.txt','_'+colNames[0]+'.txt'))[:,np.newaxis]), axis=1)
for iCol in range(1,len(colNames)):
    X = np.concatenate((X,np.loadtxt(motionFile.replace('.txt','_'+colNames[iCol]+'.txt'))[:,np.newaxis]),axis=1)

N = niiImg.shape[0]
for i in range(N):
    fit = np.linalg.lstsq(X, niiImg[i,:].T)[0]
    fittedvalues = np.dot(X, fit)
    resid = niiImg[i,:] - np.ravel(fittedvalues)
    if normalize == 'keepMean':
	niiImg[i,:] = X[:,0]*fit[0] + resid
    else:
	niiImg[i,:] = resid
	
niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step4.nii')
del niiimg 


## 4. Temporal smoothing with Gaussian kernel (sigma = 1 TR)       
if doTsmooth:
    print 'Step 4 (temporal smoothing with Gaussian kernel)'
    w = signal.gaussian(11,std=1)
    niiImg = signal.lfilter(w,1,niiImg)

niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step5.nii')
del niiimg 
    
## 5. Regress temporal drift from gray matter (3rd order polynomial)
print ('Step 5 (detrend gray matter voxels, polynomial order 3)')
# the polynomial regressors have already been created

if isCifti:
    niiImgGM = niiImg
else:
    niiImgGM = niiImg[maskGM_,:]
    
 
X  = np.concatenate((np.ones([nTRs,1]), y[1:4,:].T), axis=1)

N = niiImgGM.shape[0]
for i in range(N):
    fit = np.linalg.lstsq(X, niiImgGM[i,:].T)[0]
    fittedvalues = np.dot(X, fit)
    resid = niiImgGM[i,:] - np.ravel(fittedvalues)
    if normalize == 'keepMean':
	niiImgGM[i,:] = X[:,0]*fit[0] + resid
    else:
	niiImgGM[i,:] = resid
    
if not isCifti:
    niiImg[maskGM_,:] = niiImgGM
else:
    niiImg = niiImgGM
del niiImgGM    

niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
niiimg[maskAll,:] = niiImg
niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
nib.save(newimg,'test/step6.nii')
del niiimg 
    
## 6. Regress global mean (mask includes all voxels in brain mask,
# gray matter, white matter and CSF
print 'Step 6 (GSR)'
meanAll = np.mean(niiImg,axis=0)
meanAll = meanAll - np.mean(meanAll)
meanAll = meanAll/max(meanAll)
X  = np.concatenate((np.ones([nTRs,1]), meanAll[:,np.newaxis]), axis=1)
N = niiImg.shape[0]
for i in range(N):
    fit = np.linalg.lstsq(X, niiImg[i,:].T)[0]
    fittedvalues = np.dot(X, fit)
    resid = niiImg[i,:] - np.ravel(fittedvalues)
    if normalize == 'keepMean':
	niiImg[i,:] = X[:,0]*fit[0] + resid
    else:
	niiImg[i,:] = resid        
    
## We're done! Copy the resulting file
if isCifti:
    # write to text file
    np.savetxt(op.join(buildpath(subject,fmriRun),'.tsv'),niiImg, delimiter='\t', fmt='%.6f')
    # need to convert back to cifti
    cmd = 'wb_command -cifti-convert -from-text {} {} {}'.format(op.join(buildpath(subject,fmriRun),'.tsv'),
								 fmriFile,outFile)
    call(cmd,shell=True)
    # delete temporary files
    cmd = 'rm -r {}/*.tsv'.format(buildpath(subject,fmriRun))
    call(cmd,shell=True)
    del niiImg
else:
    niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
    niiimg[maskAll,:] = niiImg
    del niiImg
    niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
    newimg = nib.Nifti1Image(niiimg, img.affine, header=img.header)
    nib.save(newimg,outFile)
    del niiimg

del data     



