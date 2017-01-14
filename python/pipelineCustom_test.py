
# coding: utf-8

# ### Required libraries

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
from subprocess import call, Popen, check_output
import nibabel as nib
from shutil import copyfile, rmtree
import pandas as pd
import scipy.io as sio
from sklearn import cross_validation
from sklearn import linear_model
from numpy.polynomial.legendre import Legendre
import shlex
from scipy import signal
import operator
import gzip


# ### Utils

def regress(niiImg, nTrs, regressors, keepMean):
    X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
    N = niiImg.shape[0]
    print 'inside regress {}'.format(N)
    for i in range(N):
        fit = np.linalg.lstsq(X, niiImg[i,:].T)[0]
        fittedvalues = np.dot(X, fit)
        resid = niiImg[i,:] - np.ravel(fittedvalues)
        if keepMean:
            niiImg[i,:] = X[:,0]*fit[0] + resid
        else:
            niiImg[i,:] = resid
    return niiImg     

def normalize(niiImg,flavor):
    if flavor == 'zscore':
        niiImg = stats.zscore(niiImg, axis=1, ddof=1)
        return niiImg
    elif flavor == 'pcSigCh':
        niiImg = 100 * (niiImg - np.mean(niiImg,axis=1)[:,np.newaxis]) / np.mean(niiImg,axis=1)[:,np.newaxis]
    else:
        print 'Warning! Wrong normalization flavor. Nothing was done'
    return niiImg    

def legendre_poly(order, nTRs):
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
                           'poly_detrend_legendre' + str(i) + '.txt'), y[i,:] ,fmt='%.2f')
    return y

def load_img(fmriFile):
    if isCifti:
        toUnzip = fmriFile.replace('_Atlas.dtseries.nii','.nii.gz')
        cmd = 'wb_command -cifti-convert -to-text {} {}'.format(fmriFile,op.join(buildpath(subject,fmriRun),'.tsv'))
        call(cmd,shell=True)
    else:
        toUnzip = fmriFile

    with open(toUnzip, 'rb') as fFile:
        decompressedFile = gzip.GzipFile(fileobj=fFile)
        outFilePath = op.join(buildpath(subject, fmriRun), fmriRun+'.nii')
        with open(outFilePath, 'wb') as outfile:
            outfile.write(decompressedFile.read())

    volFile = outFilePath

    img = nib.load(volFile)
    myoffset = img.header.sizeof_hdr + 4 + img.header.get_data_offset()
    data = np.memmap(volFile, dtype=img.header.get_data_dtype(), mode='c', order='F',
                     offset=myoffset,shape=img.header.get_data_shape())

    nRows, nCols, nSlices, nTRs = img.header.get_data_shape()
    niiImg = data.reshape([nRows*nCols*nSlices, nTRs], order='F')
    niiImg = niiImg[maskAll,:]
    return niiImg, nRows, nCols, nSlices, nTRs, img.affine

def plot_hist(score,title,xlabel):
    h,b = np.histogram(score, bins='auto')
    plt.hist(score,bins=b)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    return h

def makeTissueMasks(subject,fmriRun,overwrite):
    fmriFile = op.join(buildpath(subject,fmriRun), fmriRun+suffix+'.nii.gz')
    WMmaskFileout = op.join(buildpath(subject,fmriRun), 'WMmask.nii')
    CSFmaskFileout = op.join(buildpath(subject,fmriRun), 'CSFmask.nii')
    GMmaskFileout = op.join(buildpath(subject,fmriRun), 'GMmask.nii')
    
    if not op.isfile(GMmaskFileout) or overwrite:
        # load ribbon.nii.gz and wmparc.nii.gz
        ribbonFilein = op.join(DATADIR, subject, 'MNINonLinear','ribbon.nii.gz')
        wmparcFilein = op.join(DATADIR, subject, 'MNINonLinear', 'wmparc.nii.gz')
        # make sure it is resampled to the same space as the functional run
        ribbonFileout = op.join(buildpath(subject,fmriRun), 'ribbon.nii.gz')
        wmparcFileout = op.join(buildpath(subject,fmriRun), 'wmparc.nii.gz')
        # make identity matrix to feed to flirt for resampling
        with open('eye.mat','w') as fid:
            fid.write('1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1')
        
        flirt_ribbon = fsl.FLIRT(in_file=ribbonFilein, out_file=ribbonFileout,
					reference=fmriFile, apply_xfm=True,
					in_matrix_file='eye.mat', interp='nearestneighbour')
        flirt_ribbon.run()

        flirt_wmparc = fsl.FLIRT(in_file=wmparcFilein, out_file=wmparcFileout,
					reference=fmriFile, apply_xfm=True,
					in_matrix_file='eye.mat', interp='nearestneighbour')
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
        GMmask = np.double(np.logical_or(np.in1d(ribbon,ribbonGMstrucures),np.in1d(wmparc,wmparcGMstructures)))
        
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
        
        
    tmpnii = nib.load(WMmaskFileout)
    myoffset = tmpnii.header.sizeof_hdr + 4 + tmpnii.header.get_data_offset()
    data = np.memmap(WMmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
                     offset=myoffset,shape=tmpnii.header.get_data_shape())
    nRows, nCols, nSlices = tmpnii.header.get_data_shape()
    maskWM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    del data
    tmpnii = nib.load(CSFmaskFileout)
    myoffset = tmpnii.header.sizeof_hdr + 4 + tmpnii.header.get_data_offset()
    data = np.memmap(CSFmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F', 
                     offset=myoffset,shape=tmpnii.header.get_data_shape())
    maskCSF = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    del data
    tmpnii = nib.load(GMmaskFileout)
    myoffset = tmpnii.header.sizeof_hdr + 4 + tmpnii.header.get_data_offset()
    data = np.memmap(GMmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F', 
                     offset=myoffset,shape=tmpnii.header.get_data_shape())
    maskGM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    del data
    maskAll = np.logical_or(np.logical_or(maskWM, maskCSF), maskGM)
    maskWM_ = maskWM[maskAll]
    maskCSF_ = maskCSF[maskAll]
    maskGM_ = maskGM[maskAll]


    return maskAll, maskWM_, maskCSF_, maskGM_

def buildpath(subject,fmriRun):
    #return op.join(DATADIR, subject,'MNINonLinear','Results',fmriRun)
    return 'test'

def testpath(subject,fmriRun):
    return op.join(DATADIR, 'Testing', subject,'Results',fmriRun)


# ### Parameters

# In[106]:

subject = '734045'
fmriRun = 'rfMRI_REST1_LR'
fmriFile = 'test/rfMRI_REST1_LR.nii.gz'
behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
release = 'Q2'
outScore = 'PMAT24_A_CR'
DATADIR = '/media/paola/HCP/'
PARCELDIR = '/home/paola/parcellations'
parcellation = 'shenetal_neuroimage2013'
overwrite = False
thisRun = 'rfMRI_REST1'
isDataClean = True
doPlot = True
useLegendre = True
nPoly = 4
queue = False
useFeat = False
preproOnly = True
doTsmooth = True
normalize = 'zscore'
isCifti = False
keepMean = False

if thisRun == 'rfMRI_REST1':
    outMat = 'rest_1_mat'
elif thisRun == 'rfMRI_REST2':
    outMat = 'rest_1_mat'
else:
    sys.exit("Invalid run code")  
    
suffix = '_hp2000_clean' if isDataClean else ''   


# ### Pipeline definition

# The pipeline workflow is defined by two dictionaries. 
# 
# The dictionary <b>Operations</b> encodes the order of generic pipeline steps, with 0 for skipping an operation, and otherwise a number indicating when the operation should be performed. Note that several operations may have the same position (e.g., motion regression and tissue regression may both have order = 3, which means they should be performed in the same regression).
# 
# The dictionary <b>Flavors</b> encodes the flavor of each step.

# #### Finn's pipeline

# In[241]:

Operations={
    'MotionRegression'       : 4,
    'Scrubbing'              : 0,
    'TissueRegression'       : 3,
    'DetrendingWMCSF'        : 2,
    'DetrendingGM'           : 6, 
    'SpatialSmoothing'       : 0,
    'TemporalFiltering'      : 0,
    'ICAdenoising'           : 0,
    'GlobalSignalRegression' : 7,
    'VoxelNormalization'     : 1,
    'TemporalSmoothing'      : 5,
}

Flavors={
    'MotionRegression'       : '[R R^2]' ,
    'Scrubbing'              : 'fd_0.2',
    'TissueRegression'       : 'CompCor_5',
    'DetrendingWMCSF'        : 'legendre_3',
    'DetrendingGM'           : 'legendre_3',  
    'SpatialSmoothing'       : 'Gaussian_6',
    'TemporalFiltering'      : 'DFT_0.009_0.1',  
    'ICAdenoising'           : 'ICAFIX',
    'GlobalSignalRegression' : '',  
    'VoxelNormalization'     : 'zscore',
    'TemporalSmoothing'      : 'Gaussian_1',
}


# ### Pipeline setup

# Every step is associated with a function.

# In[239]:

def MotionRegression(niiImg, flavor):
    motionFile = op.join(buildpath(subject,fmriRun), 'Movement_Regressors_dt.txt')
    colNames = ['mmx','mmy','mmz','degx','degy','degz','dmmx','dmmy','dmmz','ddegx','ddegy','ddegz']
    df = pd.read_csv(motionFile,delim_whitespace=True,header=None)
    df.columns = colNames
    for iCol in range(len(colNames)):
        with open(motionFile.replace('.txt','_'+colNames[iCol]+'.txt'),'w') as tmp:
            df.to_csv(path_or_buf=tmp,sep='\n',columns=[colNames[iCol]],header=None,index=False)
    X = np.empty((nTRs, 0)) 
    for iCol in range(len(colNames)):
        X = np.concatenate((X,np.loadtxt(motionFile.replace('.txt','_'+colNames[iCol]+'.txt'))[:,np.newaxis]),axis=1)
    return X

def Scrubbing(niiImg, flavor):
    print 'Scrubbing : '+flavor    

def TissueRegression(niiImg, flavor):
    if isCifti:
	niiImgGM = niiImg		
    else:
	niiImgGM = niiImg[maskGM_,:]

   
    meanWM = np.mean(np.float64(niiImg[maskWM_,:]),axis=0)
    meanWM = meanWM - np.mean(meanWM)
    meanWM = meanWM/max(meanWM)
    meanCSF = np.mean(np.float64(niiImg[maskCSF_,:]),axis=0)
    meanCSF = meanCSF - np.mean(meanCSF)
    meanCSF = meanCSF/max(meanCSF)
    X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)
    niiImgGM = regress(niiImgGM, nTRs, X, keepMean)
    if not isCifti:
	niiImg[maskGM_,:] = niiImgGM
    else:
	niiImg = niiImgGM
    return niiImg

def DetrendingWMCSF(niiImg, flavor):
    if flavor == 'legendre_3':
        y = legendre_poly(3,nTRs)
        niiImgWMCSF = niiImg[np.logical_or(maskWM_,maskCSF_),:]    
        niiImgWMCSF = regress(niiImgWMCSF, nTRs, y.T, keepMean)
        niiImg[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
    return niiImg 

def DetrendingGM(niiImg, flavor):
    if isCifti:
        niiImgGM = niiImg
    else:
        niiImgGM = niiImg[maskGM_,:]
    if flavor == 'legendre_3':
        y = legendre_poly(3,nTRs)
        niiImgGM = regress(niiImgGM, nTRs, y.T, keepMean)

    if not isCifti:
        niiImg[maskGM_,:] = niiImgGM
    else:
        niiImg = niiImgGM
        
    return niiImg 
    

def SpatialSmoothing(niiImg, flavor):
    print 'SpatialSmoothing : '+flavor    

def TemporalFiltering(niiImg, flavor):
    print 'TemporalFiltering : '+flavor    
    
def ICAdenoising(niiImg, flavor):
    print 'ICAdenoising : '+flavor

def GlobalSignalRegression(niiImg, flavor):
    meanAll = np.mean(niiImg,axis=0)
    meanAll = meanAll - np.mean(meanAll)
    meanAll = meanAll/max(meanAll)
    return meanAll[:,np.newaxis]

def VoxelNormalization(niiImg, flavor):
    if flavor == 'zscore':
        niiImg = stats.zscore(niiImg, axis=1, ddof=1)
        return niiImg
    elif flavor == 'pcSigCh':
        niiImg = 100 * (niiImg - np.mean(niiImg,axis=1)[:,np.newaxis]) / np.mean(niiImg,axis=1)[:,np.newaxis]
    else:
        print 'Warning! Wrong normalization flavor. Nothing was done'
    return niiImg  

def TemporalSmoothing(niiImg, flavor):
    w = signal.gaussian(11,std=1)
    niiImg = signal.lfilter(w,1,niiImg)
    return niiImg


# In[243]:

Hooks={
    'MotionRegression'       : MotionRegression,
    'Scrubbing'              : Scrubbing,
    'TissueRegression'       : TissueRegression,
    'DetrendingWMCSF'        : DetrendingWMCSF,
    'DetrendingGM'           : DetrendingGM,
    'SpatialSmoothing'       : SpatialSmoothing,
    'TemporalFiltering'      : TemporalFiltering,  
    'ICAdenoising'           : ICAdenoising,
    'GlobalSignalRegression' : GlobalSignalRegression,  
    'VoxelNormalization'     : VoxelNormalization,
    'TemporalSmoothing'      : TemporalSmoothing
}


# Operations are sorted according to the assigned order.

# In[245]:

sortedOperations = sorted(Operations.items(), key=operator.itemgetter(1))
sortedOperations


# In[246]:

steps = {}
cstep = 0
for opr in sortedOperations:
    if opr[1]==0:
        continue
    else:
        if opr[1]!=cstep:
            cstep=cstep+1
            steps[cstep] = [opr[0]]
        else:
            steps[cstep].append(opr[0])


# Steps are executed sequentially.

# In[ ]:

print 'Step 0'
print 'Building WM, CSF and GM masks...'
maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(subject,fmriRun,False)

print 'Loading data in memory...'
niiImg, nRows, nCols, nSlices, nTRs, affine = load_img(fmriFile)

for step in steps.items():    
    print 'Step '+str(step[0])+' '+str(step[1])
    if len(step[1]) == 1:
        if 'Regression' in step[1][0]:
	    if step[1][0]=='TissueRegression':
	        niiImg = Hooks[step[1][0]](niiImg, Flavors[step[1][0]])
	    else:
    	        r0 = Hooks[step[1][0]](niiImg, Flavors[step[1][0]])
    	        niiImg = regress(niiImg, nTRs, r0, normalize=='keepMean')
        else:
            niiImg = Hooks[step[1][0]](niiImg, Flavors[step[1][0]])
    else:
        r = np.empty((nTRs, 0))
        for opr in step[1]:
            if 'Regression' in opr[0]:
		if opr[0]=='TissueRegression':
		    niiImg = Hooks[opr](niiImg, Flavors[opr])
		else:    
                    r0 = Hooks[opr](niiImg, Flavors[opr])
                    r = np.append(r, r0, axis=1)
            else:
                niiImg = Hooks[opr](niiImg, Flavors[opr])
        if r.shape[1] > 0:
       	    niiImg = regress(niiImg, nTRs, r, normalize=='keepMean')    
    niiImg[np.isnan(niiImg)] = 0
    
    niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
    niiimg[maskAll,:] = niiImg
    niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
    newimg = nib.Nifti1Image(niiimg, affine)
    nib.save(newimg,'test/step{}p.nii.gz'.format(step[0])) 
    del niiimg            
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
    newimg = nib.Nifti1Image(niiimg, affine)
    nib.save(newimg,'test/outfile.nii.gz')
    del niiimg            

