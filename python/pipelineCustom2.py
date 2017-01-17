
# coding: utf-8

# ### Required libraries

# In[87]:

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
from nilearn.signal import clean
from nilearn.image import smooth_img
from nilearn.input_data import NiftiMasker
import scipy.linalg as linalg
import string
import random
import xml.etree.cElementTree as ET
from time import localtime, strftime


# ### Utils

# In[95]:

def regress(niiImg, nTRs, regressors, keepMean):
    print 'inside regress', niiImg.shape
    X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
    N = niiImg.shape[0]
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
    x = np.arange(nTRs)
    x = x - x.max()/2
    num_pol = range(order+1)
    y = np.ones((len(num_pol),len(x)))   
    coeff = np.eye(len(num_pol))
    # Print out text file for each polynomial to be used as a regressor
    for i in num_pol:
        myleg = Legendre(coeff[i])
        y[i,:] = myleg(x) 
        if i>0:
            y[i,:] = y[i,:] - np.mean(y[i,:])
            y[i,:] = y[i,:]/np.max(y[i,:])
        np.savetxt(op.join(buildpath(subject,fmriRun),
                           'poly_detrend_legendre' + str(i) + '.txt'), y[i,:] ,fmt='%.4f')
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
    TR = img.header.structarr['pixdim'][4]
    niiImg = data.reshape([nRows*nCols*nSlices, nTRs], order='F')
    niiImg = niiImg[maskAll,:]
    return niiImg, nRows, nCols, nSlices, nTRs, img.affine, TR

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
        
        flirt_ribbon = fsl.FLIRT(in_file=ribbonFilein, out_file=ribbonFileout,\
            reference=fmriFile, apply_xfm=True,\
            in_matrix_file='eye.mat', interp='nearestneighbour')
        flirt_ribbon.run()

        flirt_wmparc = fsl.FLIRT(in_file=wmparcFilein, out_file=wmparcFileout,\
            reference=fmriFile, apply_xfm=True,\
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


def extract_noise_components(niiImg, num_components=5, extra_regressors=None):
    """Largely based on https://github.com/nipy/nipype/blob/master/examples/
    rsfmri_vol_surface_preprocessing_nipy.py#L261
    Derive components most reflective of physiological noise according to
    aCompCor method (Behzadi 2007)
    Parameters
    ----------
    niiImg: raw data
    num_components: number of components to use for noise decomposition
    extra_regressors: additional regressors to add
    Returns
    -------
    components: n_time_points x regressors
    """
    niiImgWMCSF = niiImg[np.logical_or(maskWM_,maskCSF_),:] 
    
    niiImgWMCSF[np.isnan(np.sum(niiImgWMCSF, axis=1)), :] = 0
    # remove mean and normalize by variance
    # voxel_timecourses.shape == [nvoxels, time]
    X = niiImgWMCSF.T
    stdX = np.std(X, axis=0)
    stdX[stdX == 0] = 1.
    stdX[np.isnan(stdX)] = 1.
    stdX[np.isinf(stdX)] = 1.
    X = (X - np.mean(X, axis=0)) / stdX
    u, _, _ = linalg.svd(X, full_matrices=False)
    components = u[:, :num_components]
    
    if extra_regressors:
        components = np.hstack((components, regressors))

    return components

def conf2XML(inFile, dataDir, operations, startTime, endTime, fname):
    doc = ET.Element("pipeline")
    
    nodeInput = ET.SubElement(doc, "input")
    nodeInFile = ET.SubElement(nodeInput, "inFile")
    nodeInFile.text = inFile
    nodeDataDir = ET.SubElement(nodeInput, "dataDir")
    nodeDataDir.text = dataDir
    
    nodeDate = ET.SubElement(doc, "date")
    nodeDay = ET.SubElement(nodeDate, "day")
    day = strftime("%Y-%m-%d", localtime())
    nodeDay.text = day
    stime = strftime("%H:%M:%S", startTime)
    etime = strftime("%H:%M:%S", endTime)
    nodeStart = ET.SubElement(nodeDate, "timeStart")
    nodeStart.text = stime
    nodeEnd = ET.SubElement(nodeDate, "timeEnd")
    nodeEnd.text = etime
    
    nodeSteps = ET.SubElement(doc, "steps")
    for op in operations:
        if op[1] == 0: continue
        nodeOp = ET.SubElement(nodeSteps, "operation", name=op[0])
        nodeOrder = ET.SubElement(nodeOp, "order")
        nodeOrder.text = str(op[1])
        nodeFlavor = ET.SubElement(nodeOp, "flavor")
        nodeFlavor.text = str(op[2])
    tree = ET.ElementTree(doc)
    tree.write(fname)


# ### Parameters

# In[84]:

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
queue = False
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

# customize path to get access to single runs
def buildpath(subject,fmriRun):
    return 'test'


# In[101]:

fmriFile = 'test/rfMRI_REST1_LR.nii.gz'
subject = '734045'
fmriRun = 'rfMRI_REST1_LR'
rstring = ''.join(random.SystemRandom().choice(string.ascii_lowercase +string.ascii_uppercase + string.digits) for _ in range(8))
outFile = fmriRun+'_'+rstring
outXML = rstring+'.xml'


# ### Pipeline definition

# The pipeline workflow is defined by two dictionaries. 
# 
# The struct <b>Operations</b> encodes the order of generic pipeline steps, with 0 for skipping an operation, and otherwise a number indicating when the operation should be performed. Note that several operations may have the same position (e.g., motion regression and tissue regression may both have order = 3, which means they should be performed in the same regression). For each operation an array encodes the flavor of each step and parameters when needed.

# #### Finn's pipeline

# In[32]:

Operations= [
    ['VoxelNormalization',      1, ['zscore']],
    ['Detrending',              2, ['legendre', 3, 'WMCSF']],
    ['TissueRegression',        3, ['WMCSF']],
    ['MotionRegression',        4, ['[R dR]']],
    ['TemporalFiltering',       5, ['Gaussian', 1]],
    ['Detrending',              6, ['legendre', 3,'GM']],
    ['GlobalSignalRegression',  7, []],
    ['Scrubbing',               0, ['fd', 0.2]],
    ['SpatialSmoothing',        0, ['Gaussian', 6]],
    ['ICAdenoising',            0, ['ICAFIX']],
]


# ### Pipeline setup

# Every step is associated with a function.

# In[71]:

def MotionRegression(niiImg, flavor):
    motionFile = op.join(buildpath(subject,fmriRun), 'Movement_Regressors_dt.txt')
    data = np.genfromtxt(motionFile)
    if flavor[0] == 'R':
        X = data[:,:6]
    elif flavor[0] == 'R dR':
        X = data
    elif flavor[0] == 'R R^2':
        data_squared = data ** 2
        X = np.concatenate((data, data_squared), axis=1)
    elif flavor[0] == 'R R^2 R-1 R-1^2':
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared), axis=1)
    elif flavor[0] == 'R R^2 R-1 R-1^2 R-2 R-2^2':
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        data_roll2 = np.roll(data_roll, 1, axis=0)
        data_roll2[0] = 0
        data_roll2_squared = data_roll2 ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared, data_roll2, data_roll2_squared), axis=1)
    else:
        'Wrong flavor, using default regressors: R dR'
        X = data   
    
    return X

def Scrubbing(niiImg, flavor):
    print 'Scrubbing : '+flavor    

def TissueRegression(niiImg, flavor):
    if isCifti:
        niiImgGM = niiImg
    else:
        niiImgGM = niiImg[maskGM_,:]
        
    if flavor[0] == 'CompCor':
        X = extract_noise_components(niiImg, num_components=flavor[1])
        niiImgGM = regress(niiImgGM, nTRs, X, keepMean)
    elif flavor[0] == 'WMCSF':
        meanWM = np.mean(np.float64(niiImg[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float64(niiImg[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)
        niiImgGM = regress(niiImgGM, nTRs, X, keepMean)
    else:
        print 'Warning! Wrong flavor. Nothing was done'    
    
    if not isCifti:
        niiImg[maskGM_,:] = niiImgGM
    else:
        niiImg = niiImgGM
    return niiImg

def Detrending(niiImg, flavor):
    if flavor[2] == 'WMCSF':
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1],nTRs)
	    np.savetxt(op.join(buildpath(subject,fmriRun),
                       'poly_detrend_legendrei_WMCSF_b.txt'), y ,fmt='%.4f')
	    y = np.round(y, decimals=4)
	    np.savetxt(op.join(buildpath(subject,fmriRun),
                       'poly_detrend_legendrei_WMCSF_a.txt'), y ,fmt='%.4f')
            niiImgWMCSF = niiImg[np.logical_or(maskWM_,maskCSF_),:]    
            niiImgWMCSF = regress(niiImgWMCSF, nTRs, y[1:4,:].T, keepMean)
            niiImg[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            nPoly = flavor[0] + 1
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:]) 
	    y = np.round(y, decimals=4)
            niiImgWMCSF = niiImg[np.logical_or(maskWM_,maskCSF_),:]    
            niiImgWMCSF = regress(niiImgWMCSF, nTRs, y[1:4,:].T, keepMean)
            niiImg[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
        else:
            print 'Warning! Wrong detrend flavor. Nothing was done'    
    elif flavor[2] == 'GM':
        if isCifti:
            niiImgGM = niiImg
        else:
            niiImgGM = niiImg[maskGM_,:]

        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1], nTRs)
	    y = np.round(y, decimals=4)
            niiImgGM = regress(niiImgGM, nTRs, y[1:4,:].T, keepMean)
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            nPoly = flavor[0] + 1
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])
            niiImgGM = regress(niiImgGM, nTRs, y[1:4,:].T, keepMean)
        else:
            print 'Warning! Wrong detrend flavor. Nothing was done'

        if not isCifti:
            niiImg[maskGM_,:] = niiImgGM
        else:
            niiImg = niiImgGM
    else:
        print 'Warning! Wrong detrend mask. Nothing was done' 
    return niiImg     

   

def SpatialSmoothing(niiImg, flavor):
    niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
    niiimg[maskAll,:] = niiImg
    niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
    newimg = nib.Nifti1Image(niiimg, affine)
    if flavor[0] == 'Gaussian':
        newimg = smooth_img(newimg, flavor[1])
        niiimg = np.reshape(newimg.get_data(), (nRows*nCols*nSlices, nTRs), order='F')
        niiImg = niiimg[maskAll,:]
    elif flavor[0] == 'GaussianGM':
        GMmaskFile = op.join(buildpath(subject,fmriRun),'GMmask.nii')
        NiftiMasker(mask_img=GMmaskFile, sessions=None, smoothing_fwhm=flavor[1])
        niiImg[maskGM_,:] = masker.fit_transform(newimg).T
    else:
        print 'Warning! Wrong smoothing flavor. Nothing was done'
    return niiImg  

def TemporalFiltering(niiImg, flavor):
    if flavor[0] == 'Butter':
        niiImg = clean(niiImg.T, detrend=False, standardize=False, 
                              t_r=TR, high_pass=flavor[1], low_pass=flavor[2]).T
    elif flavor[0] == 'Gaussian':
        w = signal.gaussian(11,std=1)
        niiImg = signal.lfilter(w,1,niiImg)
    else:
        print 'Warning! Wrong temporal filtering flavor. Nothing was done'    
    return niiImg    
    
def ICAdenoising(niiImg, flavor):
    print 'ICAdenoising : '+flavor

def GlobalSignalRegression(niiImg, flavor):
    meanAll = np.mean(niiImg,axis=0)
    meanAll = meanAll - np.mean(meanAll)
    meanAll = meanAll/max(meanAll)
    return meanAll[:,np.newaxis]

def VoxelNormalization(niiImg, flavor):
    if flavor[0] == 'zscore':
        niiImg = stats.zscore(niiImg, axis=1, ddof=1)
        return niiImg
    elif flavor[0] == 'pcSigCh':
        niiImg = 100 * (niiImg - np.mean(niiImg,axis=1)[:,np.newaxis]) / np.mean(niiImg,axis=1)[:,np.newaxis]
    else:
        print 'Warning! Wrong normalization flavor. Nothing was done'
    return niiImg  


Hooks={
    'MotionRegression'       : MotionRegression,
    'Scrubbing'              : Scrubbing,
    'TissueRegression'       : TissueRegression,
    'Detrending'             : Detrending,
    'SpatialSmoothing'       : SpatialSmoothing,
    'TemporalFiltering'      : TemporalFiltering,  
    'ICAdenoising'           : ICAdenoising,
    'GlobalSignalRegression' : GlobalSignalRegression,  
    'VoxelNormalization'     : VoxelNormalization,
}


# Operations are sorted according to the assigned order.

# In[34]:

sortedOperations = sorted(Operations, key=operator.itemgetter(1))
sortedOperations


# In[44]:

steps = {}
Flavors = {}
cstep = 0
for opr in sortedOperations:
    if opr[1]==0:
        continue
    else:
        if opr[1]!=cstep:
            cstep=cstep+1
            steps[cstep] = [opr[0]]
            Flavors[cstep] = [opr[2]]
        else:
            steps[cstep].append(opr[0])
            Flavors[cstep].append(opr[2])


# Steps are executed sequentially.

# In[ ]:

def runPipeline(subject, fmriRun, fmriFile):
    
    timeStart = localtime()
    sortedOperations = sorted(Operations, key=operator.itemgetter(1))
    steps = {}
    Flavors = {}
    cstep = 0
    for opr in sortedOperations:
        if opr[1]==0:
            continue
        else:
            if opr[1]!=cstep:
                cstep=cstep+1
                steps[cstep] = [opr[0]]
                Flavors[cstep] = [opr[2]]
            else:
                steps[cstep].append(opr[0])
                Flavors[cstep].append(opr[2])


    print 'Step 0'
    print 'Building WM, CSF and GM masks...'
    global maskAll, maskWM_, maskCSF_, maskGM_
    maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(subject,fmriRun,False)

    print 'Loading data in memory...'
    global niiImg, nRows, nCols, nSlices, nTRs, affine
    niiImg, nRows, nCols, nSlices, nTRs, affine, TR = load_img(fmriFile)
    nsteps = len(steps)
    for i in range(1,nsteps+1):
        step = steps[i]
        print 'Step '+str(i)+' '+str(step[0])
        if len(step) == 1:
            if 'Regression' in step[0]:
                if step[0]=='TissueRegression':
                    niiImg = Hooks[step[0]](niiImg, Flavors[i][0])
		    print 'sono qui'
                else:
                    r0 = Hooks[step[0]](niiImg, Flavors[i][0])
                    niiImg = regress(niiImg, nTRs, r0, keepMean)
            else:
                niiImg = Hooks[step[0]](niiImg, Flavors[i][0])
        else:
            r = np.empty((nTRs, 0))
            for j in range(len(step)):
                opr = step[j]
                if 'Regression' in opr:
                    if opr=='TissueRegression':
                        niiImg = Hooks[opr](niiImg, Flavors[i][j])
                    else:    
                        r0 = Hooks[opr](niiImg, Flavors[i][j])
                        r = np.append(r, r0, axis=1)
                else:
                    niiImg = Hooks[opr](niiImg, Flavors[i][j])
            if r.shape[1] > 0:
                niiImg = regress(niiImg, nTRs, r, keepMean)    
        niiImg[np.isnan(niiImg)] = 0
	niiimg = np.zeros((nRows*nCols*nSlices, nTRs))
        niiimg[maskAll,:] = niiImg
        niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
        newimg = nib.Nifti1Image(niiimg, affine)
        nib.save(newimg,'test/step{}p2.nii.gz'.format(i))
        del niiimg


    print 'Done! Copy the resulting file...'
    if isCifti:
        # write to text file
        np.savetxt(outFile+'.tsv',niiImg, delimiter='\t', fmt='%.6f')
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
        nib.save(newimg,outFile+'.nii.gz')
        del niiimg 

    timeEnd = localtime()  

    conf2XML(fmriFile, DATADIR, sortedOperations, timeStart, timeEnd, buildpath(subject,fmriRun)+outXML)

runPipeline(subject, fmriRun, fmriFile)