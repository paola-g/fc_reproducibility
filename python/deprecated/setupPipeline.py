from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import os.path as op
from os import mkdir, makedirs, getcwd, remove, listdir, environ
import scipy.stats as stats
import nipype.interfaces.fsl as fsl
from subprocess import call, check_output, CalledProcessError, getoutput
import nibabel as nib
import scipy.io as sio
from sklearn import cross_validation
from sklearn import linear_model
from numpy.polynomial.legendre import Legendre
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
from time import localtime, strftime, sleep
import socket
import fnmatch
import re
from past.utils import old_div
import os
# ### Parameters

# these functions allow Paola & Julien to run code locally with their own path definitions
def getDataDir(x):
    return {
        'csclprd': '/home/duboisjx/scratch/data/HCP/MRI',
        'esplmat': '/home/duboisjx/vault/data/HCP/MRI',
        'sculpin': '/data/jdubois/data/HCP/MRI',
    }.get(x, '/data/jdubois/data/HCP/MRI')    # default if x not found
def getParcelDir(x):
    return {
        'csclprd':'/home/duboisjx/scratch/data/parcellations/',
        'esplmat': '/home/duboisjx/vault/data/parcellations/',
        'sculpin': '/data/jdubois/data/parcellations/',
    }.get(x, '/data/pgaldi/parcellations/')    # default if x not found
HOST=socket.gethostname()
DATADIR=getDataDir(HOST[0:7])
PARCELDIR=getParcelDir(HOST[0:7])
print DATADIR
print HOST[0:7]

# customize path to get access to single runs
def buildpath(subject,fmriRun):
    return op.join(DATADIR, subject,'MNINonLinear','Results',fmriRun)

class config(object):
    behavFile    = op.join(DATADIR,'..','neuropsych','unrestricted_luckydjuju_11_17_2015_0_47_11.csv')
    release      = 'Q2'
    outScore     = 'PMAT24_A_CR'
    pipelineName = 'testPipeline'
    parcellation = 'shenetal_neuroimage2013_new'
    overwrite    = False
    thisRun      = 'rfMRI_REST1'
    isDataClean  = False
    doPlot       = False
    queue        = True
    isCifti 	 = False

# ### Pipeline definition

# The pipeline workflow is defined by two dictionaries. 
# 
# The struct <b>Operations</b> encodes the order of generic pipeline steps, with 0 for skipping an operation, and otherwise a number indicating when the operation should be performed. Note that several operations may have the same position (e.g., motion regression and tissue regression may both have order = 3, which means they should be performed in the same regression). For each operation an array encodes the flavor of each step and parameters when needed.

# Variable config.preWhitening controls if pre-whitening should performed at each regression step.

# #### Finn's pipeline

# In[32]:

# if set to True, pre-whitening is performed at each regression step
config.preWhitening = False

#Operations= [
#    ['VoxelNormalization',      1, ['demean']],
#    ['Detrending',              2, ['poly', 1, 'wholebrain']],
#    ['TissueRegression',        3, ['WMCSF','wholebrain']],
#    ['MotionRegression',        3, ['R R^2 R-1 R-1^2']],
#    ['TemporalFiltering',       4, ['Butter', 0.009, 0.08]],
#    ['Detrending',              0, ['legendre', 3,'GM']],
#    ['GlobalSignalRegression',  3, []],
#    ['Scrubbing',               5, ['FD', 0.2]],
#    ['SpatialSmoothing',        0, ['Gaussian', 6]],
#]

Operations= [
    ['VoxelNormalization',      1, ['demean']],
    ['Detrending',              2, ['poly', 3, 'GM']],
    ['TissueRegression',        2, ['WMCSF+dt+sq','wholebrain']],
    ['MotionRegression',        3, ['ICA-AROMA', 'aggr']],
    ['TemporalFiltering',       4, ['Butter', 0.009, 0.08]],
    ['GlobalSignalRegression',  2, ['GS+dt+sq']],
    ['Scrubbing',               3, ['FD', 0.2]],
    ['SpatialSmoothing',        0, ['Gaussian', 6]],
]


# ### Utils

# In[95]:

if config.queue: priority=-100

if config.thisRun == 'rfMRI_REST1':
    outMat = 'rest_1_mat'
elif config.thisRun == 'rfMRI_REST2':
    outMat = 'rest_2_mat'
else:
    sys.exit("Invalid run code")  
    
suffix = '_hp2000_clean' if config.isDataClean else ''   

# these variables are initialized here and used later in the pipeline, do not change
config.filtering = []
config.doScrubbing = False

if config.isCifti:
    config.ext = '.dtseries.nii'
else:
    config.ext = '.nii.gz'


# regressors: to filter, no. time points x no. regressors
def filter_regressors(regressors, filtering, nTRs, TR):
    if len(filtering)==0:
        print 'Error! Missing or wrong filtering flavor. Regressors were not filtered.'
    else:
        if filtering[0] == 'Butter':
            regressors = clean(regressors, detrend=False, standardize=False, 
                                  t_r=TR, high_pass=filtering[1], low_pass=filtering[2])
        elif filtering[0] == 'Gaussian':
            w = signal.gaussian(11,std=filtering[1])
            regressors = signal.lfilter(w,1,regressors, axis=0)  
    return regressors
    
def regress(niiImg, nTRs, TR, regressors, preWhitening=False):
    if preWhitening:
        W = prewhitening(niiImg, nTRs, TR, regressors)
        niiImg = np.dot(niiImg,W)
        regressors = np.dot(W,regressors)
    X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
    N = niiImg.shape[0]
    for i in range(N):
        fit = np.linalg.lstsq(X, niiImg[i,:].T)[0]
        fittedvalues = np.dot(X, fit)
        resid = niiImg[i,:] - np.ravel(fittedvalues)
        niiImg[i,:] = resid
    return niiImg 

def partial_regress(niiImg, nTRs, TR, regressors, partialIdx, preWhitening=False):    
    if preWhitening:
        W = prewhitening(niiImg, nTRs, TR, regressors)
        niiImg = np.dot(niiImg,W)
        regressors = np.dot(W,regressors)
    X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
    N = niiImg.shape[0]
    for i in range(N):
        fit = np.linalg.lstsq(X, niiImg[i,:].T)[0]
        fittedvalues = np.dot(X[:,partialIdx], fit[partialIdx])
        resid = niiImg[i,:] - fittedvalues
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
    coeff = np.eye(order+1)
    for i in num_pol:
        myleg = Legendre(coeff[i])
        y[i,:] = myleg(x) 
        if i>0:
            y[i,:] = y[i,:] - np.mean(y[i,:])
            y[i,:] = y[i,:]/np.max(y[i,:])
    return y

def load_img(fmriFile, maskAll):
    if config.isCifti:
        toUnzip = fmriFile.replace('_Atlas.dtseries.nii','.nii.gz')
        cmd = 'wb_command -cifti-convert -to-text {} {}'.format(fmriFile,op.join(buildpath(config.subject,config.fmriRun),'.tsv'))
        call(cmd,shell=True)
    else:
        toUnzip = fmriFile

    with open(toUnzip, 'rb') as fFile:
        decompressedFile = gzip.GzipFile(fileobj=fFile)
        outFilePath = op.join(buildpath(config.subject, config.fmriRun), config.fmriRun+'.nii')
        with open(outFilePath, 'wb') as outfile:
            outfile.write(decompressedFile.read())

    volFile = outFilePath

    img = nib.load(volFile)
    
    myoffset = img.dataobj.offset
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
        ribbonMat = op.join(buildpath(subject,fmriRun), 'ribbon_flirt.mat')
        wmparcMat = op.join(buildpath(subject,fmriRun), 'wmparc_flirt.mat')
        eyeMat = op.join(buildpath(subject,fmriRun), 'eye.mat')
        with open(eyeMat,'w') as fid:
            fid.write('1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1')
        
        flirt_ribbon = fsl.FLIRT(in_file=ribbonFilein, out_file=ribbonFileout, 
				reference=fmriFile, apply_xfm=True, 
				in_matrix_file=eyeMat, out_matrix_file=ribbonMat, interp='nearestneighbour')
        flirt_ribbon.run()

        flirt_wmparc = fsl.FLIRT(in_file=wmparcFilein, out_file=wmparcFileout, 
				reference=fmriFile, apply_xfm=True, 
				in_matrix_file=eyeMat, out_matrix_file=wmparcMat, interp='nearestneighbour')
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
        img = nib.Nifti1Image(WMmask.astype('<f4'), ref.affine)
        nib.save(img, WMmaskFileout)
        
        CSFmask = np.reshape(CSFmask,ref.shape)
        img = nib.Nifti1Image(CSFmask.astype('<f4'), ref.affine)
        nib.save(img, CSFmaskFileout)
        
        GMmask = np.reshape(GMmask,ref.shape)
        img = nib.Nifti1Image(GMmask.astype('<f4'), ref.affine)
        nib.save(img, GMmaskFileout)

        # delete temporary files
        cmd = 'rm {} {} {}'.format(eyeMat, ribbonMat, wmparcMat)
        call(cmd,shell=True)
        
        
    tmpnii = nib.load(WMmaskFileout)
    myoffset = tmpnii.dataobj.offset
    data = np.memmap(WMmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
                     offset=myoffset,shape=tmpnii.header.get_data_shape())
    nRows, nCols, nSlices = tmpnii.header.get_data_shape()
    maskWM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    del data
    tmpnii = nib.load(CSFmaskFileout)
    myoffset = tmpnii.dataobj.offset
    data = np.memmap(CSFmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F', 
                     offset=myoffset,shape=tmpnii.header.get_data_shape())
    maskCSF = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    del data
    tmpnii = nib.load(GMmaskFileout)
    myoffset = tmpnii.dataobj.offset
    data = np.memmap(GMmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F', 
                     offset=myoffset,shape=tmpnii.header.get_data_shape())
    maskGM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    del data
    maskAll = np.logical_or(np.logical_or(maskWM, maskCSF), maskGM)
    maskWM_ = maskWM[maskAll]
    maskCSF_ = maskCSF[maskAll]
    maskGM_ = maskGM[maskAll]


    return maskAll, maskWM_, maskCSF_, maskGM_

def extract_noise_components(niiImg, WMmask, CSFmask, num_components=5, flavor=None):
    """
    Largely based on https://github.com/nipy/nipype/blob/master/examples/
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
    if flavor == 'WMCSF' or flavor == None:
        niiImgWMCSF = niiImg[np.logical_or(WMmask,CSFmask),:] 

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
    elif flavor == 'WM+CSF':    
        niiImgWM = niiImg[WMmask,:] 
        niiImgWM[np.isnan(np.sum(niiImgWM, axis=1)), :] = 0
        niiImgCSF = niiImg[CSFmask,:] 
        niiImgCSF[np.isnan(np.sum(niiImgCSF, axis=1)), :] = 0
        # remove mean and normalize by variance
        # voxel_timecourses.shape == [nvoxels, time]
        X = niiImgWM.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0)) / stdX
        u, _, _ = linalg.svd(X, full_matrices=False)
        components = u[:, :num_components]
        X = niiImgCSF.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0)) / stdX
        u, _, _ = linalg.svd(X, full_matrices=False)
        components = np.hstack((components, u[:, :num_components]))
    else:
	print 'Warning! Wrong CompCor flavor. Nothing was done.'
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

def checkXML(inFile, operations, params, resDir):
    for xfile in listdir(resDir):
        if fnmatch.fnmatch(op.join(resDir,xfile), op.join(resDir,'????????.xml')):
            tree = ET.parse(op.join(resDir,xfile))
            root = tree.getroot()
            tvalue = root[0][0].text == inFile
            if not tvalue:
                continue
            for el in root[2]:
                tvalue = tvalue and (el.attrib['name'] in operations[int(el[0].text)])
                tvalue = tvalue and (el[1].text in [repr(param) for param in params[int(el[0].text)]])
                if not tvalue:
                    continue
                else:    
                    rcode = xfile.replace('.xml','')
                    return op.join(resDir,rcode,config.fmriRun+'_prepro'+config.ext)
    return None
    
def fnSubmitToCluster(strScript, strJobFolder, strJobUID, resources):
    specifyqueue = ''
    # clean up .o and .e
    tmpfname = op.join(strJobFolder,strJobUID)
    try:
        remove(tmpfname+'.e')       
    except OSError:
        pass
    try:
        remove(tmpfname+'.o')       
    except OSError:
        pass
   
    strCommand = 'qsub {} -cwd -V {} -N {} -e "{}" -o "{}" "{}"'.format(specifyqueue,resources,strJobUID,
                      op.join(strJobFolder,strJobUID+'.e'), op.join(strJobFolder,strJobUID+'.o'), strScript)
    # write down the command to a file in the job folder
    with open(op.join(strJobFolder,strJobUID+'.cmd'),'w+') as hFileID:
        hFileID.write(strCommand+'\n')
    # execute the command
    cmdOut = check_output(strCommand, shell=True)
    return cmdOut.split()[2]    

def _interpolate(a, b, fraction):
    """Returns the point at the given fraction between a and b, where
    'fraction' must be between 0 and 1.
    """
    return a + (b - a)*fraction;

def scoreatpercentile(a, per, limit=(), interpolation_method='fraction'):
    """
    This function is grabbed from scipy

    """
    values = np.sort(a, axis=0)
    if limit:
        values = values[(limit[0] <= values) & (values <= limit[1])]

    idx = per /100. * (values.shape[0] - 1)
    if (idx % 1 == 0):
        score = values[idx]
    else:
        if interpolation_method == 'fraction':
            score = _interpolate(values[int(idx)], values[int(idx) + 1],
                                 idx % 1)
        elif interpolation_method == 'lower':
            score = values[np.floor(idx)]
        elif interpolation_method == 'higher':
            score = values[np.ceil(idx)]
        else:
            raise ValueError("interpolation_method can only be 'fraction', " \
                             "'lower' or 'higher'")

    return score

def dctmtx(N):
    K=N
    n = range(N)
    C = np.zeros((len(n), K),dtype=np.float32)
    C[:,0] = np.ones((len(n)))/np.sqrt(N)
    doublen = [2*x+1 for x in n]
    for k in range(1,K):
        C[:,k] = np.sqrt(2/N)*np.cos([np.pi*x*(k-1)/(2*N) for x in doublen])        
    return C 
    
# from nipy
def orth(X, tol=1.0e-07):
    """
    
    Compute orthonormal basis for the column span of X.
    
    Rank is determined by zeroing all singular values, u, less
    than or equal to tol*u.max().
    INPUTS:
        X  -- n-by-p matrix
    OUTPUTS:
        B  -- n-by-rank(X) matrix with orthonormal columns spanning
              the column rank of X
    """

    B, u, _ = linalg.svd(X, full_matrices=False)
    nkeep = np.greater(u, tol*u.max()).astype(np.int).sum()
    return B[:,:nkeep]

def reml(sigma, components, design=None, n=1, niter=128,
         penalty_cov=np.exp(-32), penalty_mean=0):
    """
    Adapted from spm_reml.m
    ReML estimation of covariance components from sigma using design matrix.
    INPUTS:
        sigma        -- m-by-m covariance matrix
        components   -- q-by-m-by-m array of variance components
                        mean of sigma is modeled as a some over components[i]
        design       -- m-by-p design matrix whose effect is to be removed for
                        ReML. If None, no effect removed (???)
        n            -- degrees of freedom of sigma
        penalty_cov  -- quadratic penalty to be applied in Fisher algorithm.
                        If the value is a float, f, the penalty is
                        f * identity(m). If the value is a 1d array, this is
                        the diagonal of the penalty. 
        penalty_mean -- mean of quadratic penalty to be applied in Fisher
                        algorithm. If the value is a float, f, the location
                        is f * np.ones(m).
    OUTPUTS:
        C            -- estimated mean of sigma
        h            -- array of length q representing coefficients
                        of variance components
        cov_h        -- estimated covariance matrix of h
    """

    # initialise coefficient, gradient, Hessian

    Q = components
    PQ = np.zeros(Q.shape,dtype=np.float32)
    
    q = Q.shape[0]
    m = Q.shape[1]

    # coefficient
    h = np.array([np.diag(Q[i]).mean() for i in range(q)])

    ## SPM initialization
    ## h = np.array([np.any(np.diag(Q[i])) for i in range(q)]).astype(np.float)

    C = np.sum([h[i] * Q[i] for i in range(Q.shape[0])], axis=0)

    # gradient in Fisher algorithm
    
    dFdh = np.zeros(q,dtype=np.float32)

    # Hessian in Fisher algorithm
    dFdhh = np.zeros((q,q),dtype=np.float32)

    # penalty terms

    penalty_cov = np.asarray(penalty_cov)
    if penalty_cov.shape == ():
        penalty_cov = penalty_cov * np.identity(q)
    elif penalty_cov.shape == (q,):
        penalty_cov = np.diag(penalty_cov)
        
    penalty_mean = np.asarray(penalty_mean)
    if penalty_mean.shape == ():
        penalty_mean = np.ones(q) * penalty_mean
        
    # compute orthonormal basis of design space

    if design is not None:
        X = orth(design)
    else:
        X = None

    _iter = 0
    _F = np.inf
    
    while True:

        # Current estimate of mean parameter

        iC = linalg.inv(C + np.identity(m) / np.exp(32))

        # E-step: conditional covariance 

        if X is not None:
            iCX = np.dot(iC, X)
            Cq = linalg.inv(np.dot(X.T, iCX))
            P = iC - np.dot(iCX, np.dot(Cq, iCX.T))
        else:
            P = iC

        # M-step: ReML estimate of hyperparameters
 
        # Gradient dF/dh (first derivatives)
        # Expected curvature (second derivatives)

        U = np.identity(m) - np.dot(P, sigma) / n

        for i in range(q):
            PQ[i] = np.dot(P, Q[i])
            dFdh[i] = -(PQ[i] * U).sum() * n / 2

            for j in range(i+1):
                dFdhh[i,j] = -(PQ[i]*PQ[j]).sum() * n / 2
                dFdhh[j,i] = dFdhh[i,j]
                
        # Enforce penalties:

        dFdh  = dFdh  - np.dot(penalty_cov, h - penalty_mean)
        dFdhh = dFdhh - penalty_cov

        dh = linalg.solve(dFdhh, dFdh)
        h -= dh
        C = np.sum([h[i] * Q[i] for i in range(Q.shape[0])], axis=0)
        
        df = (dFdh * dh).sum()
        if np.fabs(df) < 1.0e-01:
            break

        _iter += 1
        if _iter >= niter:
            break

    return C, h, -dFdhh

def sqrtm(V):
    u, s, _  = linalg.svd(V)
    s = np.sqrt(np.abs(np.diag(s)))
    m = s.shape[0]
    return np.dot(u, np.dot(s, u.T))

def prewhitening(niiImg, nTRs, TR, X):
    T = np.arange(nTRs) * TR
    d = 2 ** (np.floor(np.arange(np.log2(TR/4), 7)))
    Q = np.array([linalg.toeplitz((T**j)*np.exp(-T/d[i])) for i in range(len(d)) for j in [0,1]])
    CY = np.cov(niiImg.T)
    V, h, _ = reml(CY, Q, design=X, n=1)
    W = linalg.inv(sqrtm(V))
    return W

def get_rcode(mystring):
    if not config.isCifti:
        rcode = re.search('.*/(........)/.*\.nii.gz', mystring).group(1)
    else:
        rcode = re.search('.*/(........)/.*\.dtseries.nii', mystring).group(1)
    return rcode

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.read)
    return int(sum( buf.count(b'\n') for buf in f_gen ))

"""
The following functions implement the ICA-AROMA algorithm (Pruim et al. 2015) 
and are adapted from https://github.com/rhr-pruim/ICA-AROMA
"""

def feature_time_series(melmix, mc):
    """ This function extracts the maximum RP correlation feature scores. 
    It determines the maximum robust correlation of each component time-series 
    with a model of 72 realigment parameters.

    Parameters
    ---------------------------------------------------------------------------------
    melmix:     Full path of the melodic_mix text file
    mc:     Full path of the text file containing the realignment parameters
    
    Returns
    ---------------------------------------------------------------------------------
    maxRPcorr:  Array of the maximum RP correlation feature scores for the components of the melodic_mix file"""

    # Read melodic mix file (IC time-series), subsequently define a set of squared time-series
    mix = np.loadtxt(melmix)
    mixsq = np.power(mix,2)

    # Read motion parameter file
    RP6 = np.loadtxt(mc)[:,:6]

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    RP6_der = np.array(RP6[list(range(1,RP6.shape[0])),:] - RP6[list(range(0,RP6.shape[0]-1)),:])
    RP6_der = np.concatenate((np.zeros((1,6)),RP6_der),axis=0)

    # Create an RP-model including the RPs and its derivatives
    RP12 = np.concatenate((RP6,RP6_der),axis=1)

    # Add the squared RP-terms to the model
    RP24 = np.concatenate((RP12,np.power(RP12,2)),axis=1)

    # Derive shifted versions of the RP_model (1 frame for and backwards)
    RP24_1fw = np.concatenate((np.zeros((1,24)),np.array(RP24[list(range(0,RP24.shape[0]-1)),:])),axis=0)
    RP24_1bw = np.concatenate((np.array(RP24[list(range(1,RP24.shape[0])),:]),np.zeros((1,24))),axis=0)

    # Combine the original and shifted mot_pars into a single model
    RP_model = np.concatenate((RP24,RP24_1fw,RP24_1bw),axis=1)

    # Define the column indices of respectively the squared or non-squared terms
    idx_nonsq = np.array(np.concatenate((list(range(0,12)), list(range(24,36)), list(range(48,60))),axis=0))
    idx_sq = np.array(np.concatenate((list(range(12,24)), list(range(36,48)), list(range(60,72))),axis=0))

    # Determine the maximum correlation between RPs and IC time-series
    nSplits=int(1000)
    maxTC = np.zeros((nSplits,mix.shape[1]))
    for i in range(0,nSplits):
        # Get a random set of 90% of the dataset and get associated RP model and IC time-series matrices
        idx = np.array(random.sample(list(range(0,mix.shape[0])),int(round(0.9*mix.shape[0]))))
        RP_model_temp = RP_model[idx,:]
        mix_temp = mix[idx,:]
        mixsq_temp = mixsq[idx,:]

        # Calculate correlation between non-squared RP/IC time-series
        RP_model_nonsq = RP_model_temp[:,idx_nonsq]
        cor_nonsq = np.array(np.zeros((mix_temp.shape[1],RP_model_nonsq.shape[1])))
        for j in range(0,mix_temp.shape[1]):
            for k in range(0,RP_model_nonsq.shape[1]):
                cor_temp = np.corrcoef(mix_temp[:,j],RP_model_nonsq[:,k])
                cor_nonsq[j,k] = cor_temp[0,1]

        # Calculate correlation between squared RP/IC time-series
        RP_model_sq = RP_model_temp[:,idx_sq]
        cor_sq = np.array(np.zeros((mix_temp.shape[1],RP_model_sq.shape[1])))
        for j in range(0,mixsq_temp.shape[1]):
            for k in range(0,RP_model_sq.shape[1]):
                cor_temp = np.corrcoef(mixsq_temp[:,j],RP_model_sq[:,k])
                cor_sq[j,k] = cor_temp[0,1]

        # Combine the squared an non-squared correlation matrices
        corMatrix = np.concatenate((cor_sq,cor_nonsq),axis=1)

        # Get maximum absolute temporal correlation for every IC
        corMatrixAbs = np.abs(corMatrix)
        maxTC[i,:] = corMatrixAbs.max(axis=1)

    # Get the mean maximum correlation over all random splits
    maxRPcorr = maxTC.mean(axis=0)

    # Return the feature score
    return maxRPcorr

def feature_frequency(melFTmix, TR):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function extracts the high-frequency content feature scores. 
    It determines the frequency, as fraction of the Nyquist frequency, 
    at which the higher and lower frequencies explain half of the total power between 0.01Hz and Nyquist. 
    
    Parameters
    ---------------------------------------------------------------------------------
    melFTmix:   Full path of the melodic_FTmix text file
    TR:     TR (in seconds) of the fMRI data (float)
    
    Returns
    ---------------------------------------------------------------------------------
    HFC:        Array of the HFC ('High-frequency content') feature scores for the components of the melodic_FTmix file"""

    
    # Determine sample frequency
    Fs = old_div(1,TR)

    # Determine Nyquist-frequency
    Ny = old_div(Fs,2)
        
    # Load melodic_FTmix file
    FT=np.loadtxt(melFTmix)

    # Determine which frequencies are associated with every row in the melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
    f = Ny*(np.array(list(range(1,FT.shape[0]+1))))/(FT.shape[0])

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where( f > 0.01 )))
    FT=FT[fincl,:]
    f=f[fincl]

    # Set frequency range to [0-1]
    f_norm = old_div((f-0.01),(Ny-0.01))

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = old_div(np.cumsum(FT,axis=0), np.sum(FT,axis=0))

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    idx_cutoff=np.argmin(np.abs(fcumsum_fract-0.5),axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    HFC = f_norm[idx_cutoff]
         
    # Return feature score
    return HFC

def feature_spatial(fslDir, tempDir, aromaDir, melIC):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function extracts the spatial feature scores. 
    For each IC it determines the fraction of the mixture modeled thresholded Z-maps 
    respecitvely located within the CSF or at the brain edges, using predefined standardized masks.

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    tempDir:    Full path of a directory where temporary files can be stored (called 'temp_IC.nii.gz')
    aromaDir:   Full path of the ICA-AROMA directory, containing the mask-files (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz) 
    melIC:      Full path of the nii.gz file containing mixture-modeled threholded (p>0.5) Z-maps, registered to the MNI152 2mm template
    
    Returns
    ---------------------------------------------------------------------------------
    edgeFract:  Array of the edge fraction feature scores for the components of the melIC file
    csfFract:   Array of the CSF fraction feature scores for the components of the melIC file"""

    # Get the number of ICs
    numICs = int(getoutput('%sfslinfo %s | grep dim4 | head -n1 | awk \'{print $2}\'' % (fslDir, melIC) ))

    # Loop over ICs
    edgeFract=np.zeros(numICs)
    csfFract=np.zeros(numICs)
    for i in range(0,numICs):
        # Define temporary IC-file
        tempIC = op.join(tempDir,'temp_IC.nii.gz')

        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        os.system(' '.join([op.join(fslDir,'fslroi'),
            melIC,
            tempIC,
            str(i),
            '1']))

        # Change to absolute Z-values
        os.system(' '.join([op.join(fslDir,'fslmaths'),
            tempIC,
            '-abs',
            tempIC]))
        
        # Get sum of Z-values within the total Z-map (calculate via the mean and number of non-zero voxels)
        totVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-V | awk \'{print $1}\''])))
        
        if not (totVox == 0):
            totMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-M'])))
        else:
            print '     - The spatial map of component ' + str(i+1) + ' is empty. Please check!'
            totMean = 0

        totSum = totMean * totVox
        
        # Get sum of Z-values of the voxels located within the CSF (calculate via the mean and number of non-zero voxels)
        csfVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k mask_csf.nii.gz',
                            '-V | awk \'{print $1}\''])))

        if not (csfVox == 0):
            csfMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k mask_csf.nii.gz',
                            '-M'])))
        else:
            csfMean = 0

        csfSum = csfMean * csfVox   

        # Get sum of Z-values of the voxels located within the Edge (calculate via the mean and number of non-zero voxels)
        edgeVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k mask_edge.nii.gz',
                            '-V | awk \'{print $1}\''])))
        if not (edgeVox == 0):
            edgeMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k mask_edge.nii.gz',
                            '-M'])))
        else:
            edgeMean = 0
        
        edgeSum = edgeMean * edgeVox

        # Get sum of Z-values of the voxels located outside the brain (calculate via the mean and number of non-zero voxels)
        outVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k mask_out.nii.gz',
                            '-V | awk \'{print $1}\''])))
        if not (outVox == 0):
            outMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k mask_out.nii.gz',
                            '-M'])))
        else:
            outMean = 0
        
        outSum = outMean * outVox

        # Determine edge and CSF fraction
        if not (totSum == 0):
            edgeFract[i] = old_div((outSum + edgeSum),(totSum - csfSum))
            csfFract[i] = old_div(csfSum, totSum)
        else:
            edgeFract[i]=0
            csfFract[i]=0

    # Remove the temporary IC-file
    remove(tempIC)

    # Return feature scores
    return edgeFract, csfFract

def classification(outDir, maxRPcorr, edgeFract, HFC, csfFract):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function classifies a set of components into motion and non-motion 
    components based on four features; maximum RP correlation, high-frequency content, 
    edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    outDir:     Full path of the output directory
    maxRPcorr:  Array of the 'maximum RP correlation' feature scores of the components
    edgeFract:  Array of the 'edge fraction' feature scores of the components
    HFC:        Array of the 'high-frequency content' feature scores of the components
    csfFract:   Array of the 'CSF fraction' feature scores of the components

    Return
    ---------------------------------------------------------------------------------
    motionICs   Array containing the indices of the components identified as motion components

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    classified_motion_ICs.txt   A text file containing the indices of the components identified as motion components """

    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
    
    # Project edge & maxRPcorr feature scores to new 1D space
    x = np.array([maxRPcorr, edgeFract])
    proj = hyp[0] + np.dot(x.T,hyp[1:])

    # Classify the ICs
    motionICs = np.squeeze(np.array(np.where((proj > 0) + (csfFract > thr_csf) + (HFC > thr_HFC))))

    return motionICs

# ### Pipeline setup

# Every step is associated with a function.

# In[71]:

def MotionRegression(niiImg, flavor, masks, imgInfo):
    # assumes that data is organized as in the HCP
    motionFile = op.join(buildpath(config.subject,config.fmriRun), 'Movement_Regressors_dt.txt')
    data = np.genfromtxt(motionFile)
    if flavor[0] == 'R':
        X = data[:,:6]
    elif flavor[0] == 'R dR':
        X = data
    elif flavor[0] == 'R R^2':
        data = data[:,:6]
        data_squared = data ** 2
        X = np.concatenate((data, data_squared), axis=1)
    elif flavor[0] == 'R R^2 R-1 R-1^2':
        data = data[:,:6]
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared), axis=1)
    elif flavor[0] == 'R dR R^2 dR^2':
        data_squared = data ** 2
        X = np.concatenate((data, data_squared), axis=1)
    elif flavor[0] == 'R R^2 R-1 R-1^2 R-2 R-2^2':
        data = data[:,:6]
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        data_roll2 = np.roll(data_roll, 1, axis=0)
        data_roll2[0] = 0
        data_roll2_squared = data_roll2 ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared, data_roll2, data_roll2_squared), axis=1)
    elif flavor[0] == 'ICA-AROMA':
	nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        fslDir = op.join(environ["FSLDIR"],'bin','')
        icaOut = op.join(buildpath(config.subject,config.fmriRun), 'rfMRI_REST1_LR_hp2000.ica','filtered_func_data.ica')
        melIC_MNI = op.join(icaOut,'melodic_IC.nii.gz')
        melmix = op.join(icaOut,'melodic_mix')
        melFTmix = op.join(icaOut,'melodic_FTmix')
        
        edgeFract, csfFract = feature_spatial(fslDir, icaOut, getcwd(), melIC_MNI)
        maxRPcorr = feature_time_series(melmix, motionFile)
        HFC = feature_frequency(melFTmix, TR)
        motionICs = classification(icaOut, maxRPcorr, edgeFract, HFC, csfFract)
        
        if len(motionICs) > 0:
            melmix = op.join(icaOut,'melodic_mix')
            if len(flavor)>1:
                denType = flavor[1]
            else:
                denType = 'aggr'
            if denType == 'aggr':
                X = np.loadtxt(melmix)[:,motionICs]
            else:
                # Partial regression
                X = np.loadtxt(melmix)
                # if filtering has already been performed, regressors need to be filtered too
                if len(config.filtering)>0:
                    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
                    X = filter_regressors(X, config.filtering, nTRs, TR)  

                if config.doScrubbing:
                    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
                    toCensor = np.loadtxt(op.join(buildpath(config.subject,config.fmriRun), 'Censored_TimePoints.txt'), dtype=np.dtype(np.int32))
                    npts = len(toCensor)
                    toReg = np.zeros((nTRs, npts),dtype=np.float32)
                    for i in range(npts):
                        toReg[toCensor[i],i] = 1
                    X = np.concatenate((X, toReg), axis=1)
                    
                niiImg = partial_regress(niiImg, nTRs, TR, X, motionICs, config.preWhitening)
                return niiImg
        else:
            print 'ICA-AROMA: None of the components was classified as motion, so no denoising is applied.'
            
    else:
        print 'Wrong flavor, using default regressors: R dR'
        X = data   

    # if filtering has already been performed, regressors need to be filtered too
    if len(config.filtering)>0:
        nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        X = filter_regressors(X, config.filtering, nTRs, TR)
    if config.doScrubbing:
        nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        toCensor = np.loadtxt(op.join(buildpath(config.subject,config.fmriRun), 'Censored_TimePoints.txt'), dtype=np.dtype(np.int32))
        npts = len(toCensor)
        toReg = np.zeros((nTRs, npts),dtype=np.float32)
        for i in range(npts):
            toReg[toCensor[i],i] = 1
        X = np.concatenate((X, toReg), axis=1)
    return X
    

def Scrubbing(niiImg, flavor, masks, imgInfo):
    thr = flavor[1]
    if flavor[0] == 'DVARS':
        # pcSigCh
        meanImg = np.mean(niiImg,axis=1)[:,np.newaxis]
        niiImg2 = 100 * (niiImg - meanImg) / meanImg
        niiImg2[np.where(np.isnan(niiImg2))] = 0
        dt = np.diff(niiImg2, n=1, axis=1)
        dt = np.concatenate((np.zeros((dt.shape[0],1),dtype=np.float32), dt), axis=1)
        score = np.sqrt(np.mean(dt**2,0))   
    elif flavor[0] == 'FD':
        motionFile = op.join(buildpath(config.subject,config.fmriRun), 'Movement_Regressors_dt.txt')
        dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
        headradius=50 #50mm as in Powers et al.
        disp=dmotpars.copy()
        disp[:,3:]=np.pi*headradius*2*(disp[:,3:]/360)
        score=np.sum(disp,1)
    elif flavor[0] == 'FD+DVARS':
        motionFile = op.join(buildpath(config.subject,config.fmriRun), 'Movement_Regressors_dt.txt')
        dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
        headradius=50 #50mm as in Powers et al. 2012
        disp=dmotpars.copy()
        disp[:,3:]=np.pi*headradius*2*(disp[:,3:]/360)
        score=np.sum(disp,1)
        # pcSigCh
        meanImg = np.mean(niiImg,axis=1)[:,np.newaxis]
        niiImg2 = 100 * (niiImg - meanImg) / meanImg
        niiImg2[np.where(np.isnan(niiImg2))] = 0
        dt = np.diff(niiImg2, n=1, axis=1)
        dt = np.concatenate((np.zeros((dt.shape[0],1),dtype=np.float32), dt), axis=1)
        scoreDVARS = np.sqrt(np.mean(dt**2,0)) 
    elif flavor[0] == 'RMS':
        RelRMSFile = op.join(buildpath(config.subject, config.fmriRun), 'Movement_RelativeRMS.txt')
        score = np.loadtxt(RelRMSFile)
    else:
        print 'Wrong scrubbing flavor. Nothing was done'
        return niiImg        
    
    if flavor[0] == 'FD+DVARS':
        thr2 = flavor[2]
        censored = np.where(np.logical_and(score>thr,scoreDVARS>thr2))
    else:
        censored = np.where(score>thr)
    if len(flavor)>3 and flavor[0]=='FD+DVARS':
        pad = flavor[3]
        a_minus = [i-k for i in censored[0] for k in range(1, pad+1)]
        a_plus  = [i+k for i in censored[0] for k in range(1, pad+1)]
        censored = np.concatenate((censored[0], a_minus, a_plus))
        censored = np.unique(censored[np.where(np.logical_and(censored>=0, censored<len(score)))])
    elif len(flavor)>2 and flavor[0]!='FD+DVARS':
        pad = flavor[2]
        a_minus = [i-k for i in censored[0] for k in range(1, pad+1)]
        a_plus  = [i+k for i in censored[0] for k in range(1, pad+1)]
        censored = np.concatenate((censored[0], a_minus, a_plus))
        censored = np.unique(censored[np.where(np.logical_and(censored>=0, censored<len(score)))])

    np.savetxt(op.join(buildpath(config.subject,config.fmriRun), 'Censored_TimePoints.txt'), censored, delimiter='\n', fmt='%d')
    config.doScrubbing = True
    return niiImg

def TissueRegression(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo

        
    if flavor[0] == 'CompCor':
        X = extract_noise_components(niiImg, maskWM_, maskCSF_, num_components=flavor[1], flavor=flavor[2])
        
    elif flavor[0] == 'WMCSF':
        meanWM = np.mean(np.float32(niiImg[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float32(niiImg[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)
        
    elif flavor[0] == 'WMCSF+dt':
        meanWM = np.mean(np.float32(niiImg[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float32(niiImg[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        dtWM=np.zeros(meanWM.shape,dtype=np.float32)
        dtWM[1:] = np.diff(meanWM, n=1)
        dtCSF=np.zeros(meanCSF.shape,dtype=np.float32)
        dtCSF[1:] = np.diff(meanCSF, n=1)
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis], 
                             dtWM[:,np.newaxis], dtCSF[:,np.newaxis]), axis=1)
    elif flavor[0] == 'WMCSF+dt+sq':
        meanWM = np.mean(np.float32(niiImg[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float32(niiImg[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        dtWM=np.zeros(meanWM.shape,dtype=np.float32)
        dtWM[1:] = np.diff(meanWM, n=1)
        dtCSF=np.zeros(meanCSF.shape,dtype=np.float32)
        dtCSF[1:] = np.diff(meanCSF, n=1)
        sqmeanWM = meanWM ** 2
        sqmeanCSF = meanCSF ** 2
        sqdtWM = dtWM ** 2
        sqdtCSF = dtCSF ** 2
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis], 
                             dtWM[:,np.newaxis], dtCSF[:,np.newaxis], 
                             sqmeanWM[:,np.newaxis], sqmeanCSF[:,np.newaxis], 
                             sqdtWM[:,np.newaxis], sqdtCSF[:,np.newaxis]),axis=1) 
    elif flavor[0] == 'GM':
        meanGM = np.mean(np.float32(niiImg[maskGM_,:]),axis=0)
        meanGM = meanGM - np.mean(meanGM)
        meanGM = meanGM/max(meanGM)
        X = meanGM[:,np.newaxis]
    elif flavor[0] == 'WM':
        meanWM = np.mean(np.float32(niiImg[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        X = meanWM[:,np.newaxis]   
    else:
        print 'Warning! Wrong tissue regression flavor. Nothing was done'
    
        
    if flavor[-1] == 'GM':
        if config.isCifti:
            niiImgGM = niiImg
        else:
            niiImgGM = niiImg[maskGM_,:]
        niiImgGM = regress(niiImgGM, nTRs, TR, X, config.preWhitening)
        if not config.isCifti:
            niiImg[maskGM_,:] = niiImgGM
        else:
            niiImg = niiImgGM
        return niiImg
    elif flavor[-1] == 'wholebrain':
        return X
    else:
        print 'Warning! Wrong tissue regression flavor. Nothing was done'

def Detrending(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo

    nPoly = flavor[1] + 1
    if flavor[2] == 'WMCSF':
        niiImgWMCSF = niiImg[np.logical_or(maskWM_,maskCSF_),:]
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1],nTRs)                
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)            
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:]) 
        else:
            print 'Warning! Wrong detrend flavor. Nothing was done'
        niiImgWMCSF = regress(niiImgWMCSF, nTRs, TR, y[1:nPoly,:].T, config.preWhitening)
        niiImg[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
    elif flavor[2] == 'GM':
        if config.isCifti:
            niiImgGM = niiImg
        else:
            niiImgGM = niiImg[maskGM_,:]

        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1], nTRs)
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])
        niiImgGM = regress(niiImgGM, nTRs, TR, y[1:nPoly,:].T, config.preWhitening)
        if not config.isCifti:
            niiImg[maskGM_,:] = niiImgGM
        else:
            niiImg = niiImgGM
    elif flavor[2] == 'wholebrain':
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1], nTRs)
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])        
        else:
            print 'Warning! Wrong detrend flavor. Nothing was done'
        return y[1:nPoly,:].T    
    else:
        print 'Warning! Wrong detrend mask. Nothing was done' 
    return niiImg 

   

def SpatialSmoothing(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo

    niiimg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
    niiimg[maskAll,:] = niiImg
    niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
    newimg = nib.Nifti1Image(niiimg, affine)
    if flavor[0] == 'Gaussian':
        newimg = smooth_img(newimg, flavor[1])
        niiimg = np.reshape(newimg.get_data(), (nRows*nCols*nSlices, nTRs), order='F')
        niiImg = niiimg[maskAll,:]
    elif flavor[0] == 'GaussianGM':
        GMmaskFile = op.join(buildpath(config.subject,config.fmriRun),'GMmask.nii')
        masker = NiftiMasker(mask_img=GMmaskFile, sessions=None, smoothing_fwhm=flavor[1])
        niiImg[maskGM_,:] = masker.fit_transform(newimg).T
    else:
        print 'Warning! Wrong smoothing flavor. Nothing was done'
    return niiImg  

def TemporalFiltering(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo

    if flavor[0] == 'Butter':
        niiImg = clean(niiImg.T, detrend=False, standardize=False, 
                              t_r=TR, high_pass=flavor[1], low_pass=flavor[2]).T
    elif flavor[0] == 'Gaussian':
        w = signal.gaussian(11,std=flavor[1])
        niiImg = signal.lfilter(w,1,niiImg)
    elif flavor[0] == 'DCT':
        K = dctmtx(nTRs)
        HPC = 1/flavor[1]
        LPC = 1/flavor[2]
        nHP = np.fix(2*(nTRs*TR)/HPC + 1)
        nLP = np.fix(2*(nTRs*TR)/LPC + 1)
        K = K[:,np.concatenate((range(1,nHP),range(int(nLP)-1,nTRs)))]
        return K
    else:
        print 'Warning! Wrong temporal filtering flavor. Nothing was done'    
        return niiImg
    config.filtering = flavor
    return niiImg
    
def GlobalSignalRegression(niiImg, flavor, masks, imgInfo):
    meanAll = np.mean(niiImg,axis=0)
    meanAll = meanAll - np.mean(meanAll)
    meanAll = meanAll/max(meanAll)
    if flavor[0] == 'GS':
        return meanAll[:,np.newaxis]
    elif flavor[0] == 'GS+dt':
        dtGS=np.zeros(meanAll.shape,dtype=np.float32)
        dtGS[1:] = np.diff(meanAll, n=1)
        X  = np.concatenate((meanAll[:,np.newaxis], dtGS[:,np.newaxis]), axis=1)
        return X
    elif flavor[0] == 'GS+dt+sq':
        dtGS = np.zeros(meanAll.shape,dtype=np.float32)
        dtGS[1:] = np.diff(meanAll, n=1)
        sqGS = meanAll ** 2
        sqdtGS = dtGS ** 2
        X  = np.concatenate((meanAll[:,np.newaxis], dtGS[:,np.newaxis], sqGS[:,np.newaxis], sqdtGS[:,np.newaxis]), axis=1)
        return X
    else:
        print 'Warning! Wrong normalization flavor. Using defalut regressor: GS'
        return meanAll[:,np.newaxis]

def VoxelNormalization(niiImg, flavor, masks, imgInfo):
    if flavor[0] == 'zscore':
        niiImg = stats.zscore(niiImg, axis=1, ddof=1)
        return niiImg
    elif flavor[0] == 'pcSigCh':
        niiImg = 100 * (niiImg - np.mean(niiImg,axis=1)[:,np.newaxis]) / np.mean(niiImg,axis=1)[:,np.newaxis]
    elif flavor[0] == 'demean':
        niiImg = niiImg - niiImg.mean(1)[:,np.newaxis]
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
    'GlobalSignalRegression' : GlobalSignalRegression,  
    'VoxelNormalization'     : VoxelNormalization,
}

sortedOperations = sorted(Operations, key=operator.itemgetter(1))
steps = {}
Flavors = {}
cstep = 0

# If requested, scrubbing is performed first, before any denoising step
scrub_idx = -1
curr_idx = -1
for opr in sortedOperations:
    curr_idx = curr_idx+1
    if opr[0] == 'Scrubbing' and opr[1] != 1 and opr[1] != 0:
        scrub_idx = opr[1]
        break
        
if scrub_idx != -1:        
    for opr in sortedOperations:  
        if opr[1] != 0 and opr[1] <= scrub_idx:
            opr[1] = opr[1]+1

    sortedOperations[curr_idx][1] = 1    
    sortedOperations = sorted(Operations, key=operator.itemgetter(1))

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

config.nParcels = int(rawgencount(op.join(PARCELDIR,config.parcellation,'labels.txt'))/2)
