from __future__ import division

# initialize variables
class config(object):
    overwrite    = False
    joblist = list()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
#from sklearn import cross_validation
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
from time import localtime, strftime, sleep, time
from scipy.spatial.distance import pdist, squareform
import socket
import fnmatch
import re
from past.utils import old_div
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

# customize path to get access to single runs
def buildpath():
    return op.join(config.DATADIR, config.subject,'MNINonLinear','Results',config.fmriRun)


config.operationDict = {
    'Finn': [
        ['VoxelNormalization',      1, ['zscore']],
        ['Detrending',              2, ['legendre', 3, 'WMCSF']],
        ['TissueRegression',        3, ['WMCSF', 'GM']],
        ['MotionRegression',        4, ['R dR']],
        ['TemporalFiltering',       5, ['Gaussian', 1]],
        ['Detrending',              6, ['legendre', 3 ,'GM']],
        ['GlobalSignalRegression',  7, ['GS']]
        ],
    'Gordon1': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']],
        ['TissueRegression',        3, ['WMCSF', 'wholebrain']],
        ['MotionRegression',        3, ['R R^2 R-1 R-1^2']],
        ['GlobalSignalRegression',  3, ['GS']],
        ['TemporalFiltering',       4, ['Butter', 0.009, 0.08]],
        ['Scrubbing',               5, ['FD', 0.2]]
        ],
    'Gordon2': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']],
        ['TissueRegression',        3, ['WMCSF', 'wholebrain']],
        ['MotionRegression',        3, ['R R^2 R-1 R-1^2']],
        ['TemporalFiltering',       3, ['DCT', 0.009, 0.08]],
        ['GlobalSignalRegression',  3, ['GS']],
        ['Scrubbing',               3, ['FD', 0.2]]
        ],
    'Ciric1': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['TissueRegression',        4, ['WMCSF', 'wholebrain']]
        ],
    'Ciric2': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R']]
        ],
    'Ciric3': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R']],
        ['TissueRegression',        4, ['WMCSF', 'wholebrain']],
        ['GlobalSignalRegression',  4, ['GS']]
        ],
    'Ciric4': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R dR R^2 dR^2']]
        ],
    'Ciric5': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R dR R^2 dR^2']],
        ['TissueRegression',        4, ['WMCSF+dt+sq', 'wholebrain']],
        ['GlobalSignalRegression',  4, ['GS+dt+sq']]
        ],
    'Ciric6': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R dR R^2 dR^2']],
        ['TissueRegression',        4, ['WMCSF+dt+sq', 'wholebrain']],
        ['GlobalSignalRegression',  4, ['GS+dt+sq']],
        ['Scrubbing',               4, ['RMS', 0.25]]
        ],
    'Ciric9': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R dR']],
        ['TissueRegression',        4, ['CompCor', 5, 'WM+CSF']],
        ['GlobalSignalRegression',  4, ['GS+dt+sq']]
        ],
    'Ciric13': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['ICA-AROMA', 'aggr']],
        ['TissueRegression',        4, ['WMCSF', 'wholebrain']]
        ],
    'Ciric14': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['ICA-AROMA', 'aggr']],
        ['TissueRegression',        4, ['WMCSF', 'wholebrain']],
        ['GlobalSignalRegression',  4, ['GS']]
        ],
    'SiegelA': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']]
        ],
    'SiegelB': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']],
        ['TissueRegression',        3, ['CompCor', 5, 'WMCSF']],
        ['TissueRegression',        3, ['GM', 'wholebrain']], 
        ['GlobalSignalRegression',  3, ['GS']],
        ['Scrubbing',               3, ['FD+DVARS', 0.025, 5]], #missing
        ['TemporalFiltering',       4, ['Butter', 0.009, 0.08]]
        ]
    }
    
# regressors: to be filtered, no. time points x no. regressors
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
        resid = niiImg[i,:] - np.ravel(fittedvalues)
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
    # Print out text file for each polynomial to be used as a regressor
    for i in num_pol:
        myleg = Legendre(coeff[i])
        y[i,:] = myleg(x) 
        if i>0:
            y[i,:] = y[i,:] - np.mean(y[i,:])
            y[i,:] = y[i,:]/np.max(y[i,:])
        #np.savetxt(op.join(buildpath(),
        #                   'poly_detrend_legendre' + str(i) + '.txt'), y[i,:] ,fmt='%.4f')
    return y

def load_img(fmriFile,maskAll):
    if config.isCifti:
        toUnzip = fmriFile.replace('_Atlas','').replace('.dtseries.nii','.nii.gz')
        if not op.isfile(fmriFile.replace('.dtseries.nii','.tsv')):
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(fmriFile,fmriFile.replace('.dtseries.nii','.tsv'))
            call(cmd,shell=True)
    else:
        toUnzip = fmriFile
    # outFilePath = toUnzip.replace('.gz','') 
    # if not op.isfile(outFilePath):
    #     with open(toUnzip, 'rb') as fFile:
    #         decompressedFile = gzip.GzipFile(fileobj=fFile)
    #         #op.join(buildpath(), config.fmriRun+'.nii')
    #         with open(outFilePath, 'wb') as outfile:
    #             outfile.write(decompressedFile.read())
    # volFile = outFilePath
    # img = nib.load(volFile)
   
    img = nib.load(toUnzip)

    # myoffset = img.dataobj.offset
    try:
        nRows, nCols, nSlices, nTRs = img.header.get_data_shape()
    except:
        nRows, nCols, nSlices = img.header.get_data_shape()
        nTRs = 1

    TR = img.header.structarr['pixdim'][4]
    # data = np.memmap(volFile, dtype=img.header.get_data_dtype(), mode='c', order='F',
    #                  offset=myoffset,shape=img.header.get_data_shape())
    # data = data.reshape([nRows*nCols*nSlices, nTRs], order='F')
    # data = data[maskAll,:]

    if nTRs==1:
        data = np.reshape(np.asarray(img.dataobj),nRows*nCols*nSlices, order='F')[maskAll]
    else:
        data = np.reshape(np.asarray(img.dataobj),[nRows*nCols*nSlices,nTRs], order='F')[maskAll,:]

    return data, nRows, nCols, nSlices, nTRs, img.affine, TR

def plot_hist(score,title,xlabel):
    h,b = np.histogram(score, bins='auto')
    plt.hist(score,bins=b)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    return h

def makeTissueMasks(overwrite=False):
    fmriFile = op.join(buildpath(), config.fmriRun+config.suffix+'.nii.gz')
    WMmaskFileout = op.join(buildpath(), 'WMmask.nii')
    CSFmaskFileout = op.join(buildpath(), 'CSFmask.nii')
    GMmaskFileout = op.join(buildpath(), 'GMmask.nii')
    
    if not op.isfile(GMmaskFileout) or overwrite:
        # load ribbon.nii.gz and wmparc.nii.gz
        ribbonFilein = op.join(config.DATADIR, config.subject, 'MNINonLinear','ribbon.nii.gz')
        wmparcFilein = op.join(config.DATADIR, config.subject, 'MNINonLinear', 'wmparc.nii.gz')
        # make sure it is resampled to the same space as the functional run
        ribbonFileout = op.join(buildpath(), 'ribbon.nii.gz')
        wmparcFileout = op.join(buildpath(), 'wmparc.nii.gz')
        # make identity matrix to feed to flirt for resampling
        ribbonMat = op.join(buildpath(), 'ribbon_flirt.mat')
        wmparcMat = op.join(buildpath(), 'wmparc_flirt.mat')
        eyeMat = op.join(buildpath(), 'eye.mat')
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
        ribbon = np.asarray(nib.load(ribbonFileout).dataobj)
        wmparc = np.asarray(nib.load(wmparcFileout).dataobj)
        
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
    # myoffset = tmpnii.dataobj.offset
    # data = np.memmap(WMmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
    #                  offset=myoffset,shape=tmpnii.header.get_data_shape())
    nRows, nCols, nSlices = tmpnii.header.get_data_shape()
    # maskWM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    maskWM = np.reshape(np.asarray(tmpnii.dataobj) > 0,nRows*nCols*nSlices, order='F')

    tmpnii = nib.load(CSFmaskFileout)
    # myoffset = tmpnii.dataobj.offset
    # data = np.memmap(CSFmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F', 
    #                  offset=myoffset,shape=tmpnii.header.get_data_shape())
    # maskCSF = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    maskCSF = np.reshape(np.asarray(tmpnii.dataobj) > 0,nRows*nCols*nSlices, order='F')

    tmpnii = nib.load(GMmaskFileout)
    # myoffset = tmpnii.dataobj.offset
    # data = np.memmap(GMmaskFileout, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F', 
    #                  offset=myoffset,shape=tmpnii.header.get_data_shape())
    # maskGM = np.reshape(data > 0,nRows*nCols*nSlices, order='F')
    maskGM = np.reshape(np.asarray(tmpnii.dataobj) > 0,nRows*nCols*nSlices, order='F')

    tmpnii = None
    maskAll  = np.logical_or(np.logical_or(maskWM, maskCSF), maskGM)
    maskWM_  = maskWM[maskAll]
    maskCSF_ = maskCSF[maskAll]
    maskGM_  = maskGM[maskAll]
    
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
    """
    Grabbed from https://github.com/tacorna/taco/blob/master/taco/lib/stats.py
    Returns the point at the given fraction between a and b, where
    'fraction' must be between 0 and 1.
    """
    return a + (b - a)*fraction;

def scoreatpercentile(a, per, limit=(), interpolation_method='fraction'):
    """
    Grabbed from https://github.com/tacorna/taco/blob/master/taco/lib/stats.py
    Adapted from scipy.
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
    """
    Largely based on http://www.mrc-cbu.cam.ac.uk/wp-content/uploads/2013/01/rsfMRI_GLM.m
    """
    K=N
    n = range(N)
    C = np.zeros((len(n), K),dtype=np.float32)
    C[:,0] = np.ones((len(n)),dtype=np.float32)/np.sqrt(N)
    doublen = [2*x+1 for x in n]
    for k in range(1,K):
        C[:,k] = np.sqrt(2/N)*np.cos([np.pi*x*(k-1)/(2*N) for x in doublen])        
    return C 

# from nipy
def orth(X, tol=1.0e-07):
    """
    Grabbed from https://github.com/nipy/nipy/blob/master/nipy/modalities/fmri/spm/reml.py
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

def reml(sigma, components, design=None, n=1, niter=128, penalty_cov=np.exp(-32), penalty_mean=0):
    """
    Grabbed from https://github.com/nipy/nipy/blob/master/nipy/modalities/fmri/spm/reml.py
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
    """
    Largely based on http://www.mrc-cbu.cam.ac.uk/wp-content/uploads/2013/01/rsfMRI_GLM.m
    """
    u, s, _  = linalg.svd(V)
    s = np.sqrt(np.abs(np.diag(s)))
    m = s.shape[0]
    return np.dot(u, np.dot(s, u.T))

def prewhitening(niiImg, nTRs, TR, X):
    """
    Largely based on http://www.mrc-cbu.cam.ac.uk/wp-content/uploads/2013/01/rsfMRI_GLM.m
    """
    T = np.arange(nTRs) * TR
    d = 2 ** (np.floor(np.arange(np.log2(TR/4), 7)))
    Q = np.array([linalg.toeplitz((T**j)*np.exp(-T/d[i])) for i in range(len(d)) for j in [0,1]])
    CY = np.cov(niiImg.T)
    V, h, _ = reml(CY, Q, design=X, n=1)
    W = linalg.inv(sqrtm(V))
    return W

def plot_corrs(x,y):
    # fit a curve to the data using a least squares 1st order polynomial fit
    z = np.polyfit(x,y,1)

    p = np.poly1d(z)
    fit = p(x)

    # get the coordinates for the fit curve
    c_x = [np.min(x),np.max(x)]
    c_y = p(c_x)

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y - p_y

    # create series of new test x-values to predict for
    p_x = np.arange(np.min(x),np.max(x)+1,1)

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)         # mean of x
    n = len(x)              # number of samples in origional fit
    DF = n_subj - z.size                            # degrees of freedom
    t = stats.t.ppf(0.95, DF)           # used for CI and PI bands
    s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

    #confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))
    confs = t * np.sqrt((s_err/(n-2)) * (1.0/n + (np.power((p_x-mean_x),2)/(np.sum(np.power(x - mean_x,2))))))
    # now predict y based on test x-values
    p_y = z[0]*p_x+z[1]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    # set-up the plot
    #plt.subplots(figsize=(8,8))
    plt.axes().set_aspect('equal')
    plt.xlabel('Original score')
    plt.ylabel('Predicted score')
    plt.title('Linear regression and confidence limits')

    # plot sample data
    plt.plot(x,y,'bo')

    # plot line of best fit
    plt.plot(c_x,c_y,'r-',label='Regression line')

    # plot confidence limits
    plt.plot(p_x,lower,'b--',label='Lower confidence limit (95%)')
    plt.plot(p_x,upper,'b--',label='Upper confidence limit (95%)')

    # set coordinate limits
    plt.xlim(4,25)
    plt.ylim(5,25)

    # configure legend
    plt.legend(loc=0)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)

    # add text
    rho,p = stats.pearsonr(np.ravel(predictions_pos),np.squeeze(gF))
    s = 'r={:0.2f}\np={:1.4f}'.format(rho,p)
    plt.text(20, 7, s, fontsize=12)

    # show the plot
    plt.show()
    
    
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
                    return op.join(resDir,config.fmriRun+'_prepro_'+rcode+config.ext)
    return None

def get_rcode(mystring):
    if not config.isCifti:
        #re.search('.*_(........)\.nii.gz', mystring).group(1)
        return re.search('.*_(........)\.nii.gz', mystring).group(1)
    else:
        #re.search('.*_(........)\.dtseries.nii', mystring).group(1)
        return re.search('.*_(........)\.dtseries.nii', mystring).group(1)

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.read)
    return sum( buf.count(b'\n') for buf in f_gen )        


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

def denoising(fslDir, inFile, outDir, melmix, denType, denIdx):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function classifies the ICs based on the four features; maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be denoised
    outDir:     Full path of the output directory
    melmix:     Full path of the melodic_mix text file
    denType:    Type of requested denoising ('aggr': aggressive, 'nonaggr': non-aggressive, 'both': both aggressive and non-aggressive 
    denIdx:     Indices of the components that should be regressed out

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    denoised_func_data_<denType>.nii.gz:        A nii.gz file of the denoised fMRI data"""

    # Check if denoising is needed (i.e. are there components classified as motion)
    check = len(denIdx) > 0

    if check==1:
        # Put IC indices into a char array
        denIdxStr = np.char.mod('%i',(denIdx+1))

        # Non-aggressive denoising of the data using fsl_regfilt (partial regression), if requested
        if (denType == 'nonaggr') or (denType == 'both'):       
            os.system(' '.join([op.join(fslDir,'fsl_regfilt'),
                '--in=' + inFile,
                '--design=' + melmix,
                '--filter="' + ','.join(denIdxStr) + '"',
                '--out=' + op.join(outDir,'denoised_func_data_nonaggr.nii.gz')]))

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if (denType == 'aggr') or (denType == 'both'):
            os.system(' '.join([op.join(fslDir,'fsl_regfilt'),
                '--in=' + inFile,
                '--design=' + melmix,
                '--filter="' + ','.join(denIdxStr) + '"',
                '--out=' + op.join(outDir,'denoised_func_data_aggr.nii.gz'),
                '-a']))
    else:
        print "  - None of the components was classified as motion, so no denoising is applied (a symbolic link to the input file will be created)."
        if (denType == 'nonaggr') or (denType == 'both'):
            os.symlink(inFile,op.join(outDir,'denoised_func_data_nonaggr.nii.gz'))
        if (denType == 'aggr') or (denType == 'both'):
            os.symlink(inFile,op.join(outDir,'denoised_func_data_aggr.nii.gz'))

def MotionRegression(niiImg, flavor, masks, imgInfo):
    # assumes that data is organized as in the HCP
    motionFile = op.join(buildpath(), 'Movement_Regressors_dt.txt')
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
        data[:,:6]
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        data_roll2 = np.roll(data_roll, 1, axis=0)
        data_roll2[0] = 0
        data_roll2_squared = data_roll2 ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared, data_roll2, data_roll2_squared), axis=1)
    elif flavor[0] == 'ICA-AROMA':
        fslDir = op.join(environ["FSLDIR"],'bin','')
        icaOut = op.join(buildpath(), 'rfMRI_REST1_LR_hp2000.ica','filtered_func_data.ica')
        melIC_MNI = op.join(icaOut,'melodic_IC.nii.gz')
        mc = op.join(buildpath(), 'Movement_Regressors_dt.txt')
        melmix = op.join(icaOut,'melodic_mix')
        melFTmix = op.join(icaOut,'melodic_FTmix')
        
        edgeFract, csfFract = feature_spatial(fslDir, icaOut, getcwd(), melIC_MNI)
        maxRPcorr = feature_time_series(melmix, mc)
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
            elif denType == 'nonaggr':  
                # Partial regression
                X = np.loadtxt(melmix)
                # if filtering has already been performed, regressors need to be filtered too
                if len(config.filtering)>0:
                    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
                    X = filter_regressors(X, config.filtering, nTRs, TR)  

                if config.doScrubbing:
                    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
                    toCensor = np.loadtxt(op.join(buildpath(), 'Censored_TimePoints.txt'), dtype=np.dtype(np.int32))
                    npts = len(toCensor)
                    toReg = np.zeros((nTRs, npts),dtype=np.float32)
                    for i in range(npts):
                        toReg[toCensor[i],i] = 1
                    X = np.concatenate((X, toReg), axis=1)
                    
                nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
                niiImg = partial_regress(niiImg, nTRs, TR, X, motionICs, config.preWhitening)
                return niiImg
            else:
                print 'Warning! Wrong ICA-AROMA flavor. Using default full regression.'
                X = np.loadtxt(melmix)[:,motionICs]
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
        toCensor = np.loadtxt(op.join(buildpath(), 'Censored_TimePoints.txt'), dtype=np.dtype(np.int32))
        npts = len(toCensor)
        toReg = np.zeros((nTRs, npts),dtype=np.float32)
        for i in range(npts):
            toReg[toCensor[i],i] = 1
        X = np.concatenate((X, toReg), axis=1)
        
    return X


def Scrubbing(niiImg, flavor, masks, imgInfo):
    """
    Largely based on: 
    - https://git.becs.aalto.fi/bml/bramila/blob/master/bramila_dvars.m
    - https://github.com/poldrack/fmriqa/blob/master/compute_fd.py
    """
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
        motionFile = op.join(buildpath(), 'Movement_Regressors_dt.txt')
        dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
        headradius=50 #50mm as in Powers et al. 2012
        disp=dmotpars.copy()
        disp[:,3:]=np.pi*headradius*2*(disp[:,3:]/360)
        score=np.sum(disp,1)
    else:
        print 'Wrong scrubbing flavor. Nothing was done'
        return niiImg
    
    censored = np.where(score>thr)
    
    if len(flavor)>2:
        pad = flavor[2]
        a_minus = [i-k for i in censored[0] for k in range(1, pad+1)]
        a_plus  = [i+k for i in censored[0] for k in range(1, pad+1)]
        censored = np.concatenate((censored[0], a_minus, a_plus))
        censored = np.unique(censored[np.where(np.logical_and(censored>=0, censored<len(score)))])
    
    np.savetxt(op.join(buildpath(), 'Censored_TimePoints.txt'), censored, delimiter='\n', fmt='%d')
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
        niiimg = np.reshape(np.asarray(newimg.dataobj), (nRows*nCols*nSlices, nTRs), order='F')
        niiImg = niiimg[maskAll,:]
    elif flavor[0] == 'GaussianGM':
        GMmaskFile = op.join(buildpath(),'GMmask.nii')
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
   

# these variables are initialized here and used later in the pipeline, do not change
config.filtering = []
config.doScrubbing = False

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

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def computeFD():
    # Frame displacement
    motionFile = op.join(buildpath(), 'Movement_Regressors_dt.txt')
    dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
    headradius=50 #50mm as in Powers et al. 2012
    disp=dmotpars.copy()
    disp[:,3:]=np.pi*headradius*2*(disp[:,3:]/360)
    score=np.sum(disp,1)
    return score

def makeGrayPlot(displayPlot=False,overwrite=False):
    savePlotFile = config.fmriFile_dn.replace(config.ext,'_grayplot.png')
    if not op.isfile(savePlotFile) or overwrite:
        # FD
        score = computeFD()
        # load masks
        #t=time()
        maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(False)
        #print "makeGrayPlot -- loaded masks in {:0.2f}s".format(time()-t)
        # original volume
        #t=time()
        X, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.fmriFile, maskAll)
        # print "makeGrayPlot -- loaded orig fMRI in {:0.2f}s".format(time()-t)
        # pct signal change
        # t=time()
        meanX = np.mean(X,axis=1)
        X -= meanX[:,np.newaxis]
        X /= meanX[:,np.newaxis]
        # print "makeGrayPlot -- calculated pc sig change in {:0.2f}s".format(time()-t)
        #X = np.vstack((X[maskGM_,:], X[maskWM_,:], X[maskCSF_,:]))
        # t=time()
        Xgm  = 100*X[maskGM_,:]
        Xwm  = 100*X[maskWM_,:]
        Xcsf = 100*X[maskCSF_,:]
        X    = None
        # print "makeGrayPlot -- separated GM, WM, CSF in {:0.2f}s".format(time()-t)
        #
        # t=time()
        fig = plt.figure(figsize=(15,20))
        ax1 = plt.subplot(311)
        plt.plot(np.arange(nTRs), score)
        plt.title('Subject {}, run {}, denoising {}'.format(config.subject,config.fmriRun,config.pipelineName))
        plt.ylabel('FD (mm)')
        # print "makeGrayPlot -- plotted FD in {:0.2f}s".format(time()-t)
        #
        # t=time()
        ax2 = plt.subplot(312, sharex=ax1)
        im = plt.imshow(np.vstack((Xgm,Xwm,Xcsf)), aspect='auto', interpolation='none', cmap=plt.cm.gray)
        im.set_clim(vmin=-5, vmax=5)
        plt.title('Before denoising')
        plt.ylabel('Voxels')
        plt.axhline(y=np.sum(maskGM_), color='r')
        plt.axhline(y=np.sum(maskGM_)+np.sum(maskWM_), color='b')
        # print "makeGrayPlot -- plotted orig fMRI in {:0.2f}s".format(time()-t)

        # denoised volume
        # t=time()
        X, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.fmriFile_dn, maskAll)
        # print "makeGrayPlot -- loaded denoised fMRI in {:0.2f}s".format(time()-t)
        #X = np.vstack((X[maskGM_,:], X[maskWM_,:], X[maskCSF_,:]))
        # t=time()
        Xgm  = X[maskGM_,:]
        Xwm  = X[maskWM_,:]
        Xcsf = X[maskCSF_,:]
        X    = None
        # print "makeGrayPlot -- separated GM, WM, CSF in {:0.2f}s".format(time()-t)
        #
        # t=time()
        ax3 = plt.subplot(313, sharex=ax1)
        im = plt.imshow(np.vstack((Xgm,Xwm,Xcsf)), aspect='auto', interpolation='none', cmap=plt.cm.gray)
        im.set_clim(vmin=-5, vmax=5)
        plt.title('After denoising')
        plt.ylabel('Voxels')
        plt.axhline(y=np.sum(maskGM_), color='r')
        plt.axhline(y=np.sum(maskGM_)+np.sum(maskWM_), color='b')
        # print "makeGrayPlot -- plotted denoised fMRI in {:0.2f}s".format(time()-t)

        # prettify
        # t=time()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.03, 0.4])
        fig.colorbar(im, cax=cbar_ax)
        # print "makeGrayPlot -- prettified figure in {:0.2f}s".format(time()-t)
        # save figure
        t=time()
        fig.savefig(savePlotFile, bbox_inches='tight')
        print "makeGrayPlot -- saved figure in {:0.2f}s".format(time()-t)

    else:
        image = mpimg.imread(savePlotFile)
        fig = plt.figure(figsize=(15,20))
        plt.axis("off")
        plt.imshow(image)

    if displayPlot:
        plt.show(fig)
    else:
        plt.close(fig)

def parcellate(overwrite=False):
    # After preprocessing, functional connectivity is computed
    tsDir = op.join(buildpath(),config.parcellationName)
    if not op.isdir(tsDir): mkdir(tsDir)
    tsDir = op.join(tsDir,config.fmriRun+config.ext)
    if not op.isdir(tsDir): mkdir(tsDir)

    maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(False)
    #####################
    # read parcels
    #####################
    if not config.isCifti:
        # outFilePath = config.parcellationFile.replace('.gz','')
        # if not op.isfile(outFilePath):
        #     with open(config.parcellationFile, 'rb') as fFile:
        #         decompressedFile = gzip.GzipFile(fileobj=fFile)
        #         with open(outFilePath, 'wb') as outfile:
        #             outfile.write(decompressedFile.read())
        # tmpnii   = nib.load(outFilePath)
        tmpnii   = nib.load(config.parcellationFile)
        # myoffset = tmpnii.dataobj.offset
        # niiImg   = np.memmap(outFilePath, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
        #                  offset=myoffset,shape=tmpnii.header.get_data_shape())
        nRows, nCols, nSlices = tmpnii.header.get_data_shape()
        # allparcels = np.reshape(np.uint16(niiImg),nRows*nCols*nSlices, order='F')[maskAll]
        allparcels = np.reshape(np.uint16(np.asarray(tmpnii.dataobj)),nRows*nCols*nSlices, order='F')[maskAll]
    else:
        if not op.isfile(config.parcellationFile.replace('.dlabel.nii','.tsv')):    
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.parcellationFile,
                                                                   config.parcellationFile.replace('.dlabel.nii','.tsv'))
            call(cmd, shell=True)
        allparcels = np.loadtxt(config.parcellationFile.replace('.dlabel.nii','.tsv'));
    
    ####################
    # original data
    ####################
    alltsFile = op.join(tsDir,'allParcels.txt')
    if not op.isfile(alltsFile) or overwrite:
        # read original volume
        if not config.isCifti:
            imgInfo = load_img(config.fmriFile, maskAll)
            niiImg, nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        else:
            if not op.isfile(config.fmriFile.replace('.dtseries.nii','.tsv')):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,
                                                                           config.fmriFile.replace('.dtseries.nii','.tsv'))
                call(cmd, shell=True)
            niiImg = np.loadtxt(config.fmriFile.replace('.dtseries.nii','.tsv'));
        
        for iParcel in np.arange(config.nParcels):
            tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel+1))
            if not op.isfile(tsFile) or overwrite:
                np.savetxt(tsFile,np.nanmean(niiImg[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.6f',delimiter='\n')

        # concatenate all ts
        print 'Concatenating data'
        cmd = 'paste '+op.join(tsDir,'parcel???.txt')+' > '+alltsFile
        call(cmd, shell=True)

    ####################
    # denoised data
    ####################
    rstring = get_rcode(config.fmriFile_dn)
    alltsFile = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
    if not op.isfile(alltsFile) or overwrite:
        # read denoised volume
        if not config.isCifti:
            imgInfo = load_img(config.fmriFile_dn, maskAll)
            niiImg, nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        else:
            if not op.isfile(config.fmriFile_dn.replace('.dtseries.nii','.tsv')):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile_dn,
                                                                           config.fmriFile_dn.replace('.dtseries.nii','.tsv'))
                call(cmd, shell=True)
            niiImg = np.loadtxt(config.fmriFile_dn.replace('.dtseries.nii','.tsv'));
                   
        for iParcel in np.arange(config.nParcels):
            tsFile = op.join(tsDir,'parcel{:03d}_{}.txt'.format(iParcel+1,rstring))
            if not op.isfile(tsFile) or overwrite:
                np.savetxt(tsFile,np.nanmean(niiImg[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.6f',delimiter='\n')
        # concatenate all ts
        print 'Concatenating data'
        cmd = 'paste '+op.join(tsDir,'parcel???_{}.txt'.format(rstring))+' > '+alltsFile
        call(cmd, shell=True)

def computeFC(overwrite=False):
    parcellate(overwrite)
    tsDir = op.join(buildpath(),config.parcellationName,config.fmriRun+config.ext)
    ###################
    # original
    ###################
    alltsFile = op.join(tsDir,'allParcels.txt')
    fcFile    = alltsFile.replace('.txt','_Pearson.txt')
    if not op.isfile(fcFile) or overwrite:
        ts = np.loadtxt(alltsFile)
        # correlation
        corrMat = np.corrcoef(ts,rowvar=0)
        np.fill_diagonal(corrMat,1)
        # save as .txt
        np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
    ###################
    # denoised
    ###################
    rstring = get_rcode(config.fmriFile_dn)
    alltsFile = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
    fcFile    = alltsFile.replace('.txt','_Pearson.txt')
    if not op.isfile(fcFile) or overwrite:
        ts = np.loadtxt(alltsFile)
        # correlation
        corrMat = np.corrcoef(ts,rowvar=0)
        np.fill_diagonal(corrMat,1)
        # save as .txt
        np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
      
def plotFC(displayPlot=False,overwrite=False):

    savePlotFile=config.fmriFile_dn.replace(config.ext,'_'+config.parcellationName+'_fcMat.png')

    if not op.isfile(savePlotFile) or overwrite:
        computeFC(overwrite)
    
    tsDir = op.join(buildpath(),config.parcellationName,config.fmriRun+config.ext)
    fcFile = op.join(tsDir,'allParcels_Pearson.txt')
    fcMat = np.genfromtxt(fcFile,delimiter=",")
    rstring = get_rcode(config.fmriFile_dn)
    fcFile_dn = op.join(tsDir,'allParcels_{}_Pearson.txt'.format(rstring))
    fcMat_dn = np.genfromtxt(fcFile_dn,delimiter=",")
    
    fig = plt.figure(figsize=(20,7.5))
    ####################
    # original
    ####################
    ax1 = plt.subplot(121)
    im = plt.imshow(fcMat, aspect='auto', interpolation='none', cmap=plt.cm.jet)
    im.set_clim(vmin=-.5, vmax=.5)
    plt.axis('equal')  
    plt.axis('off')  
    plt.title('Subject {}, run {}'.format(config.subject,config.fmriRun))
    plt.xlabel('parcel #')
    plt.ylabel('parcel #')
    ####################
    # denoised
    ####################
    ax2 = plt.subplot(122)#, sharey=ax1)
    im = plt.imshow(fcMat_dn, aspect='auto', interpolation='none', cmap=plt.cm.jet)
    im.set_clim(vmin=-.5, vmax=.5)
    plt.axis('equal')  
    plt.axis('off')  
    plt.title('{}'.format(config.pipelineName))
    plt.xlabel('parcel #')
    plt.ylabel('parcel #')
    # prettify
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    # save figure
    fig.savefig(savePlotFile, bbox_inches='tight')
    #
    if displayPlot:
        plt.show(fig)
    else:
        plt.close(fig)

    return fcMat,fcMat_dn

def plotQCrsFC(fcMats,fcMats_dn,fdScores):
    savePlotFile=op.join(config.DATADIR,config.pipelineName+'_'+config.parcellationName+'_MFDrsFCplot.png')
    # get distance between parcels
    # outFilePath = config.parcellationFile.replace('.gz','')
    # if not op.isfile(outFilePath):
    #     with open(config.parcellationFile, 'rb') as fFile:
    #         decompressedFile = gzip.GzipFile(fileobj=fFile)
    #         with open(outFilePath, 'wb') as outfile:
    #             outfile.write(decompressedFile.read())
    # tmpnii = nib.load(outFilePath)
    tmpnii = nib.load(config.parcellationFile)
    # myoffset = tmpnii.dataobj.offset
    # data = np.memmap(outFilePath, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
    #                  offset=myoffset,shape=tmpnii.header.get_data_shape())
    nRows, nCols, nSlices = tmpnii.header.get_data_shape()
    # idx_parcels = np.array([np.where(data==i) for i in range(1,config.nParcels+1)])
    idx_parcels = np.array([np.where(np.asarray(tmpnii.dataobj)==i) for i in range(1,config.nParcels+1)])
    # mean position of parcel in voxel coordinates
    parcel_means = np.array([(point[0].mean(), point[1].mean(), point[2].mean()) for point in idx_parcels])
    # distance between parcels
    edists = squareform(pdist(parcel_means))
    # get indices
    triu_idx = np.triu_indices(config.nParcels,1)
    fig = plt.figure(figsize=(20,7.5))
    ################
    # original
    ################
    edges_across_subj = fcMats[triu_idx[0], triu_idx[1], :]
    pears = np.array([stats.pearsonr(edges_across_subj[i],fdScores)[0] for i in range(edges_across_subj.shape[0])])
    fit = lowess(pears, edists[triu_idx])
    ax1 = plt.subplot(121)
    plt.scatter(edists[triu_idx], pears, s=.01)
    plt.ylabel('FC-MFD Correlation (r)')
    plt.xlabel('Inter-node Euclidean distance (voxels)')
    plt.xlim([0,65])
    plt.axhline(y=0, color='k')
    plt.plot(fit[:,0],fit[:,1],'blue',linewidth=3)
    ################
    # denoised
    ################
    edges_across_subj = fcMats_dn[triu_idx[0], triu_idx[1], :]
    pears = np.array([stats.pearsonr(edges_across_subj[i],fdScores)[0] for i in range(edges_across_subj.shape[0])])
    fit = lowess(pears, edists[triu_idx])
    ax2 = plt.subplot(122)
    plt.scatter(edists[triu_idx], pears, s=.01)
    plt.ylabel('FC-MFD Correlation (r)')
    plt.xlabel('Inter-node Euclidean distance (voxels)')
    plt.xlim([0,65])
    plt.axhline(y=0, color='k')
    plt.plot(fit[:,0],fit[:,1],'blue',linewidth=3)
    # save figure
    fig.savefig(savePlotFile, bbox_inches='tight')
    plt.show(fig)

def plotDeltaR(fcMats,fcMats_dn):
    savePlotFile=op.join(config.DATADIR,config.pipelineName+'_'+config.parcellationName+'_deltaR.png')
    # get distance between parcels
    # outFilePath = config.parcellationFile.replace('.gz','')
    # if not op.isfile(outFilePath):
    #     with open(config.parcellationFile, 'rb') as fFile:
    #         decompressedFile = gzip.GzipFile(fileobj=fFile)
    #         with open(outFilePath, 'wb') as outfile:
    #             outfile.write(decompressedFile.read())
    # tmpnii = nib.load(outFilePath)
    tmpnii = nib.load(config.parcellationName)
    # myoffset = tmpnii.dataobj.offset
    # data = np.memmap(outFilePath, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
    #                  offset=myoffset,shape=tmpnii.header.get_data_shape())
    nRows, nCols, nSlices = tmpnii.header.get_data_shape()
    # idx_parcels = np.array([np.where(data==i) for i in range(1,config.nParcels+1)])
    idx_parcels = np.array([np.where(np.asarray(tmpnii.dataobj)==i) for i in range(1,config.nParcels+1)])
    # mean position of parcel in voxel coordinates
    parcel_means = np.array([(point[0].mean(), point[1].mean(), point[2].mean()) for point in idx_parcels])
    # distance between parcels
    edists = squareform(pdist(parcel_means))
    # get indices
    triu_idx = np.triu_indices(config.nParcels,1)

    avfcMats    = np.tanh(np.mean(np.arctanh(fcMats),2))
    avfcMats_dn = np.tanh(np.mean(np.arctanh(fcMats_dn),2))
    edges    = avfcMats[triu_idx[0], triu_idx[1]]
    edges_dn = avfcMats_dn[triu_idx[0], triu_idx[1]]
    fit = lowess(edges_dn-edges, edists[triu_idx])

    fig = plt.figure(figsize=(20,7.5))
    plt.scatter(edists[triu_idx], edges_dn-edges, s=.01)
    plt.ylabel('deltaR')
    plt.xlabel('Inter-node Euclidean distance (voxels)')
    plt.xlim([0,65])
    plt.axhline(y=0, color='k')
    plt.plot(fit[:,0],fit[:,1],'blue',linewidth=3)
    # save figure
    fig.savefig(savePlotFile, bbox_inches='tight')
    plt.show(fig)


def runPipeline():
    sortedOperations = sorted(config.Operations, key=operator.itemgetter(1))
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
            if opr[1] != 0 and opr[1] < scrub_idx:
                opr[1] = opr[1]+1

        sortedOperations[curr_idx][1] = 1
        sortedOperations = sorted(config.Operations, key=operator.itemgetter(1))

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
                            
    precomputed = checkXML(config.fmriFile,steps,Flavors,buildpath()) 
    # make sure that there are no unzipped files trailing behind
    try: 
        remove(config.fmriFile.replace('.gz',''))
    except OSError: 
        pass
    
    if precomputed and not config.overwrite:
        #print "{}, {} -- Preprocessing already computed.".format(config.subject, config.fmriRun)
        config.fmriFile_dn = precomputed
        try: 
            remove(config.fmriFile_dn.replace('.gz',''))
        except OSError: 
            pass
        #print config.fmriFile_dn

    else:
        timeStart = localtime()
        print 'Step 0 : Building WM, CSF and GM masks...'
        masks = makeTissueMasks(False)
        maskAll, maskWM_, maskCSF_, maskGM_ = masks    
        print 'Loading data in memory...'
        imgInfo = load_img(config.fmriFile, maskAll) 
        niiImg, nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        nsteps = len(steps)
        for i in range(1,nsteps+1):
            step = steps[i]
            print 'Step '+str(i)+' '+str(step)
            if len(step) == 1:
                # Atomic operations
                if 'Regression' in step[0] or ('TemporalFiltering' in step[0] and 'DCT' in Flavors[i][0]) or ('wholebrain' in Flavors[i][0]):
                    if (step[0]=='TissueRegression' and 'GM' in Flavors[i][0]) or (step[0]=='MotionRegression' and 'nonaggr' in Flavors[i][0]): 
                        #regression constrained to GM or nonagrr ICA-AROMA
                        niiImg = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
                    else:
                        r0 = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
                        niiImg = regress(niiImg, nTRs, TR, r0, config.preWhitening)
                else:
                    niiImg = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
            else:
                # When multiple regression steps have the same order, all the regressors are combined
                # and a single regression is performed (other operations are executed in order)
                r = np.empty((nTRs, 0))
                for j in range(len(step)):
                    opr = step[j]
                    if 'Regression' in opr or ('TemporalFiltering' in opr and 'DCT' in Flavors[i][j]) or ('wholebrain' in Flavors[i][j]):
                        if (opr=='TissueRegression' and 'GM' in Flavors[i][j]) or (opr=='MotionRegression' and 'nonaggr' in Flavors[i][j]): 
                            #regression constrained to GM or nonaggr ICA-AROMA
                            niiImg = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                        else:    
                            r0 = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                            r = np.append(r, r0, axis=1)
                    else:
                        niiImg = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                if r.shape[1] > 0:
                    niiImg = regress(niiImg, nTRs, TR, r, config.preWhitening)    
            niiImg[np.isnan(niiImg)] = 0

        print 'Done! Copy the resulting file...'
        rstring = ''.join(random.SystemRandom().choice(string.ascii_lowercase +string.ascii_uppercase + string.digits) for _ in range(8))
        outDir  = buildpath()
        outFile = config.fmriRun+'_prepro_'+rstring
        if config.isCifti:
            # write to text file
            np.savetxt(op.join(outDir,outFile+'.tsv'),niiImg, delimiter='\t', fmt='%.6f')
            niiImg = None
            # need to convert back to cifti
            cmd = 'wb_command -cifti-convert -from-text {} {} {}'.format(op.join(outDir,outFile+'.tsv'),
                                                                         config.fmriFile,
                                                                         op.join(outDir,outFile+'.dtseries.nii'))
            call(cmd,shell=True)
            # delete temporary files
            cmd = 'rm -r {}/*.tsv'.format(outDir)
            call(cmd,shell=True)
        else:
            niiimg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
            niiimg[maskAll,:] = niiImg
            niiImg = None
            niiimg = np.reshape(niiimg, (nRows, nCols, nSlices, nTRs), order='F')
            newimg = nib.Nifti1Image(niiimg.astype('<f4'), affine)
            nib.save(newimg,op.join(outDir,outFile+'.nii.gz'))

        # if hasattr(config,'logfile'):
        #     f=open(config.logfile, "a+")
        #     f.write('{},{},{}\n'.format(config.subject,config.fmriRun,op.join(outDir,outFile)))
        #     f.close()

        timeEnd = localtime()  

        outXML = rstring+'.xml'
        conf2XML(config.fmriFile, config.DATADIR, sortedOperations, timeStart, timeEnd, op.join(buildpath(),outXML))
    
        print 'Preprocessing complete. '
        config.fmriFile_dn = op.join(outDir,outFile+'.nii.gz')
        #print config.fmriFile_dn
        
    makeGrayPlot(False)
    plotFC(False)
        
    # delete decompressed input files
    try: 
        remove(config.fmriFile.replace('.gz','')) 
    except OSError: 
        pass
    try: 
        remove(config.fmriFile_dn.replace('.gz','')) 
    except OSError: 
        pass

def runPipelinePar():

    if config.queue: priority=-100

    config.suffix = '_hp2000_clean' if config.useFIX else '' 

    if config.isCifti:
        config.ext = '.dtseries.nii'
    else:
        config.ext = '.nii.gz'

    if config.isCifti:
        config.fmriFile = op.join(buildpath(), config.fmriRun+'_Atlas'+config.suffix+'.dtseries.nii')
    else:
        config.fmriFile = op.join(buildpath(), config.fmriRun+config.suffix+'.nii.gz')

    if not op.isfile(config.fmriFile):
        print config.subject, 'missing'
        return False

    if config.queue:
        jobDir = op.join(buildpath(),'jobs')
        if not op.isdir(jobDir): mkdir(jobDir)
        # make a script to load and preprocess that file, then save as .mat
        thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(getcwd())
        thispythonfn += 'from helpers import *\n'
        thispythonfn += 'config.subject          = "{}"\n'.format(config.subject)
        thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
        thispythonfn += 'config.fmriRun          = "{}"\n'.format(config.fmriRun)
        thispythonfn += 'config.useFIX           = {}\n'.format(config.useFIX)
        thispythonfn += 'config.pipelineName     = "{}"\n'.format(config.pipelineName)
        thispythonfn += 'config.overwrite        = {}\n'.format(config.overwrite)
        thispythonfn += 'config.queue            = {}\n'.format(config.queue)
        thispythonfn += 'config.preWhitening     = {}\n'.format(config.preWhitening)
        thispythonfn += 'config.isCifti          = {}\n'.format(config.isCifti)
        thispythonfn += 'config.Operations       = {}\n'.format(config.Operations)
        thispythonfn += 'config.suffix           = "{}"\n'.format(config.suffix)
        thispythonfn += 'config.ext              = "{}"\n'.format(config.ext)
        thispythonfn += 'config.fmriFile         = "{}"\n'.format(config.fmriFile)
        thispythonfn += 'config.logfile          = "{}"\n'.format(op.join(jobDir,'log.txt'))
        thispythonfn += 'config.parcellationName = "{}"\n'.format(config.parcellationName)
        thispythonfn += 'config.parcellationFile = "{}"\n'.format(config.parcellationFile)
        thispythonfn += 'config.nParcels         = {}\n'.format(config.nParcels)
        thispythonfn += 'runPipeline()\nEND'

        jobName = 's{}_{}'.format(config.subject,config.pipelineName)
        # prepare a script
        thisScript=op.join(jobDir,jobName+'.sh')
        try:
            remove(thisScript)
        except OSError:
            pass
	with open(thisScript,'w') as fidw:
            fidw.write('#!/bin/bash\n')
            fidw.write('echo ${FSLSUBALREADYRUN}\n')
            fidw.write('python {}'.format(thispythonfn))
        cmd='chmod 774 '+thisScript
        call(cmd,shell=True)
        # call to fnSubmitToCluster
        JobID = fnSubmitToCluster(thisScript,jobDir, jobName, '-p {} -l h_vmem={}'.format(priority,config.maxvmem))
        config.joblist.append(JobID)
    else:
        config.logfile = op.join(config.DATADIR,'log.txt')
        runPipeline()

    return True
