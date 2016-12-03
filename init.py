
# coding: utf-8

# ### Required libraries

# In[21]:
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



# ### Parameters

# In[10]:

behavFile = 'unrestricted_luckydjuju_11_17_2015_0_47_11.csv'
release = 'Q2'
outScore = 'PMAT24_A_CR'
DATADIR = '/data/jdubois/data/HCP/MRI'
PARCELDIR = '/data/jdubois/data/HCP/MRI/parcellations'
parcellation = 'shenetal_neuroimage2013'
overwrite = False
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
        WMmask = np.double(np.in1d(ribbon, ribbonWMstructures).tolist() or                            np.in1d(wmparc, wmparcWMstructures).tolist() or                            np.in1d(wmparc, wmparcCCstructures).tolist() and                            not np.in1d(wmparc, wmparcCSFstructures).tolist() and                            not np.in1d(wmparc, wmparcGMstructures).tolist())
        CSFmask = np.double(np.in1d(wmparc, wmparcCSFstructures))
        WMCSFmask = np.double((WMmask > 0) | (CSFmask > 0))
        GMmask = np.double(np.in1d(ribbon,ribbonGMstrucures).tolist() or         np.in1d(wmparc,wmparcGMstructures).tolist())
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


# In[23]:

def Finn_preprocess(fmriFile):
    subject = op.basename(op.dirname(op.dirname(op.dirname(op.dirname(fmriFile)))))
    fmriRun = op.basename(op.dirname(fmriFile))
    if not op.isfile(op.join(testpath(subject,fmriRun),fmriRun+'_FinnPrepro.nii.gz')):
        # make WM, CSF, GM masks (if not already done)
        makeTissueMasks(subject,fmriRun,False)
        # get some info
        img = nib.load(fmriFile)
        hdr = img.header.structarr
        nTRs = hdr['dim'][4]
        # retrieve TR
        TR = hdr['pixdim'][4]
        # retrieve dimensions
        dim1 = hdr['dim'][1]
        dim2 = hdr['dim'][2]
        dim3 = hdr['dim'][3]
        ## DO PREPROCESSING:
        ## 1) Regress temporal drift from CSF and white matter (3rd order polynomial)
        print 'Step 1 (detrend WMCSF voxels, polynomial order 3)'
        # ** a) create polynomial regressors **
        nPoly = 3
        # Create polynomial regressors centered on 0 and bounded by -1 and 1
        x = np.arange(nTRs)
        num_pol = range(nPoly)
        y = np.ones((len(num_pol),len(x)))

        plt.figure()
        plt.title('Polynomial regressors')

        for i in num_pol:
            y[i,:] = (x - (np.max(x)/2)) **(i+1)
            y[i,:] = y[i,:] - np.mean(y[i,:])
            y[i,:] = y[i,:]/np.max(y[i,:])    
            if(doPlot): plt.plot(x,y[i])
        
        if(doPlot):
            plt.ylim((-1,1))
            plt.xlim((-0.1,nTRs+0.1))
            plt.show()
        
        # Print out text file for each polynomial to be used as a regressor
        for i in num_pol:
            np.savetxt(op.join(testpath(subject,fmriRun),                        'poly_detrend_' + str(i+1) + '.txt'), y[i],fmt='%0.02f')
            
        # ** b) use feat to regress them out
        # keep only WM/CSF voxels to speed things up
        WMCSFmaskFile = op.join(testpath(subject,fmriRun),'WMCSFmask.nii.gz')
        if not op.isfile(op.join(testpath(subject,fmriRun),fmriRun+'_WMCSF.nii.gz')):
            mymask1 = fsl.maths.ApplyMask(in_file=fmriFile, mask_file=WMCSFmaskFile,                                               out_file=op.join(testpath(subject,fmriRun),fmriRun+'_WMCSF.nii.gz'))
            mymask1.run()
        
        # copy and alter detrendpoly3.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step1.fsf')
        copyfile(op.join('detrendpoly3.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'step1.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'            .format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'            .format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'            .format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),fmriRun+'_WMCSF.nii.gz'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'poly_detrend_1.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom2) /c\\set fmri(custom2) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'poly_detrend_2.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom3) /c\\set fmri(custom3) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'poly_detrend_3.txt'),fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step1_outFile = op.join(testpath(subject,fmriRun),                               'step1.feat','stats','res4d.nii.gz')
        if not op.isfile(step1_outFile):
            myfeat1 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat1.run()
        
        ## 2. Regress CSF/WM signal from gray matter voxels
        print 'Step 2 (regress WMCSF signal from GM)'
        # ** a) extract the WM/CSF data from the detrended volume
        WMCSFtxtFileout = op.join(testpath(subject,fmriRun),                                  'step1.feat','stats','WMCSF.txt')
        if not op.isfile(WMCSFtxtFileout):
            meants1 = fsl.ImageMeants(in_file=step1_outFile, out_file=WMCSFtxtFileout, mask=WMCSFmaskFile)
            meants1.run()
        
        # ** b) keep only GM voxels to speed things up
        GMmaskFile = op.join(testpath(subject,fmriRun),'GMmask.nii.gz')
        if not op.isfile(op.join(testpath(subject,fmriRun),fmriRun+'_GM.nii.gz')):
            mymask2 = fsl.maths.ApplyMask(in_file=fmriFile, mask_file=GMmaskFile,                                               out_file=op.join(testpath(subject,fmriRun),fmriRun+'_GM.nii.gz'))
            mymask2.run()
        
        # ** c) use feat to regress it out **
	fsfFile = op.join(testpath(subject,fmriRun), 'step2.fsf')
        copyfile(op.join('regressWMCSF.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'step2.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'            .format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'            .format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'            .format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),fmriRun+'_GM.nii.gz'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'            .format(WMCSFtxtFileout,fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step2_outFile = op.join(testpath(subject,fmriRun),                               'step2.feat','stats','res4d.nii.gz')
        if not op.isfile(step2_outFile):
            myfeat2 = fsl.FEAT(fsf_file=fsfFile)
            myfeat2.run()
        
        
        # add the results of steps 1 & 2 for input to the next stage
        step12_outFile = op.join(testpath(subject,fmriRun),'step1+2.nii.gz')
        if not op.isfile(step12_outFile):
            myadd1 = fsl.maths.BinaryMaths(in_file=step1_outFile, operation='add',                                         operand_file=step2_outFile, out_file=step12_outFile)
            myadd1.run()
            
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
                
        # ** b) use feat to regress them out **
        # copy and alter regressM12.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step3.fsf')
        copyfile(op.join('regressM12.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'step3.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'            .format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'            .format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'            .format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'            .format(step12_outFile,fsfFile)
        call(cmd,shell=True)
        for iCol in range(len(colNames)):
            cmd = 'sed -i \'/set fmri(custom{}) /c\\set fmri(custom{}) "{}"\' {}'                .format(iCol+1,iCol+1,motionFile.replace('.txt','_'+colNames[iCol]+'.txt'),fsfFile)
            call(cmd,shell=True)   
            
        # run feat
        step3_outFile = op.join(testpath(subject,fmriRun),                               'step3.feat','stats','res4d.nii.gz')
        if not op.isfile(step3_outFile):
            myfeat3 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat3.run()

        ## 4. Temporal smoothing with Gaussian kernel (sigma = 1 TR)
        print 'Step 4 (temporal smoothing with Gaussian kernel)'
        step4_outFile = op.join(testpath(subject,fmriRun),'step4.nii.gz')
        if not op.isfile(step4_outFile):
            myfilter = fsl.maths.TemporalFilter(in_file=step3_outFile,highpass_sigma=0, lowpass_sigma=1,                                               out_file=step4_outFile)
            myfilter.run()
            
        ## 5. Regress temporal drift from gray matter (3rd order polynomial)
        print ('Step 5 (detrend gray matter voxels, polynomial order 3)')
        GMmaskfile = op.join(testpath(subject,fmriRun),'GMmask.nii.gz')
        if not op.isfile(step4_outFile.replace('.nii.gz','_GM.nii.gz')):
            mymask3 = fsl.maths.ApplyMask(in_file=step4_outFile, mask_file=GMmaskFile,                                               out_file=step4_outFile.replace('.nii.gz','_GM.nii.gz'))
            mymask3.run()
        WMCSFmaskFile = op.join(testpath(subject,fmriRun),'WMCSFmask.nii.gz') 
        if not op.isfile(step4_outFile.replace('.nii.gz','_WMCSF.nii.gz')):
            mymask4 = fsl.maths.ApplyMask(in_file=step4_outFile, mask_file=WMCSFmaskFile,                                               out_file=step4_outFile.replace('.nii.gz','_WMCSF.nii.gz'))
            mymask4.run()
        
        # ** b) use feat to regress them out **
        # copy and alter detrendpoly3.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step5.fsf')
        copyfile(op.join('detrendpoly3.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'step5.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'            .format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'            .format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'            .format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'            .format(step4_outFile.replace('.nii.gz','_GM.nii.gz'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'poly_detrend_1.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom2) /c\\set fmri(custom2) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'poly_detrend_2.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom3) /c\\set fmri(custom3) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'poly_detrend_3.txt'),fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step5tmp_outFile = op.join(testpath(subject,fmriRun),                               'step5.feat','stats','res4d.nii.gz')
        if not op.isfile(step5tmp_outFile):
            myfeat4 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat4.run()
            
        # add the WMCSF voxels back in for input to the next stage
        step5_outFile = op.join(testpath(subject,fmriRun),'step5.nii.gz')
        if not op.isfile(step5_outFile):
            myadd2 = fsl.maths.BinaryMaths(in_file=step5tmp_outFile, operation='add',                operand_file=step4_outFile.replace('.nii.gz','_WMCSF.nii.gz'), out_file=step5_outFile)
            myadd2.run()
            
        ## 6. Regress global mean (mask includes all voxels in brain mask,
        # gray matter, white matter and CSF
        print 'Step 6 (GSR)'
        # ** a) extract the WM/CSF/GM data from detrended volume
        WMCSFGMmaskFile = op.join(testpath(subject,fmriRun),'WMCSFGMmask.nii.gz')
        WMCSFGMtxtFileout = op.join(testpath(subject,fmriRun),                                   'step5.feat','stats','WMCSFGM.txt')
        if not op.isfile(WMCSFGMtxtFileout):
            meants2 = fsl.ImageMeants(in_file=step5_outFile, out_file=WMCSFGMtxtFileout, mask=WMCSFGMmaskFile)
            meants2.run()
            
        # ** c) use feat to regress it out
        # copy and alter regressWMCSF.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step6.fsf')
        copyfile(op.join('regressWMCSF.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'            .format(op.join(testpath(subject,fmriRun),'step6.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'            .format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'            .format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'            .format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'            .format(step5_outFile,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'            .format(WMCSFGMtxtFileout,fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step6_outFile = op.join(testpath(subject,fmriRun),                               'step6.feat','stats','res4d.nii.gz')
        if not op.isfile(step6_outFile):
            myfeat5 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat5.run()
            
        ## We're done! Copy the resulting file
        copyfile(step6_outFile,op.join(testpath(subject,fmriRun),                                      fmriRun+'_FinnPrepro.nii.gz'))
        
            


# In[26]:

def Finn_loadandpreprocess(fmriFile, parcellation, overwrite):
    subject = op.basename(op.dirname(op.dirname(op.dirname(op.dirname(fmriFile)))))
    fmriRun = op.basename(op.dirname(fmriFile))
    ResultsDir = op.join(DATADIR,'Testing','Results')
    if not op.isdir(ResultsDir): makedirs(ResultsDir)
    ResultsDir = op.join(ResultsDir,'Finn')
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    ResultsDir = op.join(ResultsDir,parcellation)
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    if parcellation=='shenetal_neuroimage2013':
        uniqueParcels = range(268)
    else:
        print "Invalid parcellation code"
        return
    
    for iParcel in range(len(uniqueParcels)):
        parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.nii.gz'.format(iParcel+1))
        if not op.isfile(parcelMaskFile):
            print 'Making a binary volume mask for each parcel'
            mymaths = fsl.maths.MathsCommand(in_file=op.join(PARCELDIR, parcellation,'fconn_atlas_150_2mm.nii'),                out_file=parcelMaskFile, args='-thr {:.1f} -uthr {:.1f}'.format(iParcel+1-0.1, iParcel+1+0.1)) 
            mymaths.run()
    if not op.isfile(fmriFile):
        print fmriFile, 'does not exist'
        return
    
    tsDir = op.join(testpath(subject,fmriRun),parcellation)
    if not op.isdir(tsDir): mkdir(tsDir)
    alltsFile = op.join(ResultsDir,subject+'_'+fmriRun+'.txt')
    alltsGMFile = op.join(ResultsDir,subject+'_'+fmriRun+'_GM.txt')
    if not (op.isfile(alltsFile)) or not (op.isfile(alltsGMFile)) or overwrite:
        fmriFile_prepro = op.join(testpath(subject,fmriRun), fmriRun+'_FinnPrepro.nii.gz')
        # make WM, CSF, GM masks
        if not op.isfile(op.join(testpath(subject,fmriRun), fmriRun+'GMmask.nii.gz')):
            makeTissueMasks(subject,fmriRun,overwrite)
        
        # perform preprocessing (if not already done)
        if not op.isfile(fmriFile_prepro):
            Finn_preprocess(fmriFile)
            
        # calculate signal in each of the nodes by averaging across all voxels in node
        print 'Extracting mean data from',str(len(uniqueParcels)),'parcels for ',fmriFile_prepro
        #subjectParcelDir = op.join(DATADIR,subject,'MNINonLinear','Results','parcellations')
	subjectParcelDir = op.join(DATADIR,'Testing',subject,'Results','parcellations')

        if not op.isdir(subjectParcelDir): mkdir(subjectParcelDir)
        if not op.isdir(op.join(subjectParcelDir,parcellation)): mkdir(op.join(subjectParcelDir,parcellation))
        
        for iParcel in range(len(uniqueParcels)):
            parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.nii.gz'.format(iParcel+1))
            GMmaskFile = op.join(testpath(subject,fmriRun),'GMmask.nii.gz')
            # intersect GM & parcel
            parcelGMMaskFile = op.join(subjectParcelDir,parcellation,'GMparcel{:03d}.nii.gz'.format(iParcel+1))
            mymaths = fsl.maths.MathsCommand(in_file=parcelMaskFile,                out_file=parcelGMMaskFile, args='-mul '+GMmaskFile)
            mymaths.run()
            tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel+1))
            if not op.isfile(tsFile):
                # simply average the voxels within the mask
                meants1 = fsl.ImageMeants(in_file=fmriFile_prepro, out_file=tsFile, mask=parcelMaskFile)
                meants1.run()
                
            tsFile = op.join(tsDir,'GMparcel{:03d}.txt'.format(iParcel+1))    
            if not op.isfile(tsFile):
                # simply average the voxels within the mask
                meants2 = fsl.ImageMeants(in_file=fmriFile_prepro, out_file=tsFile, mask=parcelGMMaskFile)
                meants2.run()
                
        # concatenate all ts
        print 'Concatenating data'
        cmd = 'paste '+op.join(tsDir,'parcel*.txt')+' > '+alltsFile
        call(cmd, shell=True)
        cmd = 'paste '+op.join(tsDir,'GMparcel*.txt')+' > '+alltsGMFile
        call(cmd, shell=True)         
        



