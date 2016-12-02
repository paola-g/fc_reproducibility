from init_par import *

def Finn_loadandpreprocess(fmriFile, parcellation, overwrite):
    subject = op.basename(op.dirname(op.dirname(op.dirname(op.dirname(fmriFile)))))
    fmriRun = op.basename(op.dirname(fmriFile))
    ResultsDir = op.join(DATADIR,'Testing','Results')
    if not op.isdir(ResultsDir): makedirs(ResultsDir)
    ResultsDir = op.join(ResultsDir,'Finn')
    if not op.isdir(ResultsDir): mkdir(ResultsDir)

    if parcellation=='shenetal_neuroimage2013':
        uniqueParcels = range(268)
    else:
        print "Invalid parcellation code"
        return
    
    for iParcel in range(len(uniqueParcels)):
        parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.nii.gz'.format(iParcel))
        if not op.isfile(parcelMaskFile):
            print 'Making a binary volume mask for each parcel'
            mymaths = fsl.maths.MathsCommand(in_file=op.join(PARCELDIR, parcellation,'fconn_atlas_150_2mm.nii'),
					     out_file=parcelMaskFile, args='-thr {:.1f} -uthr {:.1f}'.format(iParcel-0.1, iParcel+0.1)) 
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
            parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.nii.gz'.format(iParcel))
            GMmaskFile = op.join(testpath(subject,fmriRun),'GMmask.nii.gz')
            # intersect GM & parcel
            parcelGMMaskFile = op.join(subjectParcelDir,parcellation,'GMparcel{:03d}.nii.gz'.format(iParcel))
            mymaths = fsl.maths.MathsCommand(in_file=parcelMaskFile,
					     out_file=parcelGMMaskFile, args='-mul '+GMmaskFile)
            mymaths.run()
            tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel))
            if not op.isfile(tsFile):
                # simply average the voxels within the mask
                meants1 = fsl.ImageMeants(in_file=fmriFile_prepro, out_file=tsFile, mask=parcelMaskFile)
                meants1.run()
                
            tsFile = op.join(tsDir,'GMparcel{:03d}.txt'.format(iParcel))    
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

        # Print out text file for each polynomial to be used as a regressor
        for i in num_pol:
            np.savetxt(op.join(testpath(subject,fmriRun),
			       'poly_detrend_' + str(i+1) + '.txt'), y[i],fmt='%0.02f')
            
        # ** b) use feat to regress them out
        # keep only WM/CSF voxels to speed things up
        WMCSFmaskFile = op.join(testpath(subject,fmriRun),'WMCSFmask.nii.gz')
        if not op.isfile(op.join(testpath(subject,fmriRun),fmriRun+'_WMCSF.nii.gz')):
            mymask1 = fsl.maths.ApplyMask(in_file=fmriFile, mask_file=WMCSFmaskFile,
					  out_file=op.join(testpath(subject,fmriRun),fmriRun+'_WMCSF.nii.gz'))
            mymask1.run()
        
        # copy and alter detrendpoly3.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step1.fsf')
        copyfile(op.join('detrendpoly3.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'step1.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'\
	.format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'\
	.format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'\
	.format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),fmriRun+'_WMCSF.nii.gz'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'poly_detrend_1.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom2) /c\\set fmri(custom2) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'poly_detrend_2.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom3) /c\\set fmri(custom3) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'poly_detrend_3.txt'),fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step1_outFile = op.join(testpath(subject,fmriRun),
				'step1.feat','stats','res4d.nii.gz')
        if not op.isfile(step1_outFile):
            myfeat1 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat1.run()
        
        ## 2. Regress CSF/WM signal from gray matter voxels
        print 'Step 2 (regress WMCSF signal from GM)'
        # ** a) extract the WM/CSF data from the detrended volume
        WMCSFtxtFileout = op.join(testpath(subject,fmriRun),
				  'step1.feat','stats','WMCSF.txt')
        if not op.isfile(WMCSFtxtFileout):
            meants1 = fsl.ImageMeants(in_file=step1_outFile, out_file=WMCSFtxtFileout, mask=WMCSFmaskFile)
            meants1.run()
        
        # ** b) keep only GM voxels to speed things up
        GMmaskFile = op.join(testpath(subject,fmriRun),'GMmask.nii.gz')
        if not op.isfile(op.join(testpath(subject,fmriRun),fmriRun+'_GM.nii.gz')):
            mymask2 = fsl.maths.ApplyMask(in_file=fmriFile, mask_file=GMmaskFile,
					  out_file=op.join(testpath(subject,fmriRun),fmriRun+'_GM.nii.gz'))
            mymask2.run()
        
        # ** c) use feat to regress it out **
        # copy and alter regressWMCSF.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step2.fsf')
        copyfile(op.join('regressWMCSF.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'step2.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'\
	.format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'\
	.format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'\
	.format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),fmriRun+'_GM.nii.gz'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'\
	.format(WMCSFtxtFileout,fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step2_outFile = op.join(testpath(subject,fmriRun),
				'step2.feat','stats','res4d.nii.gz')
        if not op.isfile(step2_outFile):
            myfeat2 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat2.run()
        
        
        # add the results of steps 1 & 2 for input to the next stage
        step12_outFile = op.join(testpath(subject,fmriRun),'step1+2.nii.gz')
        if not op.isfile(step12_outFile):
            myadd1 = fsl.maths.BinaryMaths(in_file=step1_outFile, operation='add',
					   operand_file=step2_outFile, out_file=step12_outFile)
            myadd1.run()
            
        ## 3. Regress motion parameters (found in the Movement_Regressors_dt_txt
        # file from HCP)    
        print 'Step 3 (regress 12 motion parameters from whole brain)'
        # ** a) load the detrended motion parameters
        motionFile = op.join(buildpath(subject,fmriRun),
			     'Movement_Regressors_dt.txt')
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
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'step3.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'\
	.format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'\
	.format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'\
	.format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'\
	.format(step12_outFile,fsfFile)
        call(cmd,shell=True)
        for iCol in range(len(colNames)):
            cmd = 'sed -i \'/set fmri(custom{}) /c\\set fmri(custom{}) "{}"\' {}'\
	    .format(iCol+1,iCol+1,motionFile.replace('.txt','_'+colNames[iCol]+'.txt'),fsfFile)
            call(cmd,shell=True)   
            
        # run feat
        step3_outFile = op.join(testpath(subject,fmriRun),
				'step3.feat','stats','res4d.nii.gz')
        if not op.isfile(step3_outFile):
            myfeat3 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat3.run()

        ## 4. Temporal smoothing with Gaussian kernel (sigma = 1 TR)
        print 'Step 4 (temporal smoothing with Gaussian kernel)'
        step4_outFile = op.join(testpath(subject,fmriRun),'step4.nii.gz')
        if not op.isfile(step4_outFile):
            myfilter = fsl.maths.TemporalFilter(in_file=step3_outFile,highpass_sigma=0, lowpass_sigma=1,
						out_file=step4_outFile)
            myfilter.run()
        ## 5. Regress temporal drift from gray matter (3rd order polynomial)
        print ('Step 5 (detrend gray matter voxels, polynomial order 3)')
        GMmaskfile = op.join(testpath(subject,fmriRun),'GMmask.nii.gz')
        if not op.isfile(step4_outFile.replace('.nii.gz','_GM.nii.gz')):
            mymask3 = fsl.maths.ApplyMask(in_file=step4_outFile, mask_file=GMmaskFile,
					  out_file=step4_outFile.replace('.nii.gz','_GM.nii.gz'))
            mymask3.run()
        WMCSFmaskFile = op.join(testpath(subject,fmriRun),'WMCSFmask.nii.gz') 
        if not op.isfile(step4_outFile.replace('.nii.gz','_WMCSF.nii.gz')):
            mymask4 = fsl.maths.ApplyMask(in_file=step4_outFile, mask_file=WMCSFmaskFile,
					  out_file=step4_outFile.replace('.nii.gz','_WMCSF.nii.gz'))
            mymask4.run()
        
        # ** b) use feat to regress them out **
        # copy and alter detrendpoly3.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step5.fsf')
        copyfile(op.join('detrendpoly3.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'step5.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'\
	.format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'\
	.format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'\
	.format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'\
	.format(step4_outFile.replace('.nii.gz','_GM.nii.gz'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'poly_detrend_1.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom2) /c\\set fmri(custom2) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'poly_detrend_2.txt'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom3) /c\\set fmri(custom3) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'poly_detrend_3.txt'),fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step5tmp_outFile = op.join(testpath(subject,fmriRun),
				   'step5.feat','stats','res4d.nii.gz')
        if not op.isfile(step5tmp_outFile):
            myfeat4 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat4.run()
            
        # add the WMCSF voxels back in for input to the next stage
        step5_outFile = op.join(testpath(subject,fmriRun),'step5.nii.gz')
        if not op.isfile(step5_outFile):
            myadd2 = fsl.maths.BinaryMaths(in_file=step5tmp_outFile, operation='add',
					   operand_file=step4_outFile.replace('.nii.gz','_WMCSF.nii.gz'), out_file=step5_outFile)
            myadd2.run()
            
        ## 6. Regress global mean (mask includes all voxels in brain mask,
        # gray matter, white matter and CSF
        print 'Step 6 (GSR)'
        # ** a) extract the WM/CSF/GM data from detrended volume
        WMCSFGMmaskFile = op.join(testpath(subject,fmriRun),'WMCSFGMmask.nii.gz')
        WMCSFGMtxtFileout = op.join(testpath(subject,fmriRun),
				    'step5.feat','stats','WMCSFGM.txt')
        if not op.isfile(WMCSFGMtxtFileout):
            meants2 = fsl.ImageMeants(in_file=step5_outFile, out_file=WMCSFGMtxtFileout, mask=WMCSFGMmaskFile)
            meants2.run()
            
        # ** c) use feat to regress it out
        # copy and alter regressWMCSF.fsf
        fsfFile = op.join(testpath(subject,fmriRun), 'step6.fsf')
        copyfile(op.join('regressWMCSF.fsf'), fsfFile)
        cmd = 'sed -i \'/set fmri(outputdir) /c\\set fmri(outputdir) "{}"\' {}'\
	.format(op.join(testpath(subject,fmriRun),'step6.feat'),fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(tr) /c\\set fmri(tr) {:.3f}\' {}'\
	.format(TR,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(npts) /c\\set fmri(npts) {}\' {}'\
	.format(nTRs,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(totalVoxels) /c\\set fmri(totalVoxels) {}\' {}'\
	.format(dim1*dim2*dim3*nTRs,fsfFile)
        call(cmd,shell=True) 
        cmd = 'sed -i \'/set feat_files(1) /c\\set feat_files(1) "{}"\' {}'\
	.format(step5_outFile,fsfFile)
        call(cmd,shell=True)
        cmd = 'sed -i \'/set fmri(custom1) /c\\set fmri(custom1) "{}"\' {}'\
	.format(WMCSFGMtxtFileout,fsfFile)
        call(cmd,shell=True)
        
        # run feat
        step6_outFile = op.join(testpath(subject,fmriRun),
				'step6.feat','stats','res4d.nii.gz')
        if not op.isfile(step6_outFile):
            myfeat5 = fsl.FEAT(fsf_file=fsfFile,terminal_output='none')
            myfeat5.run()
            
        ## We're done! Copy the resulting file
        copyfile(step6_outFile,op.join(testpath(subject,fmriRun),
				       fmriRun+'_FinnPrepro.nii.gz'))
