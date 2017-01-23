from setupPipeline import *
from nilearn.input_data import NiftiLabelsMasker
def runPipeline(subject, fmriRun, fmriFile):
    
    timeStart = localtime()
    
    if parcellation=='shenetal_neuroimage2013':
        uniqueParcels = range(268)
        isCifti = 0
        parcelVolume = 'fconn_atlas_150_2mm.nii'
    elif parcellation=='Glasser_Aseg_Suit':
        isCifti = 1
        parcelVolume = 'Parcels.dlabel.nii'
        uniqueParcels = range(405)
    else:
        print "Invalid parcellation code"
        return

    print 'Step 0'
    print 'Building WM, CSF and GM masks...'
    masks = makeTissueMasks(subject,fmriRun,False)
    maskAll, maskWM_, maskCSF_, maskGM_ = masks

    print 'Loading data in memory...'
    imgInfo = load_img(fmriFile, maskAll)
    niiImg, nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
    nsteps = len(steps)
    for i in range(1,nsteps+1):
        step = steps[i]
        print 'Step '+str(i)+' '+str(step[0])
        
        if len(step) == 1:
            # Atomic operations
            if 'Regression' in step[0] or ('TemporalFiltering' in step[0] and 'DCT' in Flavors[i][0]):
                if step[0]=='TissueRegression': #regression constrained to GM
                    niiImg = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
                else:
                    r0 = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
                    niiImg = regress(niiImg, nTRs, TR, r0, keepMean)
            else:
                niiImg = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
        else:
            # When multiple regression steps have the same order, all the regressors are combined
            # and a single regression is performed (other operations are executed in order)
            r = np.empty((nTRs, 0))
            for j in range(len(step)):
                opr = step[j]
                if 'Regression' in opr or ('TemporalFiltering' in opr and 'DCT' in Flavors[i][j]):
                    if opr=='TissueRegression': #regression constrained to GM
                        niiImg = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                    else:    
                        r0 = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                        r = np.append(r, r0, axis=1)
                else:
                    niiImg = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
            if r.shape[1] > 0:
                niiImg = regress(niiImg, nTRs, TR, r, keepMean)    
        niiImg[np.isnan(niiImg)] = 0

    print 'Done! Copy the resulting file...'
    rstring = ''.join(random.SystemRandom().choice(string.ascii_lowercase +string.ascii_uppercase + string.digits) for _ in range(8))
    outFile = fmriRun+'_'+rstring
    if isCifti:
        # write to text file
        np.savetxt(op.join(buildpath(subject,fmriRun),outfile+'.tsv'),niiImg, delimiter='\t', fmt='%.6f')
        # need to convert back to cifti
        cmd = 'wb_command -cifti-convert -from-text {} {} {}'.format(op.join(buildpath(subject,fmriRun),'.tsv'),
                                                                     fmriFile,op.join(buildpath(subject,fmriRun),outFile+'.dtseries.nii'))
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
        nib.save(newimg,op.join(buildpath(subject,fmriRun),outFile+'.nii.gz'))
        del niiimg 

    timeEnd = localtime()  

    outXML = rstring+'.xml'
    conf2XML(fmriFile, DATADIR, sortedOperations, timeStart, timeEnd, op.join(buildpath(subject,fmriRun),outXML))
    print 'Preprocessing complete. Starting FC computation...'
    # After preprocessing, functional connectivity is computed
    ResultsDir = op.join(DATADIR,'Results')
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    ResultsDir = op.join(ResultsDir,pipelineName)
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    ResultsDir = op.join(ResultsDir,parcellation)
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    
    for iParcel in uniqueParcels:
        if not isCifti:
            parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.nii.gz'.format(iParcel+1))
            if not op.isfile(parcelMaskFile):
                print 'Making a binary volume mask for each parcel'
                mymaths1 = fsl.maths.MathsCommand(in_file=op.join(PARCELDIR, parcellation,'fconn_atlas_150_2mm.nii'),\
                    out_file=parcelMaskFile, args='-thr {:.1f} -uthr {:.1f}'.format(iParcel+1-0.1, iParcel+1+0.1)) 
                mymaths1.run()
    if not op.isfile(fmriFile):
        print fmriFile, 'does not exist'
        return
    
    tsDir = op.join(buildpath(subject,fmriRun),parcellation)
    if not op.isdir(tsDir): mkdir(tsDir)
    alltsFile = op.join(ResultsDir,subject+'_'+fmriRun+'.txt')
    print tsDir, alltsFile
    #masker = NiftiLabelsMasker(labels_img=op.join(PARCELDIR, parcellation,'fconn_atlas_150_2mm.nii'))
    #time_series = masker.fit_transform(op.join(buildpath(subject,fmriRun), outFile+'.nii.gz'))
    #print time_series.shape
    #np.savetxt('mytimeseries.txt', time_series)
    if not (op.isfile(alltsFile)) or overwrite:            
        # calculate signal in each of the nodes by averaging across all voxels/grayordinates in node
        print 'Extracting mean data from',str(len(uniqueParcels)),'parcels for ',outFile
        for iParcel in uniqueParcels:
            tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel+1))
            if not op.isfile(tsFile):
                if not isCifti:
                    parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.nii.gz'.format(iParcel+1))
                    
                    # simply average the voxels within the mask
                    meants1 = fsl.ImageMeants(in_file=op.join(buildpath(subject,fmriRun), outFile+'.nii.gz'), out_file=tsFile, mask=parcelMaskFile)
                    meants1.run()
                else:
                    # extract data in the parcel
                    parcelMaskFile = op.join(PARCELDIR,parcellation,'parcel{:03d}.dscalar.nii'.format(iParcel+1))
                    cmd = 'wb_command -cifti-label-to-roi {} {} -key {}'.format(
                        op.joinpath(PARCELDIR,parcellation,parcelVolume), parcelMaskFile,iParcel+1)
                    call(cmd,shell=True)
                    cmd = 'wb_command -cifti-roi-average {} {} -cifti-roi {}'.format(
                        op.join(buildpath(subject,fmriRun), outFile+'.nii.gz'),tsFile, parcelMaskFile)
                
                
    # concatenate all ts
    print 'Concatenating data'
    cmd = 'paste '+op.join(tsDir,'parcel*.txt')+' > '+alltsFile
    call(cmd, shell=True)
