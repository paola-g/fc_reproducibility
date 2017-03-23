from setupPipeline import *
def runPipeline(subject, fmriRun, fmriFile):
    if not op.isfile(fmriFile):
        print fmriFile, 'does not exist'
        return
    ResultsDir = op.join(DATADIR,'Results')
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    ResultsDir = op.join(ResultsDir,config.pipelineName)
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
    ResultsDir = op.join(ResultsDir,config.parcellation)
    if not op.isdir(ResultsDir): mkdir(ResultsDir)
        
    if config.parcellation=='shenetal_neuroimage2013':
        uniqueParcels = range(268)
        config.isCifti = 0
        parcelVolume = 'fconn_atlas_150_2mm.nii'
    elif config.parcellation=='shenetal_neuroimage2013_new':
        uniqueParcels = range(268)
        config.isCifti = 0
        parcelVolume = 'shen_2mm_268_parcellation.nii.gz'
    elif config.parcellation=='Glasser_Aseg_Suit':
        config.isCifti = 1
        parcelVolume = 'Parcels.dlabel.nii'
        uniqueParcels = range(405)
    elif config.parcellation=='Glasser_CIT168Amy_Aseg_Suit':
        config.isCifti = 1
        parcelVolume = 'Parcels.dlabel.nii'
        uniqueParcels = range(423)
    else:
        print "Invalid parcellation code"
        return

    print 'Step 0'
    print 'Building WM, CSF and GM masks...'
    masks = makeTissueMasks(subject,fmriRun,False)
    maskAll, maskWM_, maskCSF_, maskGM_ = masks    
        
    precomputed = checkXML(fmriFile,steps,Flavors,buildpath(subject, fmriRun)) 
    if precomputed and not config.overwrite:
        print "Preprocessing already computed. Using old file..."
        with open(precomputed, 'rb') as fFile:
            decompressedFile = gzip.GzipFile(fileobj=fFile)
            outFilePath = precomputed.replace('.nii.gz', '.nii')
            with open(outFilePath, 'wb') as outfile:
                outfile.write(decompressedFile.read())

	    img = nib.load(outFilePath)
	    myoffset = img.dataobj.offset
	    data = np.memmap(outFilePath, dtype=img.header.get_data_dtype(), mode='c', order='F',
                     offset=myoffset,shape=img.header.get_data_shape())

	    nRows, nCols, nSlices, nTRs = img.header.get_data_shape()
	    TR = img.header.structarr['pixdim'][4]
	    niiimg = data.reshape([nRows*nCols*nSlices, nTRs], order='F')
	    del data
    else:
        timeStart = localtime()

        print 'Loading data in memory...'
        imgInfo = load_img(fmriFile, maskAll)
        niiImg, nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        nsteps = len(steps)
        for i in range(1,nsteps+1):
            step = steps[i]
            print 'Step '+str(i)+' '+str(step[0])

            if len(step) == 1:
                # Atomic operations
                if 'Regression' in step[0] or ('TemporalFiltering' in step[0] and 'DCT' in Flavors[i][0]) or ('wholebrain' in Flavors[i][0]):
                    if step[0]=='TissueRegression' and 'GM' in Flavors[i][0]: #regression constrained to GM
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
                        if opr=='TissueRegression' and 'GM' in Flavors[i][j]: #regression constrained to GM
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
        outFile = fmriRun+'_'+rstring
        if config.isCifti:
            # write to text file
            np.savetxt(op.join(buildpath(subject,fmriRun),outFile+'.tsv'),niiImg, delimiter='\t', fmt='%.6f')
            # need to convert back to cifti
            cmd = 'wb_command -cifti-convert -from-text {} {} {}'.format(op.join(buildpath(subject,fmriRun),outFile+'.tsv'),
                                                                         fmriFile,
                                                                         op.join(buildpath(subject,fmriRun),outFile+'.dtseries.nii'))
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


	if hasattr(config, 'logfile'):
            f=open(config.logfile, "a+")
            f.write('{},{},{}\n'.format(subject,fmriRun,outFile))
            f.close()

        timeEnd = localtime()  

        outXML = rstring+'.xml'
        conf2XML(fmriFile, DATADIR, sortedOperations, timeStart, timeEnd, op.join(buildpath(subject,fmriRun),outXML))

    print 'Preprocessing complete. Starting FC computation...'
    # After preprocessing, functional connectivity is computed
    tsDir = op.join(buildpath(subject,fmriRun),config.parcellation)
    if not op.isdir(tsDir): mkdir(tsDir)
    alltsFile = op.join(ResultsDir,subject+'_'+fmriRun+'.txt')
    
    if not (op.isfile(alltsFile)) or config.overwrite:            
        # calculate signal in each of the nodes by averaging across all voxels/grayordinates in node
        print 'Extracting mean data from',str(len(uniqueParcels)),'parcels for ',outFile
        
        if not config.isCifti:
            with open(op.join(PARCELDIR, config.parcellation, parcelVolume), 'rb') as fFile:
                decompressedFile = gzip.GzipFile(fileobj=fFile)
                outFilePath = op.join(tsDir, 'temp_parcellation.nii')
                with open(outFilePath, 'wb') as outfile:
                    outfile.write(decompressedFile.read())
            tmpnii = nib.load(outFilePath)
            myoffset = tmpnii.dataobj.offset
            tdata = np.memmap(outFilePath, dtype=tmpnii.header.get_data_dtype(), mode='r', order='F',
                             offset=myoffset,shape=tmpnii.header.get_data_shape())
            tRows, tCols, tSlices = tmpnii.header.get_data_shape()
            allparcels = np.uint16(np.reshape(tdata,tRows*tCols*tSlices, order='F'))
            del tdata
            if niiimg.shape != (nRows*nCols*nSlices, nTRs):
                niiimg = np.reshape(niiimg, (nRows*nCols*nSlices, nTRs), order='F')
            
        else:    
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(op.join(PARCELDIR,parcellation,parcelVolume),
                                                                   op.join(PARCELDIR,parcellation,parcelVolume).replace('.dlabel.nii','.tsv'))
            call(cmd, shell=True)
            allparcels = np.loadtxt(op.join(PARCELDIR,parcellation,parcelVolume).replace('.dlabel.nii','.tsv'));
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(op.join(buildpath(subject,fmriRun),outFile+'.dtseries.nii'),
                                                                   op.join(buildpath(subject,fmriRun),outFile+'.tsv'))
            call(cmd, shell=True)
            niiimg = np.loadtxt(op.join(buildpath(subject,fmriRun),outFile+'.tsv'));
                
       
        for iParcel in uniqueParcels:
            tsFile = op.join(tsDir,'parcel{:03d}_{}.txt'.format(iParcel+1,rstring))
            if not op.isfile(tsFile) or config.overwrite:
                np.savetxt(tsFile,np.nanmean(niiimg[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.6f',delimiter='\n')
                
        # concatenate all ts
        print 'Concatenating data'
        cmd = 'paste '+op.join(tsDir,'parcel???_{}.txt'.format(rstring))+' > '+alltsFile
        call(cmd, shell=True)
        
        # delete decompressed input files
        try:
            remove(op.join(buildpath(subject, fmriRun), fmriRun+'.nii'))
        except OSError:
            pass
        try:
            remove(op.join(tsDir, 'temp_parcellation.nii')) 
        except OSError:
            pass
        del niiimg
