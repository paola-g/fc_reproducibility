from setupPipeline import *

def runPipeline(subject, fmriRun, fmriFile):
    
    timeStart = localtime()
    
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
            if 'Regression' in step[0]:
                if step[0]=='TissueRegression':
                    niiImg = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
                else:
                    r0 = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
                    niiImg = regress(niiImg, nTRs, r0, keepMean)
            else:
                niiImg = Hooks[step[0]](niiImg, Flavors[i][0], masks, imgInfo[1:])
        else:
            r = np.empty((nTRs, 0))
            for j in range(len(step)):
                opr = step[j]
                if 'Regression' in opr:
                    if opr=='TissueRegression':
                        niiImg = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                    else:    
                        r0 = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
                        r = np.append(r, r0, axis=1)
                else:
                    niiImg = Hooks[opr](niiImg, Flavors[i][j], masks, imgInfo[1:])
            if r.shape[1] > 0:
                niiImg = regress(niiImg, nTRs, r, keepMean)    
        niiImg[np.isnan(niiImg)] = 0

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
