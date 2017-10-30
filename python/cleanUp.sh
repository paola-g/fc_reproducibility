#!/bin/sh
RETURNHERE=$pwd
cd ~/scratch/data/HCP/MRI
for SUB in `ls -d ??????`;do 
	for SES in rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL; do 
		for FILE in `ls ${SUB}/MNINonLinear/Results/${SES}/shen2013/${SES}.nii.gz/allParcels_????????.txt`;do 
			if [ ! -e ${SUB}/MNINonLinear/Results/${SES}/${FILE: -12:8}.xml ];then 
				echo "cleaning up ${SUB},${SES},${FILE: -12:8}"
				rm ${SUB}/MNINonLinear/Results/${SES}/shen2013/${SES}.nii.gz/*${FILE: -12:8}*;
			fi;
		done;
	done;
done
cd $RETURNHERE
