import os
os.system("fslmaths temporal_lobe_mni_3T.nii.gz -thr 10 -bin temporal_lobe_mask_3T.nii.gz")
os.system("/usr/share/fsl/5.0/bin/flirt -in /home/mboos/SpeechEncoding/temporal_lobe_mask_3T.nii.gz -applyxfm -init /home/data/psyinf/pandora_dartmouth/data/templates/grpbold/xfm/mni2tmpl_12dof.mat -out /home/mboos/SpeechEncoding/temporal_lobe_mask_grp_3T.nii.gz -paddingsize 0.0 -interp nearestneighbour -ref /home/data/psyinf/pandora_dartmouth/data/templates/grpbold/head.nii.gz")

os.system("fslmaths temporal_lobe_mni_7T.nii.gz -thr 10 -bin temporal_lobe_mask_head.nii.gz")
os.system("/usr/share/fsl/5.0/bin/flirt -in /home/mboos/SpeechEncoding/temporal_lobe_mask_7T.nii.gz -applyxfm -init /home/data/psyinf/pandora_dartmouth/data/templates/grpbold/xfm/mni2tmpl_12dof.mat -out /home/mboos/SpeechEncoding/temporal_lobe_mask_grp_7T.nii.gz -paddingsize 0.0 -interp nearestneighbour -ref /home/data/psyinf/pandora_dartmouth/data/templates/grpbold/head.nii.gz")

for i in xrange(1,20):
	os.system("""applywarp -i temporal_lobe_mask_gr_7T.nii.gz -o temporal_lobe_mask_head_subj%02dbold.nii.gz -r /home/data/psyinf/forrest_gump/anondata/sub0%02d/templates/bold7Tp1/head.nii.gz -w /home/data/psyinf/forrest_gump/anondata/sub0%02d/templates/bold7Tp1/in_grpbold7Tp1/tmpl2subj_warp.nii.gz  --interp=nn""" % (i,i,i))

for i in xrange(1,20):
	os.system("""applywarp -i temporal_lobe_mask_grp_3T.nii.gz -o temporal_lobe_mask_3T_subj%02dbold.nii.gz -r /home/data/psyinf/pandora_dartmouth/data/sub0%02d/templates/bold/head.nii.gz -w /home/data/psyinf/pandora_dartmouth/data/sub0%02d/templates/bold/in_grpbold/tmpl2subj_warp.nii.gz --interp=nn""" % (i,i,i))

