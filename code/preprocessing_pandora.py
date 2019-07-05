from __future__ import division
import os
import sys
import mvpa2.suite as mvpa
import numpy as np
import joblib
from nilearn.masking import unmask
from nibabel import save
from pandas import read_csv

#TODO: review preprocessing

cat_nrs = np.array(['ambient3', 'ambient4', 'ambient1', 'ambient0', 'ambient2',
'rocknroll4', 'rocknroll3', 'rocknroll2', 'rocknroll0',
'rocknroll1', 'metal0', 'metal4', 'metal3', 'metal1', 'metal2',
'symphonic1', 'symphonic3', 'symphonic2', 'symphonic0',
'symphonic4', 'country3', 'country0', 'country1', 'country2',
'country4'])

# sequence per participant
seqs = [[6,2,0,3,7,5,4,1],[1,5,0,7,3,2,4,6],[5,1,4,3,6,2,0,7],[0,4,6,2,5,7,1,3],[2,6,4,6,1,0,2,5],[6,3,2,0,7,1,4,5],[0,6,5,4,2,1,7,3],[6,1,7,0,3,4,5,2],[1,3,2,4,0,6,7,5],[0,7,2,4,5,1,3,6],[5,0,4,1,6,3,7,2],[7,3,1,2,6,5,0,4],[5,2,6,0,7,3,4,1],[0,2,7,1,5,6,4,3],[1,5,2,4,6,3,0,7],[6,1,7,0,2,3,5,4],[3,6,0,7,2,1,4,5],[3,7,4,1,0,5,2,6],[4,7,6,2,3,0,1,5]]

def preprocess_and_tmp_save_fmri_3T(datapath, task, subj, model, scratch_path, group_mask='/home/mboos/pandora_paper/temporal_lobe_mask_grp_3T.nii.gz'):
    '''preprocesses one subject from Pandora 3T
    aligns to group template
    run-wise linear de-trending and z-scoring'''
    from nipype.interfaces import fsl
    dhandle = mvpa.OpenFMRIDataset(datapath)
    mask_fname = os.path.join('..','masks','temporal_lobe_mask_3T_subj{}bold.nii.gz'.format(subj))
    flavor = 'moco_to_subjbold3Tp2'
    nmbrs = []
    with open(datapath+'/'+'stimuli_labels.txt') as fh:
        for i,ln in enumerate(fh.readlines()):
                nmbrs.append(ln)
        nmbrs = np.array([int(n.strip('\n')) for n in nmbrs])
        run_info = np.reshape(cat_nrs[nmbrs-1],(8,25))

    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_events = dhandle.get_bold_run_model(model,subj,run_id)
	#add unique genre description
        for i in range(len(run_events)):
            run_events[i]['condition'] = run_info[run_id-1][i]

        #warp into group space
        run_ds = dhandle.get_bold_run_dataset(subj, task, run_id, chunks=run_id-1, mask=mask_fname, flavor=flavor)
        filename = 'brain_subj_{}_run_{}.nii.gz'.format(subj, run_id)
        tmp_path = scratch_path + filename
        save(unmask(run_ds.samples.astype('float32'), mask_fname), tmp_path)
        warp = fsl.ApplyWarp()
        warp.inputs.in_file = tmp_path
        warp.inputs.out_file = scratch_path+'group_'+filename
        warp.inputs.ref_file = os.path.join(datapath,'templates','grpbold','brain.nii.gz')
        warp.inputs.field_file = os.path.join(datapath, 'sub{:03}'.format(subj), 'templates', 'bold', 'in_grpbold', 'subj2tmpl_warp.nii.gz')
        warp.inputs.interp = 'nn'
        warp.run()
        os.remove(tmp_path)
        run_ds_new = mvpa.fmri_dataset(scratch_path+'group_'+filename, mask=group_mask, chunks=run_id-1)
        mvpa.poly_detrend(run_ds_new, polyord=1)
        mvpa.zscore(run_ds_new)
        os.remove(scratch_path+'group_'+filename)

        yield (mvpa.events2sample_attr(run_events, run_ds.sa.time_coords, noinfolabel='rest'), run_ds_new.samples.astype('float32'))

def process_subj_3T(subj, scratch_path='/data/mboos/tmp/',
                 preprocessed_path='/data/mboos/pandora/fmri_3T/',
                 datapath='/data/pandora_3T/data', **kwargs):
    '''this function preprocesses subj run-wise and then saves it as a joblib pickle under preprocessed_path'''

    task = 1
    model = 1

    # preprocess participant and concatenate runs
    data_list = [dat for dat in preprocess_and_tmp_save_fmri_3T(datapath, task, subj, model, scratch_path, **kwargs)]
    labels, fmri_data = zip(*data_list)
    labels = np.concatenate(labels)
    fmri_data = np.vstack(fmri_data)

    joblib.dump(fmri_data, preprocessed_path+'fmri_subj_{}.pkl'.format(subj))
    joblib.dump(labels, preprocessed_path+'labels_subj_{}.pkl'.format(subj))

def preprocess_and_tmp_save_fmri_7T(datapath, task, subj, model, scratch_path, group_mask='group_temporal_lobe_mask.nii.gz'):
    '''preprocesses one subject from Forrest Gump
    aligns to group template
    run-wise linear de-trending and z-scoring'''
    from nipype.interfaces import fsl
    dhandle = mvpa.OpenFMRIDataset(datapath)

    flavor = 'dico_bold7Tp1_to_subjbold7Tp1'
    group_brain_mask = 'brainmask_group_template.nii.gz'
    mask_fname = os.path.join(datapath, 'sub{0:03d}'.format(subj), 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
    run_info = [read_csv(os.path.join('/home','mboos','run'+str(i)+'.csv')) for i in xrange(8)]
    run_info = [map(lambda x : x.split('_')[0]+x.split('_')[1][2],ri['stim']) for ri in run_info]


    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_ds = dhandle.get_bold_run_dataset(subj,task,run_id,chunks=run_id-1,mask=mask_fname,flavor=flavor)
        filename = 'brain_subj_{}_run_{}.nii.gz'.format(subj, run_id)
        tmp_path = scratch_path + filename
        save(unmask(run_ds.samples.astype('float32'), mask_fname), tmp_path)
        # warp into group space
        warp = fsl.ApplyWarp()
        warp.inputs.in_file = tmp_path
        warp.inputs.out_file = scratch_path+'group_'+filename
        warp.inputs.ref_file = os.path.join(datapath,'templates','grpbold7Tp1','brain.nii.gz')
        warp.inputs.field_file = os.path.join(datapath, 'sub{:03}'.format(subj), 'templates', 'bold7Tp1', 'in_grpbold7Tp1', 'subj2tmpl_warp.nii.gz')
        warp.inputs.interp = 'nn'
        warp.run()
        os.remove(tmp_path)
        run_ds_new = mvpa.fmri_dataset(scratch_path+'group_'+filename, mask=group_mask, chunks=run_id-1)
        mvpa.poly_detrend(run_ds_new, polyord=1)
        mvpa.zscore(run_ds_new)
        os.remove(scratch_path+'group_'+filename)

        run_info_subj = np.array(run_info)[seqs[subj-1]]
        run_events = dhandle.get_bold_run_model(model,subj,run_id)
        #add unique genre description
        for i in range(len(run_events)):
                run_events[i]['condition'] = run_info_subj[run_id-1][i]
        yield (mvpa.events2sample_attr(run_events,run_ds.sa.time_coords,noinfolabel='rest'), run_ds_new.samples.astype('float32'))

def process_subj_7T(subj, scratch_path='/data/mboos/tmp/',
                 preprocessed_path='/data/mboos/pandora/fmri/',
                 datapath='../data/forrest_gump/phase1', **kwargs):
    '''this function preprocesses subj run-wise and then saves it as a joblib pickle under preprocessed_path'''
    # Forrest Gump, auditory version
    task = 2
    model = 1

    # preprocess participant and concatenate runs
    data_list = [dat for dat in preprocess_and_tmp_save_fmri(datapath, task, subj, model, scratch_path, **kwargs)]
    labels, fmri_data = zip(*data_list)
    labels = np.concatenate(labels)
    fmri_data = np.vstack(fmri_data)

    joblib.dump(fmri_data, preprocessed_path+'fmri_subj_{}.pkl'.format(subj))
    joblib.dump(labels, preprocessed_path+'labels_subj_{}.pkl'.format(subj))


if __name__=='__main__':
    if len(sys.argv) == 1:
        dataset_keys = ['7T', '3T']
    else:
        dataset_keys = sys.argv[1:]
    preprocess_funcs = {'7T': process_subj_7T,
                        '3T': process_subj_3T}

    subjects = {'7T': range(1,20),
                '3T': range(1,19)}
    #save the preprocessed data here:
    preprocessed_path = {'7T': '/data/mboos/pandora/fmri',
                         '3T': '/data/mboos/pandora/fmri_3T'}
    for dataset in dataset_keys:
        for subj in subjects[dataset]:
            preprocess_funcs[dataset](subj, preprocessed_path=preprocessed_path[dataset])
