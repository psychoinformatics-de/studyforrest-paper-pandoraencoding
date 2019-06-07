#%%
import numpy as np
from os.path import join
import scipy as sp
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def conf_mat_to_acc(conf_mat):
    return np.sum(np.diag(conf_mat))/25.

def conf_mat_sum(conf_mat):
    return np.diag(conf_mat)/8.

def conf_mat_genre_sum(conf_mat):
    from scipy import linalg
    idx_mat = linalg.block_diag(*[np.ones((5,5))]*5)
    return np.diag(conf_mat.dot(idx_mat))/8.

def melt_to_df(array_to_melt,colnames):
    import pandas as pd
    assert(len(colnames)==len(array_to_melt.shape))
    shapes = [1] + list(array_to_melt.shape) + [1]
    melted_df = pd.DataFrame()
    #rvcn_list = list(reversed(list(enumerate(colnames))))
    melted_df['data'] = array_to_melt.flatten()
    for i,name in enumerate(colnames):
        melted_df[name] = np.tile(np.repeat(np.arange(shapes[i+1]),np.cumprod(shapes[i+2:])[-1]),np.cumprod(shapes[:i+1])[-1])
    return melted_df


def prepare_for_plotting(to_melt_list,colnames,Ns):
    melted_3T = melt_to_df(to_melt_list[0],colnames)
    melted_7T = melt_to_df(to_melt_list[1],colnames)
    melted_3T['Field strength'] = ['3T']*melted_3T.shape[0]
    melted_7T['Field strength'] = ['7T']*melted_7T.shape[0]
    melted_df = melted_7T.append(melted_3T)
    melted_df = melted_df.replace({'voxelnr':{ x:n for x,n in enumerate(Ns)}})
    return melted_df

def unpack_data_dict(data_dict, subjects):
    volume_dict = {'3T': 27, '7T': 2.7}
    list_of_dfs = []
    for fs, subj_list in data_dict.iteritems():
        for subj, subj_dict in zip(subjects[fs], subj_list):
            for measure, measure_dict in subj_dict.iteritems():
                for select, select_dict in measure_dict.iteritems():
                    for n_vxl, actual_data in select_dict.iteritems():
                        if measure == 'decoding':
                            actual_data_genre = conf_mat_genre_sum(actual_data)
                            actual_data = conf_mat_sum(actual_data)
                        idx_dict = {colname: [colval]*actual_data.shape[0]
                                    for colname, colval in zip(
                                        ['Field strength', 'Subject', 'Measure', 'Selection criterion', 'Number of voxels', r'Volume ($mm^3$)'],
                                        [fs, subj, measure, select, n_vxl, np.float(n_vxl)*volume_dict[fs]])}
                        idx_dict['Score'] = actual_data
                        if measure=='decoding':
                            idx_dict2 = {colname: [colval]*actual_data_genre.shape[0]
                                    for colname, colval in zip(
                                        ['Field strength', 'Subject', 'Measure', 'Selection criterion', 'Number of voxels', r'Volume ($mm^3$)'],
                                        [fs, subj, 'decoding genre', select, n_vxl, np.float(n_vxl)*volume_dict[fs]])}
                            idx_dict2['Score'] = actual_data_genre
                            list_of_dfs.append(pd.DataFrame(idx_dict2))
                        list_of_dfs.append(pd.DataFrame(idx_dict))
    return pd.concat(list_of_dfs, axis=0, ignore_index=True)


def prob_to_acc_for_array(prob_mat):
    shapes = prob_mat.shape
    prob_mat = prob_mat[:].reshape((-1,25,25))
    prob_mat_red = np.zeros((prob_mat.shape[0]))
    for i in xrange(np.cumprod(prob_mat_red.shape)[-1]):
        prob_mat_red[i] = prob_to_acc(prob_mat[i])
    return prob_mat_red.reshape(shapes[:-2])

if __name__ == '__main__':
    data_folders = {'7T': '/data/mboos/pandora/encodingscores/',
                    '3T': '/data/mboos/pandora/3T/encodingscores/'}

    subjects = {'7T': range(11,20),
                 '3T': range(1,19)}
    field_strength = ['3T', '7T']

    data_dict = {fs: [joblib.load(join(data_folders[fs], 'validation_subj_{}.pkl'.format(i)))
                        for i in subjects[fs]]
                 for fs in field_strength}
    data_df = unpack_data_dict(data_dict, subjects)
    data_mean = data_df.groupby(
            ['Subject', 'Field strength', 'Measure', r'Volume ($mm^3$)',
             'Selection criterion', 'Number of voxels']
                                )['Score'].mean().reset_index()

    transl_dict = {'binary': 'Binary retrieval accuracy', 'decoding': 'Decoding accuracy', 'rank': 'Matching rank score', 'decoding genre': 'Decoding accuracy of genre'}
    for measure in data_mean['Measure'].unique():
        g = sns.catplot(data=data_mean[data_mean['Measure']==measure], x='Number of voxels', y='Score', hue='Field strength', col='Selection criterion', kind='point')
        g.set_axis_labels(transl_dict[measure],'')
        g.savefig('../pics/{}_score_plot.pdf'.format(measure))

    for measure in data_mean['Measure'].unique():
        g = sns.catplot(data=data_mean[data_mean['Measure']==measure], x='Number of voxels', y='Score', col='Field strength', hue='Selection criterion', kind='point')
        g.set_axis_labels(transl_dict[measure],'')
        g.savefig('../pics/{}_sel_comparison.pdf'.format(measure))

    for measure in data_mean['Measure'].unique():
        g = sns.catplot(data=data_mean[data_mean['Measure']==measure], x=r'Volume ($mm^3$)', y='Score', hue='Field strength', col='Selection criterion', kind='point')
        g.set_axis_labels(transl_dict[measure],'')
        g.savefig('../pics/{}_score_plot_volume.pdf'.format(measure))

    for measure in data_mean['Measure'].unique():
        g = sns.catplot(data=data_mean[data_mean['Measure']==measure], x=r'Volume ($mm^3$)', y='Score', col='Field strength', hue='Selection criterion', kind='point')
        g.set_axis_labels(transl_dict[measure],'')
        g.savefig('../pics/{}_sel_comparison_volume.pdf'.format(measure))

##%%
#corr_bin_melted = prepare_for_plotting([np.mean(bin_3t_corr_n,axis=-1),np.mean(bin_7t_corr_n,axis=-1)],colnames[:-1],Ns)
#corr_match_melted = prepare_for_plotting([np.mean(match_3t_corr_n,axis=-1),np.mean(match_7t_corr_n,axis=-1)],colnames[:-1],Ns)
##corr_probs_melted = prepare_for_plotting([prob_to_acc_for_array(probs_3t_corr_n),prob_to_acc_for_array(probs_7t_corr_n)],colnames[:-1],Ns)
#corr_probs_melted = prepare_for_plotting([probs_3t_corr_n,probs_7t_corr_n],colnames[:-1],Ns)
#corr_svc_melted = prepare_for_plotting([dec_3t_Corr_n,dec_7t_corr_n],colnames[:-1],Ns)
#
#stable_match_melted = prepare_for_plotting([np.mean(matches_3t_n,axis=-1),np.mean(matches_7t_n,axis=-1)],colnames[:-1],Ns)
#stable_bin_melted = prepare_for_plotting([np.mean(bin_3t_n,axis=-1),np.mean(bin_7t_n,axis=-1)],colnames[:-1],Ns)
##stable_probs_melted = prepare_for_plotting([prob_to_acc_for_array(probs_3t_n),prob_to_acc_for_array(probs_7t_n)],colnames[:-1],Ns)
#stable_probs_melted = prepare_for_plotting([probs_3t_n,probs_7t_n],colnames[:-1],Ns)
#stable_svc_melted = prepare_for_plotting([dec_3t_n,dec_7t_n],colnames[:-1],Ns)
#
#corr_bin_melted['selection'] = corr_bin_melted.shape[0]*['r^2']
#corr_match_melted['selection'] = corr_match_melted.shape[0]*['r^2']
#corr_probs_melted['selection'] = corr_probs_melted.shape[0]*['r^2']
#stable_bin_melted['selection'] = stable_bin_melted.shape[0]*['stability']
#stable_match_melted['selection'] = stable_match_melted.shape[0]*['stability']
#stable_probs_melted['selection'] = stable_probs_melted.shape[0]*['stability']
#
#stable_svc_melted['selection'] = stable_svc_melted.shape[0]*['stability']
#corr_svc_melted['selection'] = corr_svc_melted.shape[0]*['r^2']
#stable_svc_melted.replace({'Field strength':{'3T' : 'SVC 3T','7T':'SVC 7T'}},inplace=True)
#corr_svc_melted.replace({'Field strength':{'3T' : 'SVC 3T','7T':'SVC 7T'}},inplace=True)
#
#bin_melted = stable_bin_melted.append(corr_bin_melted)
#match_melted = stable_match_melted.append(corr_match_melted)
#probs_melted = stable_probs_melted.append([corr_probs_melted,stable_svc_melted,corr_svc_melted])
#
#t3mm = 27
#t7mm = 2.7 #actually 2.74... but this messes up seaborns categorical plotting
#probs_melted_mm = probs_melted2.copy()
#probs_melted_mm.loc[probs_melted_mm['Field strength']=='7T','voxelnr'] *= t7mm
#probs_melted_mm.loc[probs_melted_mm['Field strength']=='3T','voxelnr'] *= t3mm
#probs_melted_mm.loc[probs_melted_mm['Field strength']=='SVC 7T','voxelnr'] *= t7mm
#probs_melted_mm.loc[probs_melted_mm['Field strength']=='SVC 3T','voxelnr'] *= t3mm
#probs_melted_mm.voxelnr /= 1000
#
#match_melted_mm = match_melted.copy()
#match_melted_mm.loc[match_melted_mm['Field strength']=='7T','voxelnr'] *= t7mm
#match_melted_mm.loc[match_melted_mm['Field strength']=='3T','voxelnr'] *= t3mm
#match_melted_mm.loc[match_melted_mm['Field strength']=='SVC 7T','voxelnr'] *= t7mm
#match_melted_mm.loc[match_melted_mm['Field strength']=='SVC 3T','voxelnr'] *= t3mm
#match_melted_mm.voxelnr /= 1000
#
#
#bin_melted_mm = bin_melted.copy()
#bin_melted_mm.loc[bin_melted_mm['Field strength']=='7T','voxelnr'] *= t7mm
#bin_melted_mm.loc[bin_melted_mm['Field strength']=='3T','voxelnr'] *= t3mm
#bin_melted_mm.loc[bin_melted_mm['Field strength']=='SVC 7T','voxelnr'] *= t7mm
#bin_melted_mm.loc[bin_melted_mm['Field strength']=='SVC 3T','voxelnr'] *= t3mm
#bin_melted_mm.voxelnr /= 1000
#
#
#
###Matching score
#plt.figure(figsize=(14,12))
#sns.set_style(style='whitegrid')
#g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=match_melted)
#sns.despine(left=True)
#for axes in g.axes.flat:
#    axes.set_xlabel('Number of voxels')
#g.axes.flat[0].set_title('Most stable voxels')
#g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
#g.axes.flat[0].set_ylabel('Matching rank score')
#plt.savefig('Nr_of_voxels_matching_score_selection.svg')
#
#
#plt.figure(figsize=(14,12))
#sns.set_style(style='whitegrid')
#g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=match_melted_mm)
#sns.despine(left=True)
#for axes in g.axes.flat:
#    axes.set_xlabel('Overall voxel volume in ' + r'$cm^3$')
#    axes.set_xticklabels(labels=['0.675','1.35','6.75','13.5','27','67.5','135','270'])
#g.axes.flat[0].set_title('Most stable voxels')
#g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
#g.axes.flat[0].set_ylabel('Matching rank score')
#plt.savefig('Nr_of_voxels_matching_score_selection_volume.svg')
#
#
###Binary retrieval score
#plt.figure(figsize=(10,8))
#sns.set_style(style='whitegrid')
#g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=bin_melted)
#sns.despine(left=True)
#for axes in g.axes.flat:
#    axes.set_xlabel('Number of voxels')
#    
#g.axes.flat[0].set_title('Most stable voxels')
#g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
#g.axes.flat[0].set_ylabel('Binary rank score')
#plt.savefig('Nr_of_voxels_binary_score_selection.svg')
#
#
#plt.figure(figsize=(10,8))
#sns.set_style(style='whitegrid')
#g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',data=bin_melted_mm)
#sns.despine(left=True)
#for axes in g.axes.flat:
#    axes.set_xlabel('Overall voxel volume in ' + r'$cm^3$')
#    axes.set_xticklabels(labels=['0.675','1.35','6.75','13.5','27','67.5','135','270'])
#    
#g.axes.flat[0].set_title('Most stable voxels')
#g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
#g.axes.flat[0].set_ylabel('Binary rank score')
#plt.savefig('Nr_of_voxels_binary_score_selection_volume.svg')
#
#
###Decoding accuracy + SVC decoding accuracy
#plt.figure(figsize=(10,8))
#sns.set_style(style='whitegrid')
#g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',linestyles=['-','-','--','--'],markers=['o','o','x','x'],data=probs_melted2)
#sns.despine(left=True)
#for i,axes in enumerate(g.axes.flat):
#    axes.set_xlabel('Number of voxels')
#
#g.axes.flat[0].set_title('Most stable voxels')
#g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
#g.axes.flat[0].set_ylabel('Decoding accuracy of music category')
#
#plt.savefig('Nr_of_voxels_decoding_accuracy_selection_svc.svg')
#
#plt.figure(figsize=(10,8))
#sns.set_style(style='whitegrid')
#g = sns.factorplot(x='voxelnr',y='data',hue='Field strength',col='selection',linestyles=['-','-','--','--'],markers=['o','o','x','x'],data=probs_melted_mm)
#sns.despine(left=True)
#for i,axes in enumerate(g.axes.flat):
#    axes.set_xlabel('Overall voxel volume in ' + r'$cm^3$')
#    axes.set_xticklabels(labels=['0.675','1.35','6.75','13.5','27','67.5','135','270'])
#g.axes.flat[0].set_title('Most stable voxels')
#g.axes.flat[1].set_title('Voxels with highest ' + r'$r^2$')
#g.axes.flat[0].set_ylabel('Decoding accuracy of music category')
#
#plt.savefig('Nr_of_voxels_decoding_accuracy_selection_svc_volume.svg')
#
