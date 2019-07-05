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

def make_r2_plot():
    # use whichever folder you saved your preditions in
    scores_3t = [joblib.load('/data/mboos/pandora/3T/predictions/scores/scores_ridge_group_{}.pkl'.format(subj)) for subj in range(1,19)]
    scores_7t = [joblib.load('/data/mboos/pandora/predictions/scores/scores_ridge_{}.pkl'.format(subj)) for subj in range(1,19)]
    scores_sorted_3t = [np.sort(score)[::-1][:10000] for score in scores_3t]
    scores_sorted_7t = [np.sort(score)[::-1][:10000] for score in scores_7t]
    df = pd.DataFrame({r'$r2$': np.concatenate([np.concatenate(scores_sorted_3t), np.concatenate(scores_sorted_7t)]), 'Field strength': np.repeat(['3T', '7T'], 18*10000), 'Subject': np.tile(np.repeat(np.arange(1,19), 10000), 2), 'voxel': np.tile(np.tile(np.arange(1,10001), 18), 2)})
    sns.lineplot(x='voxel', y=r'$r2$', hue='Field strength', data=df)
    plt.savefig('r2_plot.pdf')

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

    make_r2_plot()
