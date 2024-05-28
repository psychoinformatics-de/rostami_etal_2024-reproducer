import sys; sys.path.append('src')
epsilon = sys.float_info.epsilon
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pandas as pd
import scipy.stats as ss

# Local imports (not installed packages)
from GeneralHelper import (
    extract_info_from_keys,
    nice_figure, simpleaxis, ax_label1, CB_color_cycle)
from analyse_model import get_analysed_spiketimes

data_path = 'preprocessed_and_simulated_data/'
datafile = 'fig4_simulated_data'
file_name_analysis_sw = datafile + '_analyses'
trial = 1
fig_size = (4,3)
fontsize = 18

# load data
try:
    Anls = pd.read_pickle(data_path + file_name_analysis_sw)
except:
    params = {'N_E': 4000, 'N_I': 1000, 'I_th_E': 2.14,
              'I_th_I': 1.3, 'ff_window': 1000, 'min_vals_cv2': 1,
              'stim_length': 1000, 'isi':0., 'isi_vari': 200,
              'cut_window': [0, 1002], 'rate_kernel': 50., 'warmup': 200}
    n_jobs = 4
    save = True
    plot = False
    redo = False
    trials = 100# 20

    params['n_jobs'] = n_jobs
    setting = {'randseed': 0, 'Q': 50,
               'stim_amp': 0.0000001, 'mazzucato_jminus': False,
               'mazzucato_js': False}  
    params['fixed_indegree'] = False
    params['trials'] = trials
    for Q in [50]:
        for stim_amp in pylab.arange(epsilon, 0.71, 0.01):
            for cnt, portion_I in enumerate([1,50]):  
                for k in setting.keys():
                    params[k] = setting[k]
                params['Q'] = Q
                params['stim_clusters'] = [0,1,2,3,4] 
                params["portion_I"] = portion_I
                params["stim_amp"] = stim_amp
                if portion_I == Q:
                    params['jipfactor'] = 0.
                    params['jep'] = 3.45
                if portion_I == 1:
                    params['jipfactor'] = 0.75
                    params['jep'] = 11.
                params['s'] = 1. # s
                result = get_analysed_spiketimes(
                    params, datafile,window=params['ff_window'],
                    calc_cv2s=True, save=save)
                if result is None:
                    print('No results for ', params)
                    break
    Anls = pd.read_pickle(data_path + file_name_analysis_sw)

data_lists = {
    'amp': [], 'ff': [], 'CV': [], 'rate': [], 'ff_std': [], 'CV_std': [], 'rate_std': [],
    'portion': [], 'ff1': [], 'CV1': [], 'rate1': [], 'ff_std1': [], 'CV_std1': [], 'rate_std1': [],
}

def add_mean_std_to_data_lists(data_lists, key, data, stim, non_stim):
    data_lists[key].append(np.nanmean(data[stim], axis=0)[0])
    data_lists[key + '_std'].append(ss.sem(data[stim], axis=0, nan_policy='omit')[0])
    data_lists[key + '1'].append(np.nanmean(data[non_stim], axis=0)[0])
    data_lists[key + '_std1'].append(ss.sem(data[non_stim], axis=0, nan_policy='omit')[0])

Anls = {key: value for key, value in Anls.items() if key is not None}
for keys, sel_data in Anls.items():
    params = extract_info_from_keys({}, keys)
    portion_I = params['portion_I']
    stim = params['stim_clusters']  # ids of stim clusters
    non_stim = [c for c in range(params['Q']) if c not in stim]
    data_lists['amp'].append(params['stim_amp'])
    add_mean_std_to_data_lists(data_lists, 'ff', sel_data['ffs'], stim, non_stim)
    add_mean_std_to_data_lists(data_lists, 'CV', sel_data['cv2s'], stim, non_stim)
    # include only units with rate > median rate of non-stimulated clusters in the stim. clusters
    mask = sel_data['rates'][stim] > np.median(sel_data['rates'][stim])
    data_lists['rate'].append(np.nanmean(sel_data['rates'][stim][mask]))
    data_lists['rate_std'].append(ss.sem(sel_data['rates'][stim][mask], nan_policy='omit'))
    data_lists['rate1'].append(np.nanmean(sel_data['rates'][non_stim], axis=0)[0])
    data_lists['rate_std1'].append(ss.sem(sel_data['rates'][non_stim], axis=0, nan_policy='omit')[0])
    data_lists['portion'].append(portion_I)

# Extract unique values
amp_uni, portion_uni = np.unique(data_lists['amp']), np.unique(data_lists['portion'])

# Create matrices based on unique values
def populate_matrix(key, data, unique_values):
    mat = np.zeros((len(unique_values['amp']), len(unique_values['portion'])))
    for i, amp_val in enumerate(data['amp']):
        amp_idx = np.where(unique_values['amp'] == amp_val)[0]
        por_idx = np.where(unique_values['portion'] == data['portion'][i])[0]
        mat[amp_idx, por_idx] = data_lists[key][i]
    return mat


# Populate matrices with data
ff_mat = populate_matrix('ff', data_lists, {'amp': amp_uni, 'portion': portion_uni})
CV_mat = populate_matrix('CV', data_lists, {'amp': amp_uni, 'portion': portion_uni})
rate_mat = populate_matrix('rate', data_lists, {'amp': amp_uni, 'portion': portion_uni})
ff_mat_std = populate_matrix('ff_std', data_lists, {'amp': amp_uni, 'portion': portion_uni})
CV_mat_std = populate_matrix('CV_std', data_lists, {'amp': amp_uni, 'portion': portion_uni})
rate_mat_std = populate_matrix('rate_std', data_lists, {'amp': amp_uni, 'portion': portion_uni})

# Create non-stim matrices
ff_mat_nonstim = populate_matrix('ff1', data_lists, {'amp': amp_uni, 'portion': portion_uni})
CV_mat_nonstim = populate_matrix('CV1', data_lists, {'amp': amp_uni, 'portion': portion_uni})
rate_mat_nonstim = populate_matrix('rate1', data_lists, {'amp': amp_uni, 'portion': portion_uni})
ff_mat_nonstim_std = populate_matrix('ff_std1', data_lists, {'amp': amp_uni, 'portion': portion_uni})
CV_mat_nonstim_std = populate_matrix('CV_std1', data_lists, {'amp': amp_uni, 'portion': portion_uni})
rate_mat_nonstim_std = populate_matrix('rate_std1', data_lists, {'amp': amp_uni, 'portion': portion_uni})

# Plot
fig = nice_figure(fig_width=0.489, ratio = 0.9)
abc_fontsize = 10*0.7
fig.subplots_adjust(bottom = 0.1,left = 0.1,right=0.94,top=0.8, wspace = 0.3,hspace=0.6)

nrow,ncol = 2,2
ax1 = simpleaxis(plt.subplot2grid((nrow,ncol), (0,0), colspan=1))
x_label_val=-0.15
ax_label1(ax1, 'a',x=x_label_val, size=abc_fontsize)
plt.axis('off')

ax1 = simpleaxis(plt.subplot2grid((nrow,ncol), (1,0), rowspan=1))
FF_ylim = (-2,8)
ax_label1(ax1, 'c',x=x_label_val, size=abc_fontsize)

lw = 1.
ls = '-'
FF = ff_mat-ff_mat[0]
FF_std = ff_mat_std-ff_mat_std[0]
FF_nonstim = ff_mat_nonstim-ff_mat_nonstim[0]
FF_nonstim_std = ff_mat_nonstim_std-ff_mat_nonstim_std[0]
fc ='0.9'
plt.plot(amp_uni, FF[:,-1], lw =lw, label = 'E stim. clusters', c = CB_color_cycle[7])
plt.fill_between(amp_uni, FF[:,-1]-FF_std[:,-1], FF[:,-1]+FF_std[:,-1], 
                 facecolor='lightcoral',lw=0, alpha=0.5)
plt.plot(amp_uni, FF_nonstim[:,-1], lw =lw, ls=ls, label = 'E non-stim. clusters', c = CB_color_cycle[1])
plt.fill_between(amp_uni, FF_nonstim[:,-1]-FF_nonstim_std[:,-1], 
                 FF_nonstim[:,-1]+FF_nonstim_std[:,-1], facecolor='navajowhite', alpha=0.5,lw=0)


plt.gca().set_title('E clustered network')
plt.gca().set_ylabel(r'$\Delta$FF')
plt.xlabel('stim. Amplitude [pA]')
plt.axvline(0.4, ls ='--', lw=0.8, color='gray')
plt.axhline(0., ls ='--', lw=0.8, color='gray')
plt.ylim(-2,20)
plt.yticks([-2,0,5,10,15,20])

ax1 = simpleaxis(plt.subplot2grid((nrow,ncol), (1,1), rowspan=1))
ax_label1(ax1, 'd',x=x_label_val, size=abc_fontsize)
plt.gca().set_title('E/I clustered network')
plt.plot(amp_uni, FF[:,0], lw = lw, label = 'E/I clustered network', c = CB_color_cycle[0])
plt.fill_between(amp_uni, FF[:,0]-FF_std[:,0], FF[:,0]+FF_std[:,0],
                 facecolor='lightskyblue', alpha=0.5,lw=0)
plt.plot(amp_uni, FF_nonstim[:,0], lw = lw, ls=ls,label = 'E/I non-stim. clusters', c = CB_color_cycle[5])
plt.fill_between(amp_uni, FF_nonstim[:,0]-FF_nonstim_std[:,0], FF_nonstim[:,0]+FF_nonstim_std[:,0],
                 facecolor='plum', alpha=0.5,lw=0)


plt.gca().set_ylabel(r'$\Delta$FF')
plt.xlabel('stim. Amplitude [pA]')
plt.axvline(0.4, ls ='--', lw=0.8, color='gray')
plt.axhline(0., ls ='--', lw=0.8, color='gray')
plt.ylim(-2,1)
ax1 = simpleaxis(plt.subplot2grid((nrow,ncol), (0,1), rowspan=1))
ax_label1(ax1, 'b',x=x_label_val, size=abc_fontsize)

RATE = rate_mat-rate_mat[0]
RATE_nonstim = rate_mat_nonstim-rate_mat_nonstim[0]
RATE_std = rate_mat_std-rate_mat_std[0]
RATE_nonstim_std = rate_mat_nonstim_std-rate_mat_nonstim_std[0]
plt.plot(amp_uni, RATE[:,0], lw = lw, label = 'E/I stim. clusters', c = CB_color_cycle[0])
plt.plot(amp_uni, RATE_nonstim[:,0], lw = lw, ls=ls, label = 'E/I non-stim. clusters', c = CB_color_cycle[5])
plt.plot(amp_uni, RATE[:,-1], lw =lw, label = 'E stim. clusters', c = CB_color_cycle[7])
plt.plot(amp_uni, RATE_nonstim[:,-1], lw =lw, ls=ls, label = 'E non-stim. clusters', c = CB_color_cycle[1])
plt.axhline(0,ls='--',lw=0.8,color='gray')
plt.gca().set_ylabel(r'$\Delta$rate [1/s]')
plt.xlabel('stim. Amplitude [pA]')
plt.axvline(0.4, ls='--', lw=0.8, color='gray')
plt.legend(loc=9,ncol=2,bbox_to_anchor=(-0.1, 1.7))

plt.savefig('../data/fig_StimAmp0.eps')
plt.savefig('../data/fig_StimAmp0.jpg')

# combine figures
import pyx
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0.0, "../data/fig_StimAmp0.eps"))
c.insert(pyx.epsfile.epsfile(
    .5, 3.8,"preprocessed_and_simulated_data/sketch_ff.eps",scale=0.7))
c.writePDFfile("fig4.png")  
plt.show()
# remove intermediate files
import os
os.remove('../data/fig_StimAmp0.eps')
os.remove('../data/fig_StimAmp0.jpg')
