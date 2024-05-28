import sys;sys.path.append('src/')
import pylab
import spiketools
import defaultSimulate as default
from copy import deepcopy
from bisect import bisect_right
import ClusterModelNEST
import defaultSimulate as default
import pickle
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from GeneralHelper import ( Organiser,
    simpleaxis, 
    nice_figure
)

datapath = '../data/preprocessed_and_simulated_data/'
datafile = 'suppl_fig1_simulated_data'

def simulate_spontaneous(params):
    pylab.seed()
    trials = params['trials']
    trial_length = params['trial_length']
    sim_params = deepcopy(params)
    sim_params['simtime'] = trials*trial_length
    ff_window = params['ff_window']
    EI_Network = ClusterModelNEST.ClusteredNetwork(default, sim_params)
    # Creates object which creates the EI clustered network in NEST
    result = EI_Network.get_simulation() 
    long_spiketimes = result['spiketimes']
    order = pylab.argsort(long_spiketimes[0])
    long_spiketimes = long_spiketimes[:,order]
    # cut into trial pieces
    spiketimes = pylab.zeros((3,0))
    trial_start = 0
    
    for trial in range(trials):

        trial_end = bisect_right(long_spiketimes[0], trial_length)
        trial_spikes = long_spiketimes[:,:trial_end].copy()
        long_spiketimes = long_spiketimes[:,trial_end:]
        trial_spikes = pylab.concatenate([trial_spikes[[0],:],pylab.ones((1,trial_spikes.shape[1]))*trial,trial_spikes[[1],:]],axis=0)
        spiketimes = pylab.append(spiketimes, trial_spikes,axis=1)
        long_spiketimes[0]-= trial_length

    order = pylab.argsort(spiketimes[2])
    spiketimes = spiketimes[:,order]
    N_E = params.get('N_E',default.N_E)
    ffs = []
    cv2s = []
    counts = []
    for unit in range(N_E):
        unit_end = bisect_right(spiketimes[2], unit)
        
        unit_spikes = spiketimes[:2,:unit_end]
        spiketimes = spiketimes[:,unit_end:]
        counts.append(unit_spikes.shape[1])
        if unit_spikes.shape[1]>0:
            window_ffs = []
            tlim = pylab.array([0,ff_window])
            while tlim[0]<trial_length:
                window_ffs.append(spiketools.ff(unit_spikes,tlim = tlim))
                tlim+=ff_window
            ffs.append(pylab.nanmean(window_ffs))
            cv2s.append(spiketools.cv2(unit_spikes,pool = False))
        else:
            ffs.append(pylab.nan)
            cv2s.append(pylab.nan)
    print('ff',pylab.nanmean(ffs))
    print('cv2',pylab.nanmean(cv2s))
    return pylab.nanmean(ffs),pylab.nanmean(cv2s),pylab.nanmean(counts)

def get_spikes_fig2(params):
    EI_Network = ClusterModelNEST.ClusteredNetwork(default, params)
    # Creates object which creates the EI clustered network in NEST
    result = EI_Network.get_simulation() 
    return result

   

def plot_ff_jep_vs_Q_LitwinKumaretal_parallel(
    params, jep_range=pylab.linspace(1, 4, 41),
    Q_range=pylab.arange(2, 20, 2), jipfactor=1, reps=40,
    plot=True, vrange=[0, 15], redo=False):

    try:
        ffs = pd.read_pickle(
            datapath + "suppl_fig1_simulated_data_analysis")
    except FileNotFoundError:
        ffs = np.zeros((len(jep_range), len(Q_range), reps))
        def process_params(i, Q_idx, ffs):
            jep_ = jep_range[i]
            Q = Q_range[Q_idx]
            print('jep_', jep_, 'Q_idx', Q_idx, 'Q', Q)
            jep = float(min(jep_, Q))
            if jipfactor == 0.:
                params['portion_I'] = Q
            else:
                params['portion_I'] = 1
            jip = 1. + (jep - 1) * jipfactor
            print('##########################################################')
            print(Q, jep, jip, '---------------------------------------------')
            print('##########################################################')
            params['jplus'] = pylab.around(
                pylab.array([[jep,1.0],[jip,1.0]]),5)
            params['Q'] = int(Q)
            ORG = Organiser(params, datafile, reps=reps,
                            ignore_keys=['n_jobs'], n_jobs=1,
                            redo=False, save=True)
            results = ORG.check_and_execute(simulate_spontaneous)
            ff = [r[0] for r in results]
            ffs[i, Q_idx, :] = ff
            if jep_ > Q:
                ff = [np.nan] * reps
                ffs[i, Q_idx, :] = np.nan
            return i, Q_idx, ff
        import itertools
        # Parallelize the nested loop using joblib
        results_all = Parallel(n_jobs=4)(
            delayed(process_params)(i, Q_idx,ffs)
            for i, Q_idx in list(itertools.product(
                range(len(jep_range)), range(len(Q_range))))
        )
        for i, Q_idx, ff in results_all:
            ffs[i, Q_idx, :] = ff
        pickle.dump(
            ffs,open(
                datapath + "suppl_fig1_simulated_data_analysis",'wb'))

    if plot:
        pylab.contourf(jep_range, Q_range, np.nanmean(ffs, axis=2).T,
                       levels=[0.5, 1., 1.5, 2.], extend='both',
                       cmap='Greys')
        x = np.linspace(Q_range.min(), jep_range.max(), 1000)
        y1 = np.ones_like(x) * Q_range.min()
        y2 = x
        pylab.fill_between(x, y1, y2, facecolor='w', hatch='\\\\\\',
                           edgecolor='orange')
        pylab.xlabel(r'$\mathrm{J_{E+}}$')
        pylab.ylabel(r'Q')
        pylab.axis('tight')

    return ffs


if __name__ == '__main__':
    
    n_jobs = 12
    settings = [{'jipfactor':0.75,'fixed_indegree':False, 
                 'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,
                    'n_jobs':n_jobs,'I_th_E':2.14,'I_th_I':1.26}]  #3,5  hz
    
    plot = True
    reps = 20
    x_label_val = -0.25
    num_row, num_col = 1,1
    if plot:
        rc_params = {'axes.labelsize': 10,
                'lines.linewidth':2,
                'xtick.labelsize': 6,
                'ytick.labelsize': 6}
        fig = nice_figure(fig_width=0.489,ratio = .9, rcparams = rc_params)

        abc_fontsize = 10*0.7
        fig.subplots_adjust(bottom = 0.15,hspace = 0.4,wspace = 0.3)

    for i,params in enumerate(settings):
        row = 0
        col= 0
        if True:
            jipfactor = params.pop('jipfactor')
            jep_step = 0.5
            jep_range = pylab.arange(1.,15.+0.5*jep_step,jep_step)
            q_step = 1
            Q_range = pylab.arange(q_step,60+0.5*q_step,q_step)
            
            if plot:
                ax = simpleaxis(
                    pylab.subplot2grid((num_row,num_col),
                                       (row, col)),labelsize = 10) 
            plot_ff_jep_vs_Q_LitwinKumaretal_parallel(
                params,jep_range,Q_range,jipfactor,
                plot=plot, redo=False)
            if plot:
                cbar = pylab.colorbar()
                cbar.set_label('FF', rotation=90,size = 14)
    pylab.savefig('suppl_fig1.pdf')  
    #pylab.show()



