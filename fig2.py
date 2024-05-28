import sys;sys.path.append('src/')
import pylab
import spiketools
import defaultSimulate as default
from copy import deepcopy
from bisect import bisect_right
from matplotlib.ticker import MaxNLocator
import pandas as pd
import pickle
from joblib import Parallel, delayed
import numpy as np
# Local modules (not installed packages)
import ClusterModelNEST
from GeneralHelper import ( Organiser,
    colors, simpleaxis, 
    ax_label1, nice_figure, draw_box
)


datapath = 'preprocessed_and_simulated_data/'
datafile = 'fig2_simulated_data'

def get_spikes_fig2(params):
    params_copy = deepcopy(params)
    EI_Network = ClusterModelNEST.ClusteredNetwork(default, params_copy)
    EI_Network_copy = deepcopy(EI_Network)
    # Creates object which creates the EI clustered network in NEST
    result = EI_Network_copy.get_simulation() 
    return result

def simulate_spontaneous(params):
    pylab.seed()
    trials = params['trials']
    trial_length = params['trial_length']
    sim_params = deepcopy(params)
    sim_params['simtime'] = trials*trial_length
    ff_window = params['ff_window']
    #EI_Network = ClusterModelNEST.ClusteredNetwork(default, params)
    #EI_Network_copy = deepcopy(EI_Network)
    # Creates object which creates the EI clustered network in NEST
    results = get_spikes_fig2(sim_params)
    #results = EI_Network_copy.get_simulation()
    long_spiketimes = results['spiketimes']
    order = pylab.argsort(long_spiketimes[0])
    long_spiketimes = long_spiketimes[:,order]
    # cut into trial pieces
    spiketimes = pylab.zeros((3,0))
    
    for trial in range(trials):
        trial_end = bisect_right(long_spiketimes[0], trial_length)
        trial_spikes = long_spiketimes[:,:trial_end].copy()
        long_spiketimes = long_spiketimes[:,trial_end:]
        trial_spikes = pylab.concatenate(
            [trial_spikes[[0],:],pylab.ones((1,trial_spikes.shape[1]))*trial,
             trial_spikes[[1],:]],axis=0)
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
    return pylab.nanmean(ffs),pylab.nanmean(cv2s),pylab.nanmean(counts)


def plot_ff_cv_vs_jep(params,jep_range=pylab.linspace(1,4,41),jipfactor = 0.,reps = 10,
                      spike_js = [1.,2.,3.,4.],spike_simtime = 1000.,markersize = 0.5,
                      spikealpha = 0.5,plot_units = [0,4000],spike_randseed = 0,
                      plot = True,redo_spiketrains = False):
    ffs = []
    cv2s = []
    counts = []
    for jep in jep_range:
        print('###############################################################')
        print(jep,'-----------------------------------------------------------')
        print('###############################################################')
        if jipfactor != 0. or jep<10:
            jip = 1. +(jep-1)*jipfactor
            params['jplus'] = pylab.around(pylab.array([[jep,jip],
                                                        [jip,jip]]),5)
            ORG = Organiser(params, datafile, reps=reps, n_jobs = 4)
            ORG_copy = deepcopy(ORG)
            results = ORG_copy.check_and_execute(simulate_spontaneous)

            ff = [r[0] for r in results]
            cv2 = [r[1] for r in results]
            count = [r[2] for r in results]
            ffs.append(ff)
            cv2s.append(cv2)
            counts.append(count)
        else:
            ffs.append(ff)
            cv2s.append(cv2)
            counts.append(count)
    ffs = pylab.array(ffs)
    cv2s = pylab.array(cv2s)
    cv2s = pylab.nanmean(cv2s,axis=1)
    
    if plot:
            
        ffs = pylab.nanmean(ffs,axis=1)

        n_boxes = len(spike_js)
        box_bottom = ffs.max()*1.1
        box_top = ffs.max()*1.7
        xlim = [jep_range.min(),jep_range.max()]
        box_sep = 0.05*(xlim[1]-xlim[0])
        box_lim = [xlim[0]+box_sep,xlim[1]-box_sep]
        box_span = box_lim[1]-box_lim[0]
        box_width = (box_span-box_sep*(n_boxes-1))/float(n_boxes)
        box_height = box_top-box_bottom
        for i,j in enumerate(spike_js):
            if j == 'max':
                j = jep_range[pylab.argmax(ffs)]
            box_target_ind = pylab.argmin(pylab.absolute(j-jep_range))
            box_target = [jep_range[box_target_ind],ffs[box_target_ind]]
            box_left = box_lim[0]+i*(box_width+box_sep)
            draw_box([box_left,box_bottom,box_width,box_height], 
                              box_target,[0.6,0.6,0.6,0.6])

            jep = round(j,4)
            
            jip = round(1 + (jep-1)*jipfactor,4)
            print('spikes for ',jep,jip)
            spike_params = deepcopy(params)
            spike_params['jplus'] = pylab.array([[jep,jip],[jip,jip]])
            spike_params['randseed'] = spike_randseed
            spike_params['simtime'] = spike_simtime
            ORG = Organiser(spike_params, datafile +'_spikes')
            results = ORG.check_and_execute(get_spikes_fig2)
            spiketimes = results['spiketimes']
            spiketimes = spiketimes[:,spiketimes[1]<plot_units[1]]
            spiketimes = spiketimes[:,spiketimes[1]>=plot_units[0]]
            spiketimes[1] -= plot_units[0]
            spiketimes[0]/= spiketimes[0].max()
            spiketimes[1]/= spiketimes[1].max()

            spiketimes[0] *= box_width
            spiketimes[0] += box_left
            spiketimes[1] *= box_height
            spiketimes[1] += box_bottom

            downsampling_factor = 1  # Adjust as needed

            # Downsample the spike data
            spike_x = spiketimes[0][::downsampling_factor]
            spike_y = spiketimes[1][::downsampling_factor]
            # Plot the downsampled spike data
            #pylab.plot(spike_x, spike_y, '.k', 
            #           markersize=markersize, alpha=spikealpha)

            #pylab.scatter(spiketimes[0],spiketimes[1],s=5, marker='.', color='k',
              #            rasterized=True)
            pylab.plot(spiketimes[0],spiketimes[1],'.k',
                      markersize = markersize,alpha = spikealpha,
                      rasterized=True)

            
        
        pylab.plot(jep_range,ffs,'k')
        if jipfactor == 0.:
            pylab.gca().set_ylim(-0.2, 3)
        else:
            pylab.gca().set_ylim(-0.2, 12)
        pylab.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        pylab.xlim(jep_range.min(),jep_range.max())
def plot_ff_jep_vs_Q(params,jep_range=pylab.linspace(1,4,41),
                     Q_range = pylab.arange(2,20,2),jipfactor = 1,reps = 40,
                     plot = True,vrange = [0,15],redo = False):
    
    if jipfactor == 0.:
        model = 'E_clustered'
    else:
        model = 'EI_clustered'
    try:
        ffs = pd.read_pickle(datapath + "fig2_ffs_"+model)
    except:
        ffs = pylab.zeros((len(jep_range),len(Q_range),reps))
        for i,jep_ in enumerate(jep_range):
            for j,Q in enumerate(Q_range):
                jep = float(min(jep_,Q))
                if jipfactor == 0.:
                    params['portion_I'] = Q
                else:
                    params['portion_I'] = 1
                jip = 1. +(jep-1)*jipfactor
                print('#######################################################')
                print(Q,jep,jip,'---------------------------------------------')
                print('#######################################################')
                params['jplus'] = pylab.around(
                    pylab.array([[jep,jip],[jip,jip]]),5)
                params['Q'] = int(Q)
                ORG = Organiser(deepcopy(params), datafile,
                                reps=reps,ignore_keys=['n_jobs'],
                                redo = redo)
                ORG_copy = deepcopy(ORG)
                results = ORG_copy.check_and_execute(simulate_spontaneous)
                del ORG_copy
                ff = [r[0] for r in results]
                ffs[i,j,:] = ff
                if jep_>Q:
                    ffs[i,j,:] = pylab.nan
                
        pickle.dump(ffs,open(datapath + "fig2_ffs_"+model,'wb'))

    if plot:
        pylab.contourf(jep_range,Q_range,pylab.nanmean(ffs,axis=2).T,
                       levels = [0.5, 1.,1.5,2.],extend = 'both', 
                       cmap = 'Greys')
        x = pylab.linspace(Q_range.min(), jep_range.max(),1000)
        y1 = pylab.ones_like(x)*Q_range.min()
        y2 = x
        pylab.fill_between(x,y1, y2,facecolor = 'w',hatch = '\\\\\\',
                           edgecolor = colors['orange'])
        pylab.xlabel(r'$\mathrm{J_{E+}}$')
        pylab.ylabel(r'Q')
        pylab.axis('tight')
    return ffs
    


def plot_ff_jep_vs_Q_parallel(params, jep_range=pylab.linspace(1, 4, 41),
                     Q_range=pylab.arange(2, 20, 2), jipfactor=1, reps=40,
                     plot=True, vrange=[0, 15], redo=False):

    if jipfactor == 0.:
        model = 'E_clustered'
    else:
        model = 'EI_clustered'

    try:
        ffs = pd.read_pickle(datapath + "fig2_ffs_" + model)
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
            params['jplus'] = np.around(
                np.array([[jep, jip], [jip, jip]]), 5)
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
        pickle.dump(ffs, open(datapath + "fig2_ffs_" + model, 'wb'))
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
    settings = [{'warmup':200,'ff_window':400,'trials':20,
                 'trial_length':400.,'n_jobs':n_jobs,'Q':50,'jipfactor':0.,
                 'jep_range':pylab.arange(1,50.001,0.1),
                 'spike_js':[1.,3.,5., 8. ,10.], 'portion_I':50}, 
                {'jipfactor':0.,'fixed_indegree':False, 
                 'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,
                 'n_jobs':n_jobs,'I_th_E':2.14,'I_th_I':1.26},
                {'warmup':200,'ff_window':400,'trials':20,
                 'trial_length':400.,'n_jobs':n_jobs,'Q':50,'jipfactor':0.75,
                 'jep_range':pylab.arange(1.001,50.001, 0.1),
                 'spike_js':[1.,8.,10.5,14.,50.], 'portion_I':1},
                {'jipfactor':0.75,'fixed_indegree':False, 
                 'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,
                 'n_jobs':n_jobs,'I_th_E':2.14,'I_th_I':1.26}]  #3,5  hz


    plot = True
    reps = 20
    x_label_val = -0.25
    num_row, num_col = 2,3
    if plot:
        
        rc_params = {'axes.labelsize': 10,
                    'lines.linewidth':2,
                    'xtick.labelsize': 8,
                    'ytick.labelsize': 8}
        fig = nice_figure(ratio = .9, rcparams = rc_params)        
        fig.subplots_adjust(bottom = 0.15,hspace = 0.4,wspace = 0.3)
        abc_size = 10
        labels = ['a','b','c','d']
    title_left = ['E clustered network','','E/I clustered network']
    for i,params in enumerate(settings):
        row = int(i/2)
        col= int(i%2)
        jipfactor = params['jipfactor']
        if plot and i in [0,2]:
            ax = simpleaxis(pylab.subplot2grid((num_row,num_col),
                                               (row, col), colspan=2))
            ax_label1(ax, labels[i],x=x_label_val, size=abc_size)
            pylab.ylabel('FF')
            pylab.xlabel('$\mathrm{J_{E+}}$')
            jep_range = params.pop('jep_range')
            spike_js = params.pop('spike_js')
            plot_ff_cv_vs_jep(params,reps = reps,jipfactor =jipfactor,
                              jep_range = jep_range,spike_js = spike_js,
                              plot = plot,spike_randseed = 3,
                              spike_simtime = 2000.,markersize = 0.1,
                              spikealpha= 0.3)
            pylab.gca().text(-7, i/3.+0.5, title_left[i], 
                             rotation=90,size=abc_size)
        else:
            jep_step = 0.5
            jep_range = pylab.arange(1.,15.+0.5*jep_step,jep_step)
            q_step = 1
            Q_range = pylab.arange(q_step,60+0.5*q_step,q_step)
            
            if plot:
                ax = simpleaxis(pylab.subplot2grid((num_row,num_col),(row, col+1)))          
                ax_label1(ax, labels[i], x=x_label_val, size=abc_size)
            ffs = plot_ff_jep_vs_Q_parallel(params,jep_range,Q_range,
                                   jipfactor=jipfactor,plot=plot,reps=40)
            if plot:
                cbar = pylab.colorbar()
                cbar.set_label('FF', rotation=90)
    pylab.savefig('fig2.pdf', dpi=600)
    #pylab.savefig('fig2.eps')
    #pylab.savefig('fig2.png', dpi=300)    
    #pylab.show()



