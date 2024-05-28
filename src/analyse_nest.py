import pylab
import spiketools
from bisect import bisect_right,bisect_left
import defaultSimulate as default

def compute_cluster_rates(spiketimes,N_E,Q,kernel_sigma=50,kernel_type = 'tri',tlim = None):
    """ compute the avarage firing rate of the Q excitatory clusters in spiketimes. """
    cluster_rates = []
    cluster_size =N_E/Q
    
    order = pylab.argsort(spiketimes[1])
    ordered_spiketimes = spiketimes[:,order]
    ordered_spiketimes = ordered_spiketimes[:,ordered_spiketimes[1]<N_E]
    if kernel_type == 'tri':
        kernel = spiketools.triangular_kernel(sigma = kernel_sigma)
    elif kernel_type == 'gauss':
        kernel = spiketools.gaussian_kernel(sigma = kernel_sigma)
    for i in range(Q):
       
        cluster_maxind =(i+1)*cluster_size
        
        cluster_spiketimes = ordered_spiketimes[:,ordered_spiketimes[1,:]<cluster_maxind]
        
        cluster_spiketimes[1] -= i*cluster_size
        ordered_spiketimes = ordered_spiketimes[:,ordered_spiketimes[1,:]>=cluster_maxind]
        
        rate,t_rate = spiketools.kernel_rate(cluster_spiketimes,kernel,tlim = tlim)
        
        cluster_rates.append(rate[0])
    return pylab.array(cluster_rates),t_rate


def cut_trials(spiketimes,events,tlim):
    """ cut spiketimes into trials of tlim around events. trials cannot overlap.
        resulting spiketimes: [spiketimes,trial,unit]
        """


    order = pylab.argsort(spiketimes[0])
    spiketimes = spiketimes[:,order]
    # cut into trial pieces
    new_spiketimes = pylab.zeros((3,0))
    trial_start = 0
    
    
    for trial,event in enumerate(events):
        
        
        trial_start = bisect_left(spiketimes[0], event+tlim[0])
        trial_end = bisect_right(spiketimes[0], event+tlim[1])
        trial_spikes = spiketimes[:,trial_start:trial_end].copy()
        #spiketimes = spiketimes[:,trial_end:]
        trial_spikes = pylab.concatenate([trial_spikes[[0],:]-event,pylab.ones((1,trial_spikes.shape[1]))*trial,trial_spikes[[1],:]],axis=0)
        new_spiketimes = pylab.append(new_spiketimes, trial_spikes,axis=1)
    return new_spiketimes



def split_unit_spiketimes(spiketimes,N = None):
    """ splits a spiketimes array containig [spiketimes,trial,unit] into a dictionary
        where keys are unit numbers and entries contain corresponding spiketimes.
        N is the expected total number of units
        """
    trials = pylab.unique(spiketimes[1])
    spike_dict = {}
    order = pylab.argsort(spiketimes[2])
    spiketimes = spiketimes[:,order]
    if N is None:
        units = pylab.unique(spiketimes[2])
    else:
        units = pylab.arange(N)

    for unit in units:
        unit_end = bisect_right(spiketimes[2], unit)
        if unit_end>0:
            spike_dict[unit] = spiketimes[:2,:unit_end]
        else:
            spike_dict[unit] = pylab.zeros((2,0))
        missing_trials = list(set(trials).difference(spike_dict[unit][1,:])) 
        for mt in missing_trials:
            spike_dict[unit] = pylab.append(spike_dict[unit], 
                                            pylab.array([[pylab.nan],[mt]]),
                                            axis=1)
        spiketimes = spiketimes[:,unit_end:]

    return spike_dict

def get_cluster_inds(params):
    Q = params.get('Q',default.Q)
    N_E = params.get('N_E',default.N_E)
    cluster_size = int(N_E/Q)
    cluster_inds = [list(range(i*cluster_size,(i+1)*cluster_size)) for i in range(Q)]
    return cluster_inds