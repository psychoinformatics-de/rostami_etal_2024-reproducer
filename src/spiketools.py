import pylab
import numpy as np
from scipy.signal import convolve2d
from scipy.stats import gamma
from bisect import bisect_right
import unittest
import os
from scipy.integrate import quad
from scipy.optimize import fmin
from numpy import vectorize
import pickle
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, 
                  reload_support=True)

from Cspiketools import time_resolved_cv_two as _time_resolved_cv_two

def spiketimes_to_binary(spiketimes,tlim = None,dt = 1.):
    """ takes a n array of spiketimes and turns it into a binary
        array of spikes.
            spiketimes:  array where - spiketimes[0,:] -> spike times in [ms]
                                     - spiketimes[1,:] -> trial indices
            tlim:        [tmin,tmax] - time limits for the binary array
                         if None (default), extreme values from spiketimes 
                         are taken                             
            dt:          time resolution of the binary array in [ms]
        
        returns:
            binary:      binary array (trials,time)
            time:        time axis for the binary array (length = binary.shape[1])
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)
        
    time = pylab.arange(tlim[0],tlim[1]+dt,dt).astype(float) 
    
    if dt<=1:
        time-=0.5*float(dt)
   
    trials = pylab.array([-1]+list(range(int(spiketimes[1,:].max()+1))))+0.5
    
    tlim_spikes = cut_spiketimes(spiketimes, tlim)
    tlim_spikes = tlim_spikes[:,pylab.isnan(tlim_spikes[0,:])==False]

    
    if tlim_spikes.shape[1]>0:
        binary = pylab.histogram2d(tlim_spikes[0,:],tlim_spikes[1,:],[time,trials])[0].T
    else:
        binary = pylab.zeros((len(trials)-1,len(time)-1))
    return binary,time[:-1]



def binary_to_spiketimes(binary,time):
    """ takes a binary array of spikes and a corresponding time axis
        and converts it to an array of spike times.
            binary:  binary array of spikes shape = (ntrials,len(time))
            time:    time axis for binary
        returns:
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
        """
    time = pylab.array(time)
    spiketimes = [[],[]]
    max_count = binary.max()
    spikes = binary.copy()
    while max_count>0:
        if max_count == 1:
            trial,t_index = spikes.nonzero()

        else:
            trial,t_index = pylab.where(spikes==max_count)
        
        spiketimes[0] +=time[t_index].tolist()
        spiketimes[1] += trial.tolist()
        #  = pylab.append(time[t_index][pylab.newaxis,:],trial[pylab.newaxis,:],axis = 0)
        spikes[spikes==max_count]-=1
        max_count-=1
    spiketimes = pylab.array(spiketimes)
    #account for trials that have no spikes by adding a nan
    all_trials = set(range(binary.shape[0]))
    found_trials = set(spiketimes[1,:])
    missing_trials = list(all_trials.difference(found_trials))
    for mt in missing_trials:
        spiketimes = pylab.append(spiketimes,pylab.array([[pylab.nan],[mt]]),axis = 1)
    return spiketimes.astype(float)
def spiketimes_to_list(spiketimes):
    """converts spiketimes from array to list format.
           spiketimes:   array where - spiketimes[0,:] -> spike times in [ms]
                                     - spiketimes[1,:] -> trial indices
       returns:
           spikelist:    list, length = ntrials. list[n] -> spiketimes from trial n   
        """
    if spiketimes.shape[1] ==0:
        return []
    trials = list(range(int(spiketimes[1,:].max()+1)))#list(set(spiketimes[1,:].tolist()))
    orderedspiketimes = spiketimes.copy()
    orderedspiketimes = orderedspiketimes[:,pylab.isnan(orderedspiketimes[0])==False]
    spike_order = pylab.argsort(orderedspiketimes[0],kind = 'mergesort')
    orderedspiketimes = orderedspiketimes[:,spike_order]
    trial_order = pylab.argsort(orderedspiketimes[1],kind = 'mergesort')
    orderedspiketimes = orderedspiketimes[:,trial_order]
    
    spikelist = [None] * len(trials)
    start = 0
    for trial in trials:
        end = bisect_right(orderedspiketimes[1], trial)
        
        #trialspikes = spiketimes[0,spiketimes[1,:]==trial]
        trialspikes = orderedspiketimes[0,start:end]
        start = end
        #trialspikes = trialspikes[pylab.isnan(trialspikes)==False]
        spikelist[trial] = trialspikes
    return spikelist
        
def ff(spiketimes,mintrials=None,tlim = None):
    """ computes the fano-factor for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
        returns:
            fano-factor
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    # only counts per trial are important so binsize = tlim
    dt = tlim[1]-tlim[0]
    binary,time = spiketimes_to_binary(spiketimes,dt = dt,tlim = tlim)
    counts = binary.sum(axis = 1)
    if (counts == 0).all():
        return pylab.nan
    if mintrials is not None:
        if len(counts)< mintrials:
            return pylab.nan
    return counts.var()/counts.mean()
                 



def cv2(spiketimes,pool=True,return_all = False,bessel_correction = False,minvals = 0):
    """ computes the squared coefficient of variation for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
            pool:        if True, ISIs for all trials are pooled
                         if False, CV^2 is calculated per trial and averaged
        returns:
            squared coefficient of variation
        """

    if spiketimes.shape[1]<3:
        # need at least three spikes to calculate anything
        if return_all:
            return pylab.array([pylab.nan])
        return pylab.nan
    if bessel_correction:
        ddof = 1
    else:
        ddof = 0
    spikelist = spiketimes_to_list(spiketimes)
    maxlen = max([len(sl) for sl in spikelist])
    if maxlen <3:
        return pylab.nan
    spikearray = pylab.zeros((len(spikelist),maxlen))*pylab.nan
    spike_counts = []
    for i,sl in enumerate(spikelist):
        spikearray[i,:len(sl)] = sl
        spike_counts.append(len(sl))
    spike_counts = pylab.array(spike_counts)
    spikearray = spikearray[spike_counts>2]
    intervals = pylab.diff(spikearray,axis=1)
    

    if pool:
        var = pylab.nanvar(intervals,ddof=ddof)
        mean_squared = pylab.nanmean(intervals)**2
        return var/mean_squared
    else:
        
        intervals[pylab.isfinite(intervals).sum(axis = 1)<minvals,:] = pylab.nan
        var = pylab.nanvar(intervals,axis=1,ddof=ddof)
        mean_squared = pylab.nanmean(intervals,axis = 1)**2
        if return_all:
            return var/mean_squared
        else:
            return pylab.nanmean(var/mean_squared)
        
def time_resolved(spiketimes,window,func,kwargs = {},tlim = None,tstep = 1.):
    """ applies a function to spiketimes in a time resolved manner.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
            window:      time window considered at each step
            func:        function to be applied at each step
                         must take spiketimes as its first argument
            kwargs:      optional arguments to func
            tlim:        [tmin,tmax] time limits of the calculation
            tstep:       increment by which the window is moved in each step
        returns:
            out:        output of func for each window
            time:       time axis of out (centered on the window)
        """     
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    cut_spikes = cut_spiketimes(spiketimes,tlim)
    
    # initial time window
    tmin = tlim[0]
    tmax = tmin+window
    tcenter = tmin + 0.5 *(tmax-tstep-tmin)
    
    time=[]
    func_out = []
    while tmax <= tlim[1]:
        windowspikes = cut_spiketimes(cut_spikes,[tmin,tmax])
        time.append(tcenter) 
        func_out.append(func(windowspikes,**kwargs)) 
        
        tmin+=tstep
        tmax+=tstep
        tcenter+=tstep
    
    return func_out,pylab.array(time)


def time_resolved_new(spiketimes,window,func,kwargs = {},tlim = None,tstep = 1.):
    """ applies a function to spiketimes in a time resolved manner.
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
            window:      time window considered at each step
            func:        function to be applied at each step
                         must take spiketimes as its first argument
            kwargs:      optional arguments to func
            tlim:        [tmin,tmax] time limits of the calculation
            tstep:       increment by which the window is moved in each step
        returns:
            out:        output of func for each window
            time:       time axis of out (centered on the window)
        """     
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    
    
    binary,btime = spiketimes_to_binary(spiketimes,tlim = tlim)
    # initial time window
    tmin = tlim[0]
    tmax = tmin+window
    tcenter = 0.5 *(tmax+tmin)
    
    time=[]
    func_out = []
    while tmax <= tlim[1]:
        #print tlim,tmin,tmax,binary[:,int(tmin):int(tmax)].shape
        windowspikes = binary_to_spiketimes(binary[:,int(tmin):int(tmax)], btime[int(tmin):int(tmax)])
        time.append(tcenter) 
        func_out.append(func(windowspikes,**kwargs)) 
        
        tmin+=tstep
        tmax+=tstep
        tcenter+=tstep
    return func_out,pylab.array(time)

def _consecutive_intervals_old(spiketimes):
    trial_spikes = spiketimes_to_list(spiketimes)
    trial_isis = [pylab.diff(spikes) for spikes in trial_spikes]
    consecutive_isis = pylab.zeros((0,2))
    for isis in trial_isis:

        inds = pylab.array([list(range(i,i+2)) for i in range(len(isis)-1)])
        if len(inds)==0:
            continue
        consecutive_isis = pylab.append(consecutive_isis, isis[inds],axis = 0)
    return consecutive_isis

def _consecutive_intervals(spiketimes):
    
    order = pylab.argsort(spiketimes[0])
    sorted_spiketimes = spiketimes[:,order]
    order = pylab.argsort(sorted_spiketimes[1],kind = 'mergesort')
    sorted_spiketimes = sorted_spiketimes[:,order]
    isis = pylab.diff(sorted_spiketimes[0])
    trial = pylab.diff(sorted_spiketimes[1])
    isis[trial!=0] = pylab.nan
    inds = pylab.array([list(range(i,i+2)) for i in range(len(isis)-1)]) 
    try:
        consecutive_isis = isis[inds]
    except:
        consecutive_isis = pylab.zeros((1,2))*pylab.nan
    consecutive_isis = consecutive_isis[pylab.isfinite(consecutive_isis.sum(axis=1))]
    
    return consecutive_isis

def cv_two(spiketimes,min_vals=20):
    """
    ms = []

    spikelist = spiketimes_to_list(spiketimes)
    for trialspikes in spikelist:
        
        trial_isis = pylab.diff(pylab.sort(trialspikes))
        for i in range(len(trial_isis)-1):
            ms.append(2.*pylab.absolute(trial_isis[i] -trial_isis[i+1])/(trial_isis[i]+trial_isis[i+1]))
    """
    consecutive_isis = _consecutive_intervals(spiketimes)
    ms = 2*pylab.absolute(consecutive_isis[:,0]-consecutive_isis[:,1])/(consecutive_isis[:,0]+consecutive_isis[:,1])
    if len(ms)>=min_vals:
        return pylab.mean(ms)
    else:
        return pylab.nan
    
def lv(spiketimes,min_vals=20):
    """
    ms = []
    spikelist = spiketimes_to_list(spiketimes)
    for trialspikes in spikelist:
        trial_isis = pylab.diff(pylab.sort(trialspikes))
        for i in range(len(trial_isis)-1):
            ms.append(3.*(trial_isis[i] -trial_isis[i+1])**2/(trial_isis[i]+trial_isis[i+1])**2)
    """
    consecutive_isis = _consecutive_intervals(spiketimes)
    ms = 3*(consecutive_isis[:,0]-consecutive_isis[:,1])**2 / (consecutive_isis[:,0]+consecutive_isis[:,1])**2
    if len(ms)>=min_vals:
        return pylab.mean(ms)
    else:
        return pylab.nan
    
def gaussian_kernel(sigma,dt = 1.,nstd=3.):
    """ returns a gaussian kernel with standard deviation sigma.
        sigma:  standard deviation of the kernel
        dt:     time resolution
        nstd:   overall width of the kernel specified in multiples
                of the standard deviation.
        """
    t = pylab.arange(-nstd*sigma,nstd*sigma+dt,dt)
    gauss = pylab.exp(-t**2 / sigma**2)
    gauss/= gauss.sum()*dt
    return gauss        

def triangular_kernel(sigma,dt=1):
    half_base =int(pylab.around(sigma * pylab.sqrt(6)))
    half_kernel = pylab.linspace(0.,1.,half_base+1)
    kernel = pylab.append(half_kernel,half_kernel[:-1][::-1])
    kernel /=dt*kernel.sum()
    
    return kernel

    
    
            
def kernel_rate(spiketimes,kernel,tlim = None,dt=1.,pool = True):
    """ computes a kernel rate-estimate for spiketimes.
            spiketimes:  array where - spiketimes[0,:] -> spike times [ms]
                                     - spiketimes[1,:] -> trial indices
            kernel:      1D centered kernel
            tlim:        [tmin,tmax] time-limits for the calculation
                         if None (default), extreme values from spiketimes
                         are taken
            dt:          time resolution of output [ms]
            pool:        if True, mean rate over all trials is returned
                         if False, trial wise rates are returned
        returns:
            rates:       the rate estimate s^-1 (if not pooled (ntrials,len(time)))
            time:        time axis for rate
        """
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    
    binary,time = spiketimes_to_binary(spiketimes, tlim, dt)
    
    if pool:
        binary = binary.mean(axis = 0)[pylab.newaxis,:]
    
    
    
    rates = convolve2d(binary,kernel[pylab.newaxis,:],'same')
    kwidth = len(kernel)
    rates = rates[:,int(kwidth/2):-int(kwidth/2)]
    time = time[int(kwidth/2):-int(kwidth/2)]
    return rates*1000.,time
   
def sliding_counts(spiketimes,window,dt = 1.,tlim = None):
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    binary,time = spiketimes_to_binary(spiketimes, dt=dt,tlim = tlim) 

    kernel = pylab.ones((1,int(window*dt)))
    counts = convolve2d(binary,kernel,'valid')
    
    dif = time.shape[0]-counts.shape[1]
    time =  time[int(dif//2):-int(dif//2)]
    return counts,time[:-1]

def kernel_fano(spiketimes,window,dt = 1.,tlim = None,
        components = False):
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    if tlim[1]-tlim[0]==window:
        binary,time = spiketimes_to_binary(spiketimes,tlim = tlim)
        counts = binary.sum(axis=1)
        return pylab.array([counts.var()/counts.mean()]),pylab.array(time.mean())
    counts,time = sliding_counts(spiketimes,window,dt,tlim)

    
    vars = counts.var(axis = 0)
    means = counts.mean(axis = 0)
    fano = pylab.zeros_like(vars)*pylab.nan
    if components:
        return vars,means,time
    else:
        fano[means>0] = vars[means>0]/means[means>0]
           
        return fano,time
    
def time_resolved_cv2(spiketimes,window = None,ot = 5.,tlim = None,pool = False,tstep = 1.,bessel_correction=False,minvals = 0,return_all = False):
    
    if tlim is None:
        tlim = _get_tlim(spiketimes)
    
    
    
    
    
    if window is None:
        spikelist = spiketimes_to_list(spiketimes)
        isis = [] 
        for spikes in spikelist:
            isis.append(pylab.diff(spikes))
        meanisi = []
        for isi in isis:
            if pylab.isnan(isi.mean())==False:
                meanisi.append(isi.mean())
        if len(meanisi)>0:
            meanisi = sum(meanisi)/float(len(meanisi))
            window = meanisi*ot
        else:
            window = tlim[1]-tlim[0]
    
    time = []
    cvs = []
    tmin = tlim[0]
    tmax = tlim[0]+window
    order = pylab.argsort(spiketimes[0])
    ordered_spiketimes = spiketimes[:,order]
    while tmax<tlim[1]:
        start_ind = bisect_right(ordered_spiketimes[0],tmin)
        end_ind = bisect_right(ordered_spiketimes[0],tmax)
        window_spikes = ordered_spiketimes[:,start_ind:end_ind]

        #window_spikes = cut_spiketimes(spiketimes,[tmin,tmax])

        cvs.append(cv2(window_spikes, pool,bessel_correction=bessel_correction,minvals = minvals,return_all=return_all))
        time.append(0.5*(tmin+tmax))
        tmin+=tstep
        tmax+=tstep
    if return_all:
        if len(cvs)>0:
            # make sure the same number of values is there for each time step
            maxlen = max([len(cv) for cv in cvs])
            cvs = [cv.tolist() +[pylab.nan]*(maxlen-len(cv)) for cv in cvs]
            #print maxlen,[len(cv) for cv in cvs]
        else:
            cvs = [[]]
            
    return pylab.array(cvs),pylab.array(time)
    
   
def time_warped_cv2(spiketimes,window = None,ot=5.,tstep=1.,tlim = None,rate = None,kernel_width = 50.,nstd=3.,pool = True,dt = 1.,inspection_plots = False,kernel_type = 'gaussian',bessel_correction = False,interpolate = False,minvals = 0,return_all = False):
    
    if tlim is None:
        tlim = _get_tlim(spiketimes)   
      
    # estimate rate if not given  
    if rate is None:
        kernel = gaussian_kernel(kernel_width, dt)
        rate,trate = kernel_rate(spiketimes, kernel,tlim = tlim, dt=dt)
    else:
        
        trate = rate[1]
        rate = rate[0]
    
    # this is to avoid completely flat bits in the transformed time axis
    rate[rate==0]=1e-4
    
    rate = rate.flatten()
    notnaninds = pylab.isnan(rate)==False
    
    rate = rate[notnaninds]
    trate = trate[notnaninds]
    
    # integrate rate to obtain transformed time axis
    tdash = rate_integral(rate, dt)
    # rescale tdash to match original time
    
    tdash -= tdash.min()
    tdash/= tdash.max()
    tdash *= trate.max()-trate.min()
    tdash+=trate.min()
    
    
    transformed_spikes = spiketimes.copy().astype(float)
    
    transformed_spikes[0,:] = time_warp(transformed_spikes[0,:], trate,tdash)
    
    if inspection_plots:
        
        pylab.figure()
        pylab.subplot(1,2,1)
        stepsize =max(1, int(spiketimes.shape[1]/100))
        for i in list(range(0,spiketimes.shape[1],stepsize))+[-1]:
            if pylab.isnan(transformed_spikes[0,i]) or pylab.isnan(spiketimes[0,i]):
                continue
            pylab.plot([spiketimes[0,i]]*2,[0,tdash[pylab.find(((trate-spiketimes[0,i])**2)==((trate-spiketimes[0,i])**2).min())]],'--k',linewidth = 0.5)
            pylab.plot([0,trate[pylab.find(((tdash-transformed_spikes[0,i])**2)==((tdash-transformed_spikes[0,i])**2).min())]],[transformed_spikes[0,i]]*2,'--k',linewidth = 0.5)
        pylab.plot(trate,tdash,linewidth = 2.)
        pylab.xlabel('real time')
        pylab.ylabel('transformed time')
        pylab.title('transformation of spiketimes')
        
    
    
    cv2,tcv2 = time_resolved_cv2(transformed_spikes, window, ot, None, pool, tstep,bessel_correction=bessel_correction,minvals = minvals,return_all = return_all)
    if inspection_plots:
        pylab.subplot(1,2,2)
        stepsize =max(1, int(len(tcv2) /50))
        tcv22 = time_warp(tcv2,tdash,trate)
        for i in list(range(0,len(tcv2),stepsize))+[-1]:
            pylab.plot([tcv2[i]]*2,[0,trate[pylab.find(((tdash-tcv2[i])**2)==((tdash-tcv2[i])**2).min())]],'--k',linewidth = 0.5)
            pylab.plot([0,tdash[pylab.find(((trate-tcv22[i])**2)==((trate-tcv22[i])**2).min())]],[tcv22[i]]*2,'--k',linewidth = 0.5)
        
        pylab.plot(tdash,trate,linewidth = 2)
        pylab. title('re-transformation of cv2')
        pylab.xlabel('transformed time')
        pylab.ylabel('real time')
        
    tcv2 = time_warp(tcv2,tdash,trate)

    if interpolate:
        
        time = pylab.arange(tlim[0],tlim[1],dt)
        if len(cv2)==0 or (return_all and cv2.shape[1]==0):
            cv2 = pylab.zeros_like(time)*pylab.nan
        else:

            if return_all:
                cv2 = pylab.array([pylab.interp(time,tcv2,cv2[:,i],left = pylab.nan,right = pylab.nan) for i in range(cv2.shape[1])])

            else:
                cv2 = pylab.interp(time, tcv2, cv2,left = pylab.nan,right = pylab.nan)
        
            
        return cv2,time
    return cv2,tcv2
    
def rate_integral(rate,dt):
    """ integrates a rate in s^-1 to obtain average number of spikes. """
    return pylab.cumsum(rate/1000.)*dt
      

def time_warp(events,told,tnew):
    """transforms the events from the time axis in tnew to the one in told."""
    # get rid of events lying outside the defined range
    return pylab.interp(events,told,tnew,left = pylab.nan,right=pylab.nan)
    
def _get_tlim(spiketimes):
    tlim = [pylab.nanmin(spiketimes[0,:]),pylab.nanmax(spiketimes[0,:])+1]
    if pylab.isnan(tlim).any():
        tlim = [0,1]
    return tlim
    
def cut_spiketimes(spiketimes,tlim):

    alltrials = list(set(spiketimes[1,:]))
    cut_spikes = spiketimes[:,pylab.isfinite(spiketimes[0])]

    cut_spikes = cut_spikes[:,cut_spikes[0,:]>=tlim[0]]
    if cut_spikes.shape[1]>0:
        cut_spikes = cut_spikes[:,cut_spikes[0,:]<tlim[1]]
    for trial in alltrials:
        if not trial in cut_spikes[1,:]:
            cut_spikes = pylab.append(
                cut_spikes,pylab.array([[pylab.nan],
                                        [trial]]),axis = 1)
    return cut_spikes

def gamma_spikes(rates,order=[1],tlim = [0.,1000.],dt=0.1):
    """ produces a spike trains drawn from a homogeneous gamma process.
            rates:       list of spike-rates in s^-1. 
                         length determines number of trials
            order:      orders of the processes. can be length 1 or len(rates)
            tlim:        time window for the spikes
            dt:          time step in ms
        returns:
            spiketimes:  array where - spiketimes[0,:] -> spike times
                                     - spiketimes[1,:] -> trial indices
        """
        
    time = pylab.arange(tlim[0],tlim[1]+dt,dt)
    if len(rates)==1:
        rates = rates[0]*order[0]
    else:
        if len(order)==1:
            rates = [r*order[0] for r in rates]
        else:
            rates = [r*o for r,o in zip(rates,order)]
        rates = pylab.tile(pylab.array(rates)[:,pylab.newaxis],(1,len(time)))
    
    spikes = 1.* pylab.rand(rates.shape[0],rates.shape[1])< rates/1000.*dt
    
    if len(order)==1:
        order*=spikes.shape[0]
    for i,o in enumerate(order):
        if o ==1:
            continue
        # find trial spikes
        inds = pylab.find(spikes[i,:])
        selection = list(range(0,len(inds),int(o)))
        spikes[i,:]=0. 
        spikes[i,inds[selection]]=1 
               
    return binary_to_spiketimes(spikes, time)


def time_stretch(spiketimes,stretchstart,stretchend,endtime = None):
    if endtime is None:
        endtime = stretchend.mean()
    
    trials = pylab.unique(spiketimes[1,:])
    
    for i,trial in enumerate(trials):
        trialmask = spiketimes[1,:]==trial
        trialspikes = spiketimes[0,trialmask]
        trialspikes -= stretchstart[i]
        se = stretchend[i]-stretchstart[i]
        trialspikes/=se
        trialspikes*= endtime-stretchstart[i]
        trialspikes+= stretchstart[i]
        spiketimes[0,trialmask] = trialspikes
    
    return spiketimes
        

def rate_warped_analysis(spiketimes,window,step=1.,tlim=None,rate = None,func =lambda x:x.shape[1]/x[1].max(),kwargs = {},rate_kernel=gaussian_kernel(50.),dt=1.):
    if rate is None:   
        rate = kernel_rate(spiketimes, rate_kernel, tlim) 
        
    rate,trate = rate
    
    
    
    
    
    rate_tlim = [trate.min(),trate.max()+1]   
    
    spiketimes = cut_spiketimes(spiketimes, rate_tlim)
    
    ot = pylab.cumsum(rate)/1000.*dt
   
    w_spiketimes = spiketimes.copy()
    w_spiketimes[0,:] = time_warp(spiketimes[0,:], trate, ot)
    
    if window =='full':
        return func(w_spiketimes,**kwargs),ot.max()
    else:
        result,tresult = time_resolved(w_spiketimes, window, func, kwargs,  tstep=step)    
        
        tresult = time_warp(tresult,ot,trate)
        
        return result,tresult           


def synchrony(spikes,ignore_zero_rows= True):
    """ takes an array of spikes (or another characteristic quantity) and 
        calculates the synchrony measure according to golomb and hansel 2000
        accross the population.
        If the number of dimensions is more than 2, trials are assumed to be
        along the first axis and the average over trials is returned.
        """
    if len(spikes.shape)>2:
        return pylab.array([synchrony(s,ignore_zero_rows) for s in spikes]).mean()
    if ignore_zero_rows:
        sync= spikes[spikes.sum(axis=1)>0].mean(axis = 0).var() / spikes[spikes.sum(axis=1)>0].var(axis = 1).mean()
    else:
        sync= spikes.mean(axis = 0).var() / spikes.var(axis = 1).mean()
    return sync**0.5


def resample(vals,time,new_time):
    """ interpolate the result from time resolved analysis (vals,time) to 
        the new time axis new_time. """
    if len(vals)>0:
        return pylab.interp(new_time,time,vals,right = pylab.nan,left = pylab.nan)
    else:
        return pylab.ones(new_time.shape)*pylab.nan
    
def time_resolved_cv_two( spiketimes,window=400, tlim = None, min_vals = 10,tstep = 1):
    """ compute the time resolved local coefficient of variation. """
    return _time_resolved_cv_two(spiketimes,window,tlim,min_vals,tstep)


def generate_gamma(k,rates,warmup_isis = 10):
    """generates gamma distributed spike trains"""
    if warmup_isis is None:
        return _generate_gamma(k, rates)
    else:
        # add a warmup time
        initial_rates = rates[:,0]
        try:
            min_rate = rates[rates>0].min()
            max_isi = 1/min_rate*1000.
            warmup_samples = int(warmup_isis*max_isi)
        except:
            #rates are zero: no point in warming up
            warmup_samples = 1
        # warmup samples should not be more than say 100000
        warmup_samples = int(min(warmup_samples,1e5))
        warmup_rates = np.array([initial_rates]*warmup_samples).T
        full_rates = np.append(warmup_rates, rates,axis=1)
        full_spikes =  _generate_gamma(k,full_rates)
        return full_spikes[:,-rates.shape[1]+1:]


def _generate_gamma(k,rates):
    """generates gamma distributed spike trains"""
    if np.isscalar(k):
        k = np.array([k])
    if len(k)!=1:
        assert len(k)==rates.shape[0],'k must be a scalar or a vector of length N_trials'
    # inflate k to vector
    if len(k)==1:
        k = np.ones((rates.shape[0],))*k
    # if rates and k are constant across trials, just generate in one go
    if rates.var(axis = 0).sum()==0 and k.var()==0:
        l = rates[0,:]/1000.
        n = rates.shape[0]
        k = k[0]
        return randg_equilibrium(k,l,n)
    else:
        spikes = np.zeros_like(rates)[:,:-1]
        for i in range(rates.shape[0]):
            l = rates[i,:]/1000.
            spikes[i,:] = randg_equilibrium(k[i],l,1)
        
        return spikes

def randg(k,l,n):
    # make sure k is float
    k=float(k)
    # integrate rate
    L = np.cumsum(l)
    N = L[-1]
    
    W = 5. * np.ceil(k)
    M = 5. * np.sqrt(N)
    
    R = gamma.rvs(k,scale = 1/k,size=(np.ceil(N+M),n))
    
    R = np.cumsum(R,axis = 0)
    spikes = np.zeros((n,len(L)-1))
    for i in range(n):
        spikes[i,:] = np.histogram(R[:,i],bins = L)[0]
    return spikes
    

def randg_equilibrium(k,l,n):
    """ the first interval is drawn from UY,
        where U is uniforml distributed in [0,1]
        and  if from gamma_(k+1,l)
        """
    # make sure k is float
    k=float(k)
    # integrate rate
    L = np.cumsum(l)
    N = L[-1]
    
    W = 5. * np.ceil(k)
    M = 5. * np.sqrt(N)
    
    U = np.random.rand(n)
    Y = gamma.rvs(k+1,scale = 1/k,size=(1,n))
    UY = U*Y

    R = gamma.rvs(k,scale = 1/k,size=(int(np.ceil(N+M)),int(n)))
    R = np.append(UY,R[:-1], axis=0)
    R = np.cumsum(R,axis = 0)
    spikes = np.zeros((n,len(L)-1))
    for i in range(n):
        spikes[i,:] = np.histogram(R[:,i],bins = L)[0]
    return spikes
 


class TestRandG(unittest.TestCase):
    def test_count_consistency(self):
        n = 5000
        
        for r in [1,10,100,200]:
            time = np.arange(1000)
            rate = np.ones_like(time)*r
            k=1.
            l = rate/1000.
            spikes = randg(k,l,n) 
            counts = spikes.sum(axis = 1)
            # count should be about right (test for 5% difference)
            self.assertAlmostEqual( counts.mean(), r,delta = r/20.)
    
    def test_ff_consistency(self):
        n = 5000
        
        for ff in [0.0001,0.1,0.5,1.,1.5]:
            time = np.linspace(0,1000,5000)
            rate = np.ones_like(time)*20
            k=1/ff
            l = rate/5000.
            spikes = randg(k,l,n) 
            
            ff_estimate = ff(binary_to_spiketimes(spikes, time))
            # ff should be approcimately equal to 1/k
            self.assertAlmostEqual( ff_estimate, ff,delta = 0.1)
            
    def test_cv_ff_consistency(self):
        n = 3000
        
        for ff in [0.0001,0.1,0.5,1.,1.5]:
            time = np.linspace(0,1000,5000)
            rate = np.ones_like(time)*10
            k=1/ff
            l = rate/5000.
            spikes = randg(k,l,n) 
            ff_estimate = ff(binary_to_spiketimes(spikes, time))
            cv2_estimate = cv2(binary_to_spiketimes(spikes, time))
            # ff should be approcimately equal to cv^2
            self.assertAlmostEqual( ff_estimate, cv2_estimate,delta = 0.1)
    
    def test_rate_modulation(self):
        n = 5000
        
        time = np.arange(1000)
        rate = np.exp(-(time-500)**2/5000.)*10
        k=1.
        l = rate/1000.
        spikes = randg(k,l,n) 
        spiketimes = binary_to_spiketimes(spikes, time)
        e_rate,e_time = kernel_rate(spiketimes, gaussian_kernel(11.),tlim=[0,1000])
        e_rate = e_rate[0,:]
        len_diff = len(rate)-len(e_rate)
        rate = rate[len_diff/2:-len_diff/2]
        # mean squared error between input and estimate should be small
        error=np.mean((e_rate-rate)**2)
        self.assertTrue(error<0.02)


class TestGenerateGamma(unittest.TestCase):
    def test_rates(self):
        tmax_s = 100. 
        rates = np.zeros((5,1000*tmax_s))
        for i in range(rates.shape[0]):
            rates[i,:] = i+1
        # regular 
        k = 1e30
        spikes = generate_gamma(k,rates)
        counts = spikes.sum(axis = 1)
        self.assertTrue( ((counts-rates.mean(axis=1)  *tmax_s)**2).max()<2) 
        
        
        




def gamma_pdf(x,rho=1,alpha = 1):
    return 1/gamma(alpha)*rho *(rho*x)**(alpha-1)*pylab.exp(-rho*x)
#@memoized
def eta(T,rho = 1,alpha = 1):
    def func(s):
        return (T-s)*gamma_pdf(s,rho,alpha)
    return quad(func, 0, T)[0]

def gamma_pdf_hat(x,T=10,rho=1,alpha = 1):
    #print eta(T,rho,alpha)
    return (T-x)*gamma_pdf(x,rho,alpha)/eta(T,rho,alpha)


def pdf_mean(pdf,lower,upper,kwargs = {}):
    def func(x):
        return x*pdf(x,**kwargs)
    return quad(func, lower, upper)[0]

def pdf_var(pdf,lower,upper,kwargs = {},mu =None):
    if mu is None:
        mu = pdf_mean(pdf, lower, upper,kwargs=kwargs)
    def func(x):
        return (x-mu)**2 * pdf(x,**kwargs)
    return quad(func, lower, upper)[0]

def gamma_cv2(ot,alpha):
    kwargs = {'alpha':alpha,'rho':alpha,'T':ot}
    m = pdf_mean(gamma_pdf_hat, 0, ot,kwargs)
    v = pdf_var(gamma_pdf_hat, 0, ot,kwargs,mu = m)
    return v/m**2

def correct_cv2(measured_cv2,ot):
    if not pylab.isfinite(measured_cv2):
        return measured_cv2
    def minfunc(alpha):
        return (measured_cv2-gamma_cv2(ot, float(alpha)))**2
    
    result = fmin(minfunc,1/measured_cv2,disp = False)
    print('result',result)
    return 1/result

def _get_cv_file(fname):
    
    try:
        look_up = pickle.load(open(fname,'rb'),encoding='latin-1')
    except:
        look_up = {}
    return look_up
@vectorize    
def unbiased_cv2(measured_cv2,ot,precission=3):
    """ same as correct_cv2 but with file memory."""
    current_path = os.path.abspath(__file__)
    full_path = os.path.join(os.path.split(current_path)[0],'..','data')
    cv_bias_lookup_fname = 'cv2_bias_lookup'
    if not pylab.isfinite(measured_cv2):
        return measured_cv2
    fname = os.path.join(full_path,cv_bias_lookup_fname)
    look_up = _get_cv_file(fname)
    key = (round(measured_cv2,precission),round(ot,precission))
    try:
        return look_up[key]
    except:
        print('adding cv bias entry for ',key)
        try:
            look_up[key] = correct_cv2(key[0], key[1])
        except:
            look_up[key] = pylab.nan
        pickle.dump(look_up,open(fname,'wb'),protocol = 2)
        _get_cv_file.cache = {}
        return look_up[key]
