import joe_and_lili
import analyse_experiment as analyses
import pylab
from copy import deepcopy
import analyse_model
import numpy as np
from GeneralHelper import Organiser



def reaction_time_plot(monkey,nbins = 40,
                       condition_colors = ['0','0.3','0.6']):
    rts = []
    minrt = 1000000
    maxrt = 0
    for condition in [1,2,3]:
        params = {'monkey':monkey,'condition':condition}
        ORG = Organiser(params, 
                        'experiment_'+monkey.decode("utf-8")+'_reaction_times',)
        rt = ORG.check_and_execute(_get_mo_times) 
        rts.append(rt)
        minrt = min(minrt,min(rts[-1]))
        maxrt = max(maxrt,max(rts[-1]))


    bins = pylab.linspace(minrt,maxrt,nbins)
    for condition in [1,2,3]:
        plotbins = (bins[1:]+bins[:-1])/2.
        pylab.hist(pylab.array(rts[condition-1]),bins,histtype = 'step',
                        label = 'condition '+str(condition),
                        color = condition_colors[condition-1],
                        lw = 1.,density = True)
    
    pylab.xlim(1400,2000)


def _calc_reaction_time_analysis(original_params):
    
    params = deepcopy(original_params)
    
    tau  =params.pop('tau')
    threshold_per_condition = params.pop('threshold_per_condition')
    threshold_resolution = params.pop('threshold_resolution')
    score_reps = params.pop('score_reps')
    integrate_from_go = params.pop('integrate_from_go')
    normalise_across_clusters = params.pop('normalise_across_clusters')
    try:
        redo = params.pop('redo')
    except:
        redo  =False
    
    all_predictions = []
    all_rts = []
    all_conditions = []
    all_directions = []
    condition_thresholds = {}
    condition_threshold_scores = {}
    all_integrals = None
    for condition in [1,2,3]:
        
        params['condition'] = condition
        
        opt_params = deepcopy(params)
        if not threshold_per_condition:
            opt_params.pop('condition')
        thresh_range,scores = optimize_threshold(
            opt_params,tau  =tau, thresh_range=pylab.arange(0,1.0,threshold_resolution),
            reps =score_reps,redo = redo,integrate_from_go = integrate_from_go,
            normalise_across_clusters=normalise_across_clusters)
        condition_threshold_scores[condition] = (thresh_range,scores)
        params['condition'] = condition
        
        finite = pylab.isfinite(scores)
        thresh_range = thresh_range[finite]
        scores= scores[finite]
        threshold = thresh_range[pylab.argmax(scores)]

        condition_thresholds[condition] = threshold
        predictions,rts,conditions,directions,integrals,time = race_to_threshold(params,threshold = threshold,tau  =tau,
                                                                    integrate_from_go  =integrate_from_go,
                                                                    normalise_across_clusters=normalise_across_clusters)
        #print rts
        all_predictions += predictions.tolist()
        all_rts += rts.tolist()
        all_conditions += conditions.tolist()
        all_directions += directions.tolist()
        if all_integrals is None:
            all_integrals = integrals
        else:
            all_integrals = pylab.append(all_integrals, integrals,axis=1)

    result = {'condition_thresholds':condition_thresholds,
              'condition_threshold_scores':condition_threshold_scores,
              'predictions':pylab.array(all_predictions),
              'rts':pylab.array(all_rts),
              'conditions':pylab.array(all_conditions),
              'directions':pylab.array(all_directions),
              'integrals':all_integrals,'time':time}

    return result
    
def get_reaction_time_analysis(
    original_params,threshold_per_condition = False,
    tlim = [-500,2000],tau = 10.,threshold_resolution = 0.01,
    score_reps =1,redo  =False,integrate_from_go  =False,
    normalise_across_clusters=False,save=True,
    fname='reaction_time_analyses'):
    """Get the reaction time analysis for the model"""
    params = deepcopy(original_params)
    params['threshold_per_condition'] = threshold_per_condition
    params['tlim']  = tlim
    params['tau'] = tau
    params['threshold_resolution'] = threshold_resolution
    params['score_reps'] = score_reps
    params['integrate_from_go'] =integrate_from_go
    params['normalise_across_clusters'] = normalise_across_clusters
    params['redo'] = redo
    ORG = Organiser(params, fname, redo=redo, save=save)
    return ORG.check_and_execute(_calc_reaction_time_analysis)


def _get_mo_times(params):
    monkey=params['monkey']
    condition = params['condition']

    toc = joe_and_lili.get_toc(extra_filters = [['monkey','=',monkey]])
    gns = pylab.unique(toc['global_neuron'])
    rts = []
    for gn in gns:
        data = analyses.load_data(gn,condition)
        new_rts = data['eventtimes'][:,data['event_names'] == str.encode('MO')]
        # get rid of strange nan to int conversions
        new_rts = new_rts[new_rts<10000]
        new_rts = new_rts[new_rts>-10000]
        rts +=new_rts.tolist()
    
    return pylab.array(rts).flatten()



def integrate(counts,tau = 50.,dt = 1.):
    counts = np.array(counts)
    integral = np.zeros_like(counts).astype(float)
    
    for t in range(1,counts.shape[1]):
        integral[:,t] = integral[:,t-1] + dt * (-integral[:,t-1]/float(tau) + counts[:,t-1])

    return integral


def get_count_integrals(params,tau = 100.,integrate_from_go  =False,
                    normalise_across_clusters = False, redo=False):
    
    cluster_params = deepcopy(params)
    try:
        cluster_params.pop('condition')
    except:
        pass
    cluster_counts,time,conditions,directions = analyse_model.get_mean_cluster_counts(
        cluster_params,redo=redo)
    if integrate_from_go:
        go_time = params['sim_params'].get('prep_length',1000)
        go_ind = pylab.argmin(pylab.absolute(time-go_time))
        integrals  = pylab.array([integrate(counts[:,go_ind:],tau  =tau) for counts in cluster_counts])
        intshape = integrals.shape
        timeshape = time.shape[0]
        diff = timeshape - intshape[2]
        integrals = pylab.append(pylab.zeros((
                        intshape[0],intshape[1],diff)), 
                        integrals,axis=2)


    else:
        integrals = pylab.array([integrate(counts,tau  =tau) for counts in cluster_counts])
   
    try:
        condition = params['condition']
        integrals = integrals[:,conditions==condition]
        directions = directions[conditions == condition]
        conditions = conditions[conditions == condition]

        
    except:
        pass

    # normalise
    if normalise_across_clusters:
        integrals[integrals==0] = 1e-10
        integrals /= integrals.sum(axis=0)[None,:,:]
    else:
        
        cluster_max = integrals.max(axis=2).max(axis=1)
        integrals /= cluster_max[:,None,None]

    return integrals,time,conditions,directions



def race_to_threshold(params,tau = 100.,threshold = 0.5,
                integral_output = None,integrate_from_go = False,
                normalise_across_clusters = True,redo=False):
    print('Calculating race to threshold ...')
    if integral_output is None:
        full_integrals,time,conditions,directions = get_count_integrals(
                    params,tau ,integrate_from_go = integrate_from_go,
                    normalise_across_clusters=normalise_across_clusters,redo=redo)
    else:
        full_integrals,time,conditions,directions = integral_output
    # Go cue
    go_time = params['sim_params'].get('prep_length',1000)
    go_ind = np.argmin(np.absolute(time-go_time))
    integrals = full_integrals[:,:,go_ind:]
    
    prediction = []
    rts = []
    for trial in range(integrals.shape[1]):
        trial_ints = integrals[:,trial]
       
        max_trial_int = trial_ints.max(axis=0)
        t = np.where(max_trial_int>threshold)[0]
        if len(t)>0:
            
            direction = float(pylab.argmax(trial_ints[:,t[0]])+1)
            rt = float(t[0])
        else:
            direction  =0
            rt = 10000
               

        prediction.append(direction)
        rts.append(rt)
        


    return pylab.array(prediction),pylab.array(rts),conditions,directions,full_integrals,time


def _calc_thresh_scores(params):
    try:
        redo = params.pop('redo')
    except:
        redo  =False
    race_params = deepcopy(params)
    thresh_range = race_params.pop('thresh_range')
    tau = race_params.pop('tau')
    integrate_from_go = params.pop('integrate_from_go')
    normalise_across_clusters = params.pop('normalise_across_clusters')
    try:
        reps = race_params.pop('reps')
    except:
        reps = 1
    scores = []
    print('caclulatining threshold scores')
    
    integral_output = get_count_integrals(params,tau ,
                        integrate_from_go = integrate_from_go,
                        normalise_across_clusters=normalise_across_clusters,redo=redo)
    for threshold in thresh_range:

        predictions,rts,conditions,directions,_,_ = race_to_threshold(
                        race_params,tau = tau,threshold=threshold,
                        integral_output = integral_output,
                        normalise_across_clusters=normalise_across_clusters,redo=redo)

        finite = np.isfinite(np.array(predictions))
        predictions = predictions[finite]
        directions = directions[finite]
        if len(directions)<1:
            repscores = [pylab.nan]
        else:
            if reps>1:
                repscores = []
                for r in range(reps):
                    
                    inds = pylab.randint(0,len(directions),len(directions))
                    repscores.append(analyse_model.balanced_accuray(predictions[inds],directions[inds]))
            else:
                repscores = analyse_model.balanced_accuray(predictions,directions)
        scores.append(pylab.nanmean(repscores))

    return pylab.array(thresh_range),pylab.array(scores)

def optimize_threshold(params,tau=100,thresh_range = pylab.arange(0,1.0,0.1),
                       redo = False,reps  =10,integrate_from_go  =False,
                       normalise_across_clusters = False):
    
    calc_params = deepcopy(params)
    calc_params['thresh_range'] = thresh_range
    calc_params['tau'] = tau
    calc_params['reps'] = reps
    calc_params['integrate_from_go'] = integrate_from_go
    calc_params['normalise_across_clusters'] = normalise_across_clusters
    ORG = Organiser(calc_params, 
                    'model_race_to_threshold_scores',redo  =redo)
    return ORG.check_and_execute(_calc_thresh_scores)
   



