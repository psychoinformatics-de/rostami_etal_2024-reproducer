from matplotlib import pylab
import joe_and_lili
import pickle 
import os
import spiketools 
from time import process_time as clock
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from GeneralHelper import (
    Organiser, find, 
    off_gray, yellow, green, red
)



def load_data(gn,condition,direction  =None,alignment = 'TS'):
    toc = joe_and_lili.get_toc(filters = [])
    file = toc['file'][(toc['global_neuron']==gn)*(toc['condition'] == condition)]
    ppath = joe_and_lili.pickle_path
    data = pickle.load(open(os.path.join(ppath,file[0].decode()),'rb'),encoding='latin-1')
    if direction is not None:
        spiketimes = data['spiketimes']
        direction_inds = find(spiketimes[2]==direction)
        direction_trials = pylab.unique(spiketimes[1,direction_inds])
        spiketimes = spiketimes[:2,direction_inds]
        for nt,ot in enumerate(pylab.unique(spiketimes[1])):
            spiketimes[1,spiketimes[1]==ot] = nt
        data['spiketimes'] = spiketimes
        data['eventtimes'] = data['eventtimes'][direction_trials.astype(int)]
    

    if alignment != 'TS':
        alignment_column = data['eventtimes'][:,data['event_names'] == alignment]
        spiketimes = data['spiketimes']
        for trial,offset in enumerate(alignment_column):
            spiketimes[0,spiketimes[1]==trial] -= offset
            data['eventtimes'][trial,:] -= offset
        data['spiketimes'] = spiketimes

    return data

def balanced_accuray(targets,predictions):
    classes = pylab.unique(targets)
    accuracies = pylab.zeros(classes.shape)
    for i in range(len(classes)):
        class_inds= find(targets == classes[i])

        accuracies[i] =(targets[class_inds]==predictions[class_inds]).mean()
    return accuracies.mean()


def _calc_cv_two(params):
    """calculate the cv_two for the given params"""
    t0 = clock()
    data = load_data(params['gn'],params['condition'],
                     params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    
    
    result = spiketools.time_resolved(spiketimes, params['window'],
                                      spiketools.cv_two,
                                      kwargs = {'min_vals':params['min_vals']},
                                      tlim = params['tlim'])
    return result
def get_cv_two(gn,condition,direction,window = 400,min_vals = 20,
               tlim = [0,2000],alignment = 'TS',redo  =False, 
               monkey = b'joe',save = False):
    """try to load the cv_two for the given gn and direction and alignment
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'min_vals':min_vals,
              'alignment':alignment,'tlim':tlim}
    ORG=Organiser(params, 
        'experiment_' + monkey.decode("utf-8") +\
            '_cvtwos_file_condition'+str(condition)+\
                '_direction'+str(direction),redo = redo,save=save)
    return  ORG.check_and_execute(_calc_cv_two)


def get_ff(gn,condition,direction,window = 400,tlim = [0,2000],
           alignment = 'TS',redo = False, monkey = b'joe',save=False):
    """try to load the ff for the given gn and direction and alignment
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'alignment':alignment,
              'tlim':tlim}
    ORG = Organiser(params, 
        'experiment_' + monkey.decode("utf-8") +\
            '_ff_file_condition'+str(condition)+\
                '_direction'+str(direction),redo = redo,save=save)
    return ORG.check_and_execute(_calc_ff)

def _calc_ff(params):
    """calculate the ff for the given params"""
    data = load_data(params['gn'],params['condition'],
                     params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    ff,tff = spiketools.kernel_fano(spiketimes, params['window'],
                                    tlim  =params['tlim'])
    return ff,tff



def get_rate(gn,condition,direction,kernel = 'triangular',
             sigma = 50.,tlim = [0,2000],alignment = 'TS',
             redo  =False, monkey = b'joe',save=False):
    """try to load the rate for the given gn and direction and 
    alignment if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'kernel':kernel,'sigma':sigma,'alignment':alignment,
              'tlim':tlim}
    ORG = Organiser(params, 
        'experiment_' + monkey.decode("utf-8") +\
            '_rate_file_condition'+str(condition)+\
                '_direction'+str(direction),redo = redo,save=save)
    return ORG.check_and_execute(_calc_rate)

def _calc_rate(params):
    """calculate the rate for the given params"""
    data = load_data(params['gn'],params['condition'],
        params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    if params['kernel'] == 'triangular':
        kernel = spiketools.triangular_kernel(params['sigma'])
    elif params['kernel'] == 'gaussian':
        kernel = spiketools.gaussian_kernel(params['sigma'])

    return spiketools.kernel_rate(spiketimes, kernel,tlim  =params['tlim'])


def _calc_trial_count(params):
    """calculate the trial count for the given params"""
    data = load_data(params['gn'],params['condition'],params['direction'])
    spiketimes = data['spiketimes']
    spiketimes = spiketimes[:,pylab.isfinite(spiketimes[0])]
    return len(pylab.unique(spiketimes[1]))

def get_trial_count(gn,condition,direction, monkey = b'joe',save=False):
    """try to load the trial count for the given gn and direction and
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction}
    ORG = Organiser(params, 
        'experiment_'+monkey.decode("utf-8") + '_trial_count_file',
        save=save)
    return ORG.check_and_execute( _calc_trial_count)
    

def _calc_mean_direction_counts(params):
    """calculate the mean direction counts for the given params"""
    data = load_data(params['gn'],params['condition'],
                     params['direction'],alignment = params['alignment'])
    spiketimes = spiketools.cut_spiketimes(data['spiketimes'],tlim  =params['tlim'])
    spiketimes = spiketimes[:,pylab.isfinite(spiketimes[0])]
    trials = len(pylab.unique(spiketimes[1]))
    return spiketimes.shape[1]/float(trials)

def get_mean_direction_counts(gn,condition,direction,
                              alignment = 'TS',tlim=[0,2000], 
                              monkey = b'joe',save=False):
    """try to load the mean direction counts for the given gn and direction and
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'alignment':alignment,'tlim':tlim}
    ORG = Organiser(params, 
        'experiment_'+monkey.decode("utf-8") + '_direction_count_file',
        save=save)
    return ORG.check_and_execute(_calc_mean_direction_counts)
    
    

def _calc_lv(params):
    """calculate the lv for the given params"""
    t0 = clock()
    data = load_data(params['gn'],params['condition'],
                     params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    result = spiketools.time_resolved(spiketimes, params['window'],
                                      spiketools.lv,
                                      kwargs = {'min_vals':params['min_vals']},
                                      tlim = params['tlim'])
    return result 
def get_lv(gn,condition,direction,window = 400,
           min_vals = 20,tlim = [0,2000],alignment = 'TS',
           redo  =False, monkey = b'joe',save=False):
    """try to load the lv for the given gn and direction and alignment
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'min_vals':min_vals,
              'alignment':alignment,'tlim':tlim}
    ORG = Organiser(params, 
        'experiment_' + monkey.decode("utf-8") +\
            '_lv_file_condition'+str(condition)+\
                '_direction'+str(direction),redo = redo,save=save)
    return ORG.check_and_execute(_calc_lv)

def _calc_direction_counts(params):
    """calculate the direction counts for the given params"""
    data = load_data(params['gn'],params['condition'],
                     alignment= params['alignment'])
    spiketimes = data['spiketimes']
    counts,time = spiketools.sliding_counts(spiketimes[:2], 
                                            params['window'],tlim = params['tlim'])
    
    trial_directions = pylab.array(
        [spiketimes[2,find(spiketimes[1]==t)[0]] for t in pylab.sort(
            pylab.unique(spiketimes[1]))])
    direction_counts = []
    mean_direction_counts = pylab.zeros((6,counts.shape[1]))
   
    for direction in range(1,7):
        direction_counts.append(counts[trial_directions == direction])

    return direction_counts,time

def _get_direction_counts(gn,condition,window = 400,tlim = [0,2000],
                          alignment = 'TS'):

    params = {'gn':gn,'condition':condition,'alignment':alignment,
              'tlim':tlim,'window':window}
    ORG = Organiser(params, 
        'experiment_direction_counts_file')
    return ORG.check_and_execute(_calc_direction_counts)


def _calc_population_decoding(params):
    """calculate the population decoding for the given params"""
    pylab.seed(params.get('randseed',None))
    all_direction_counts = []
    min_trials = pylab.ones((6),dtype = int)*100000
    for gn in params['gns']:
        
        direction_counts,time = _get_direction_counts(
            gn, params['condition'], params['window'],
            params['tlim'], params['alignment'])
        
        for i,d in enumerate(direction_counts):
            min_trials[i] = min(min_trials[i],d.shape[0])
        all_direction_counts.append(direction_counts)
    gns = params['gns']
    feature_mat = pylab.zeros((0,len(gns),len(time)))
    
    targets = []

    for direction in range(6):
        direction_features = pylab.zeros((min_trials[direction],
                                          len(gns),len(time)))
        for i,d in enumerate(all_direction_counts):
            counts = d[direction]
            
            order = pylab.arange(counts.shape[0])
            pylab.shuffle(order)
            direction_features[:,i] = counts[order[:min_trials[direction]]]
        targets += [direction+1 for trial in range(int(min_trials[direction]))]
        feature_mat = pylab.append(feature_mat,direction_features,axis = 0)


    targets = pylab.array(targets)

    score = pylab.zeros_like(time).astype(float)
    
    
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Suppress specific warning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # training and testing the classifier
    print(
        "Training and testing the classifier (this will take some time) ...")
    for i in range(len(time)):
        features = feature_mat[:,:,i]
        predictions = pylab.zeros_like(targets).astype(float)
        
        n_splits = params['folds']
        skf = StratifiedKFold(n_splits=n_splits)
        
        for train_index, test_index in skf.split(features, targets):
            cl = eval(params['classifier'])(**params['classifier_args'])
            cl.fit(features[train_index],targets[train_index])
            predictions[test_index] = cl.predict(features[test_index]) 
        score[i] = balanced_accuray(targets, predictions)

    return score,time
    



    

def get_population_decoding(gns,condition,window =400.,folds = 5,
                            tlim = [0,2000],alignment = 'TS',
                            redo = False,reps = 10, 
                            classifier = 'LogisticRegression',
                            classifier_args = {},n_jobs = 1,
                            save=False):
    """try to load the population decoding for the given gn 
    and direction and alignment"""
    params = {'gns':tuple(sorted(gns)),'condition':condition,
              'alignment':alignment,
              'tlim':tlim,'window':window,'classifier':classifier,
              'folds':folds,'classifier_args':classifier_args}
    ORG = Organiser(params, 
        'experiment_population_decoding_file',
        redo = redo,reps = reps,n_jobs = n_jobs, save=save)
    return ORG.check_and_execute(_calc_population_decoding)


def get_stats(gns,min_trials  =10,min_count_rate = 5,min_count=200,
              minvals = 0,alignment = 'TS',tlim = [0,2000],window =400,
              monkey=b'joe',save_interim=False):
    """get the statistics for the given gn and direction and alignment"""
    if monkey != None:
        extra_filters = [('monkey','=',monkey)]
        toc = joe_and_lili.get_toc(extra_filters = extra_filters)
        global_gns = pylab.unique(toc['global_neuron'])

    else:
        global_gns = pylab.unique(joe_and_lili.get_toc()['global_neuron'])
    if gns is None:
        gns = global_gns

    rates = []
    trial_counts = []
    count_rates = []
    ffs = []
    
    lvs = []
    cv_twos = []
    
    count_rate_block = pylab.zeros((len(gns),3,6))
    trial_count_block = pylab.zeros((len(gns),3,6))
    
    for i,gn in enumerate(gns):
        for j,condition in enumerate([1,2,3]):
            for k,direction in enumerate([1,2,3,4,5,6]):
                count_rate_block[i,j,k] =  get_mean_direction_counts(
                    gn,condition,direction,tlim  =tlim,
                    alignment = alignment, save=save_interim)
                trial_count_block[i,j,k]  =get_trial_count(
                    gn,condition,direction, save=save_interim)
    
    enough_counts = pylab.prod(count_rate_block>=min_count_rate,axis=1)
    enough_trials = pylab.prod(trial_count_block>=min_trials,axis=1)
    
    good_directions = enough_counts * enough_trials
    
    for i,gn in enumerate(gns):
        for j,condition in enumerate([1]):
            for k,direction in enumerate([1,2,3,4,5,6]):
                if good_directions[i,k]:
                    rate,trate = get_rate(
                        gn,condition = 1,direction = direction,
                        tlim  =tlim,alignment = alignment,
                        monkey=monkey,save=save_interim)
                    rates.append(rate[0])
                    trial_counts.append(
                        get_trial_count(gn,1,direction,monkey=monkey,save=save_interim))
                    count_rates.append(
                        get_mean_direction_counts(
                            gn,1,direction,tlim  =tlim,
                            alignment = alignment,
                            monkey=monkey, save=save_interim))
                    ff,tff = get_ff(gn,condition = 1,
                            direction = direction,window = window,
                            tlim  =tlim,alignment = alignment
                            ,monkey=monkey, save=save_interim)
                    ffs.append(ff)

                    lv,tlv = get_lv(gn,condition = 1,
                                direction = direction,tlim  =tlim,
                                alignment = alignment
                                ,monkey=monkey, save=save_interim)
                    lvs.append(lv)
                    cv_two,tcv_two = get_cv_two(
                        gn,condition = 1,direction = direction,tlim  =tlim,
                        alignment = alignment,monkey=monkey, save=save_interim)
                    cv_twos.append(cv_two)

    rates = pylab.array(rates)        
    ffs = pylab.array(ffs)
    lvs = pylab.array(lvs)
    cv_twos = pylab.array(cv_twos)
    return tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, rates


def draw_hex_array(center,size=0.3,colors = [[0.5]*3]*7,axes = None,
    radius = 0.1,add = True,show_numbers = False,
    draw_center = True,lw = 1., epoch=None):
    """draw a hexagonal array of circles with the given center and size"""
    angles = pylab.array([30,90,150,210,270,330])*pylab.pi/180
    Y = size*pylab.cos(angles)
    X = size*pylab.sin(angles)
    
    
    i = 0
    circs= []
    coords = []
    number = 6
    for x,y in zip(X,Y):
        coords.append((x+center[0],y+center[1]))
        circ = pylab.Circle((x+center[0],y+center[1]), radius=radius,  fc=colors[i],clip_on = False,lw= lw)
        if axes is None:
            axes = pylab.gca()
        if add:
            axes.add_patch(circ)
        circs.append(circ)
        #pylab.text(x,y,str(i),va='center',ha = 'center')
        if show_numbers:
            pylab.text(x+center[0],y+center[1],str(number),size = 6,ha ='center',va = 'center')
            if number == 6:
                number =1
            else:
                number+=1
        i+=1

    if draw_center:
        circ = pylab.Circle((center[0],center[1]), radius=radius,  fc=colors[-1],clip_on = False,lw = lw)
        if axes is None:
            axes = pylab.gca()
        if add:
            axes.add_patch(circ)
    if epoch!=None:
        pylab.text(center[0]-80,center[1]+250, epoch,size = 5,ma='center')

    return circs,coords

 
    
def plot_experiment(size,radius,direction =1,lw = 1.,y_pos = 0,condition = 1,write_epoch=False):
    """plot the experimental protocl"""
    if write_epoch:
        epoch = 'TS'#'Trial \n Start (TS)'
    else:
        epoch=None
    colors = [off_gray]*6 + [yellow]
    draw_hex_array([0,y_pos],colors = colors,size = size,radius = radius,lw = lw,epoch=epoch)
    colors = [off_gray]*7
    if condition ==1:
        colors[direction] = green
    elif condition == 2:
        if direction in [1,2,3]:
            colors[1] = green
            colors[0] = green
        else:
            colors[4] = green
            colors[5] = green
    elif condition == 3:
        if direction in [1,2,3]:
            colors[1] = green
            colors[2] = green
            colors[0] = green
        else:
            colors[4] = green
            colors[5] = green
            colors[3] = green
    if write_epoch:
        epoch = 'PS'#'Preparatory \n Signal (PS)'
    else:
        epoch=None
    draw_hex_array([500,y_pos],colors = colors,
                   size = size,radius = radius,lw = lw,epoch=epoch)

    colors = [off_gray]*7
    colors[direction] = red

    if write_epoch:
        epoch = 'RS'#'Response \n signal (RS)'
    else:
        epoch=None
    draw_hex_array([1500,y_pos],colors = colors,
                   size = size,radius = radius,lw = lw, epoch=epoch)


def get_spike_counts(gn,condition,direction,window = 400,
                     tlim = [0,2000],alignment = 'TS',
                     redo = False,save=False):
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'alignment':alignment,
              'tlim':tlim}
    ORG = Organiser(params, 
        'experiment_spike_counts_file_condition'+str(condition)+\
            '_direction'+str(direction),redo = redo,save=save)
    return ORG.check_and_execute(_calc_spike_counts)


def _calc_spike_counts(params):
    
    data = load_data(params['gn'],params['condition'],params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    counts,tcounts = spiketools.sliding_counts(spiketimes, params['window'],dt=1,
                                    tlim  =params['tlim'])
    return counts,tcounts
