
import os
from warnings import warn
import pickle as pickle
import pylab
import numpy as np
import pandas as pd
import numpy as np
gdf_path = 'preprocessed_and_simulated_data/'
pickle_path = 'inputs/data/pickle'


def find(condition):
    """old pylab.find function"""
    res, = np.nonzero(np.ravel(condition))
    return res

marker_list = {'TS':'700','PS':'11','RS':'12','MO':'13',
               'ME':'14','RW':'19','TE':'20'}


standard_filters = [('baseline_rate','<',50),('small_intervals_2','<=',0.05),
                    ('sudden_changes','=',False),('all_conditions','=',True),
                    ('min_direction_trials','>=',5),('prep_rate','>=',1)]



missing = 0
def read_gdf(fname): 
    """ reads a gdf file and returns the markers and timestamps."""
                
    gdf = open(fname)
    
    content = gdf.read()
    
    gdf.close()
    
    markers = []
    timestamps = []
    
    for line in content.split('\n'):
        line = line.strip()
        if len(line)==0:
            break
        marker,timestamp = line.split()
        timestamp = int(timestamp)
        markers.append(marker)
        timestamps.append(timestamp)
       
    markers = pylab.array(markers)
    timestamps = pylab.array(timestamps)
     
    return markers,timestamps
       
def get_trial_matrix(markers,check = False):
    """ takes the markers from a gdf file and aranges them per trial.
        missing values are filled with nan, i.e. trials contianing 
        nan entries are inclompete.
        """
        
    # expected order:
    # TS,PS,RS,MO,ME,RW,TE
    names = ['TS','PS','RS','MO','ME','RW','TE']
    trials = []
    directions = []
    
    
    marker_inds = [find(markers == marker_list[n]) for n in names]
    
    for i in range(len(marker_inds[0])-1):
        trial = [marker_inds[0][i]] 
        directions.append(int(markers[trial[0]+1][-1]))
        for marker_ind in marker_inds[1:]:
            marker = marker_ind[(marker_ind>trial[-1]) * (marker_ind<marker_inds[0][i+1])]   
            
            if len(marker)<1:
                trial.append(pylab.nan)
            else:
                trial.append(marker)
        
        trials.append(trial)
    # now add the last trial
    trial = [marker_inds[0][-1]]
    directions.append(int(markers[trial[0]+1][-1]))
    for marker_ind in marker_inds[1:]:
        marker = marker_ind[marker_ind>trial[-1]]    
        
        if len(marker)<1:
            trial.append(pylab.nan)
        else:
            trial.append(marker)
    #print len(trial)
    trials.append(trial)
    
    trials = pylab.array(trials)
    
    if check:
        # make sure that all indices are allways increasing
        flat_trials = trials.flatten()
        diff = pylab.diff(flat_trials[pylab.isnan(flat_trials)==False])
        if (diff<0).any():
            import warnings
            warnings.warn('inconsitency found in trial marker matrix')
    
    return trials,pylab.array(names),pylab.array(directions)
        
def cut_gdf(fname,cut_marker='TS',cut_window = [-500,2500]):
    """ reads gdf file and returns its content as spiketimes per neuron 
        and trial.
        """
    
    
    
    markers,timestamps = read_gdf(fname)
    
    trials,marker_names,directions = get_trial_matrix(markers,check = True)
    
    # find position of cut marker in trials matrix
    cut_marker_ind = find(marker_names == cut_marker)[0]
    
    # remove incomplete trials
    good_trials = pylab.isnan(trials[:,cut_marker_ind])==False
    if (good_trials ==False).any():
        missing+=1
        print(missing)   
    trials = trials[good_trials,:].astype(int)
    directions = directions[good_trials]
       
    
    
    
    # spiketime,trial,neuron,direction
    column_names = ['spiketime','trial','neuron','direction','absolute_trial_time']
    spikedata = None
    for trial in range(trials.shape[0]):
        #remember whether a spike was found in the current trial
        found_spike = False
        # index of the cut event
        cut_ind =  trials[trial,cut_marker_ind]
        
        # corresponding timestamp
        cut_time = timestamps[cut_ind]
        
         # the trial matrix is now used to store the evnet times
        trials[trial,trials[trial,:]>0] = timestamps[trials[trial,trials[trial,:]>0]] -cut_time
        
        cut_start_ind = find(timestamps>=cut_time+cut_window[0])[0]
        cut_end_ind = find(timestamps<cut_time+cut_window[1])[-1]
        
        trial_markers = markers[cut_start_ind:cut_end_ind].copy()
        trial_timestamps = timestamps[cut_start_ind:cut_end_ind].copy()
        
        # remove cut_time from trial timestamps
        trial_timestamps -= cut_time
        
        for mark in pylab.unique(trial_markers):
            if len(mark) == 1:
                #is a neuron
                found_spike = True
                neuron_inds = find(trial_markers == mark)
                newdata = pylab.zeros((len(neuron_inds),5))
                newdata[:,0] = trial_timestamps[neuron_inds]
                newdata[:,1] = trial
                newdata[:,2] = int(mark)
                newdata[:,3] = directions[trial]
                newdata[:,4] = cut_time
                if spikedata is None:
                    spikedata = newdata
                else:
                    spikedata = pylab.append(spikedata,newdata,axis= 0)
        if not found_spike:
            newdata = pylab.zeros((1,5))
            newdata[:,0] = pylab.nan
            newdata[:,1] = trial
            newdata[:,2] = pylab.nan
            newdata[:,3] = directions[trial]
            newdata[:,4] = cut_time
            if spikedata is None:
                spikedata = newdata
            else:
                spikedata = pylab.append(spikedata,newdata,axis= 0)
            
                    
    # there are some double spikes in the gdfs
    # this geds rid of them
    spikedata = pylab.array(list(set([tuple(spikedata[i,:]) for i in range(spikedata.shape[0])])))
    
    
    # sort first for time
    order = pylab.argsort(spikedata[:,0],kind='heapsort')
    spikedata = spikedata[order,:]
    # then for trial
    order = pylab.argsort(spikedata[:,1],kind='heapsort')
    spikedata = spikedata[order,:]
    
    
        
    
    return spikedata,column_names,trials,marker_names

def gdf2pickle(fname,outpath,cut_window = [0,3000]):
    
    spiketimes,cnames,events,eventnames = cut_gdf(fname,cut_marker = 'TS',cut_window = cut_window)
    
    fname = os.path.split(fname)[-1]
    print('converting '+fname)
    session,neurons = fname[:-4].split('-')
    all_trials = set(spiketimes[:,1])
    
    for neuron in neurons:
        
        neuron = int(neuron)
        
        neurondata = spiketimes[spiketimes[:,2]==neuron,:][:,[0,1,3]]
        
        neuron_trials = set(neurondata[:,1])
        missing_trials = all_trials.difference(neuron_trials)
        if len(missing_trials)>0:
            #print 'trials missing'
            missingdata = pylab.zeros((len(missing_trials),3))
            for i,mt in enumerate(missing_trials):
                direction = spiketimes[spiketimes[:,1]==mt,3][0]
                
                missingdata[i,:] = pylab.nan,mt,direction
            neurondata = pylab.append(neurondata,missingdata,axis = 0)
        
        data = {'spiketimes':neurondata.T,'column_names':['spiketime','trial','direction'],\
                'eventtimes':events,'event_names':eventnames,'gdf_file':fname}
        
        file = open(os.path.join(pickle_path,session+'_'+str(neuron)+'.pickle'),'w')
        pickle.dump(data,file,protocol = 2)
        file.close()
     
def get_toc(filters = standard_filters,extra_filters = []):
    
    dpath = pickle_path
    if 'toc' in os.listdir(dpath):
        toc = pd.read_pickle(os.path.join(dpath,'toc'))
    else:
        # generate new toc
        print('creating new toc')
        #information contained in the filename
        file = []
        monkey = []
        session = []
        neuron = []
        for f  in os.listdir(dpath):
            if not 'pickle' in f:
                continue
            file.append(f)
            if 'joe' in f:
                monkey.append('joe')
                f = f.replace('joe','')
            elif 'lili' in f:
                monkey.append('lili')
                f = f.replace('lili','')
            s,n = f.split('.')[0].split('_')
            session.append(int(s))
            neuron.append(int(n))
        toc = {'file':pylab.array(file),\
               'monkey':pylab.array(monkey),\
               'session':pylab.array(session),\
               'neuron':pylab.array(neuron)}   
        
        # extract condition_information from sparse files
        sparse_names = os.listdir(os.path.join(gdf_path,'joe','sparse'))
        sparse_names += os.listdir(os.path.join(gdf_path,'lili','sparse'))
        
        condition = pylab.zeros_like(toc['session'])
        
        for sn in sparse_names:
            if not 'TS' in sn:
                continue
            monkeysession = sn.split('-')[0]
            session = monkeysession[-3:]
            monkey = monkeysession.replace(session,'')
            session = int(session)
            c = int(sn.split('-')[2][1])
            condition[(toc['monkey'] ==monkey)*(toc['session']==session)] = c
        toc['condition'] = condition
        
        # add information from martins file lists
        joefile = open(os.path.join(gdf_path,'joe','gdf','FileConditionList15June00.txt'))
        
        content = [line for line in joefile.read().split('Nawrot')[1].split('_')[0].strip().replace('@','').split('\n') if line !='']
        # initialize global neuron count
        global_neurons = pylab.zeros_like(toc['condition'])
        gn = 0
        for i,line in enumerate(content):
            if 'Y' in line or 'N' in line:
                
                if 'Y' in line :
                    #accepted
                    gn += 1
                    
                    for j in range(i,i+3):
                        line = content[j].split()[0].replace('joe','')
                        session=int(line[:-1])
                        neuron=int(line[-1])
                        global_neurons[(toc['monkey'] == 'joe')*(toc['session'] == session)*(toc['neuron']==neuron)] = gn
                        if len(global_neurons[(toc['monkey'] == 'joe')*(toc['session'] == session)*(toc['neuron']==neuron)])==0:
                            print('missing joe ',session ,neuron)
        
        
        lilifile = open(os.path.join(gdf_path,'lili','Lili_Fano_2.txt'))
        content = [line for line in lilifile.read().split('% conditions.')[1].strip().replace('*','').split('\n') if line !='']
        for i,line in enumerate(content):
            if 'ok' in line or 'n' in line:
                
                if 'ok' in line :
                    #accepted
                    gn += 1
                    
                    for j in range(i,i+3):
                        line = content[j].split()[0].replace('lili','')
                        session=int(line[:-1])
                        neuron=int(line[-1])
                        global_neurons[(toc['monkey'] == 'lili')*(toc['session'] == session)*(toc['neuron']==neuron)] = gn
                        if len(global_neurons[(toc['monkey'] == 'lili')*(toc['session'] == session)*(toc['neuron']==neuron)])==0:
                            print('missing lili',session ,neuron)
        
        
        
        toc['global_neuron'] = global_neurons      
        
        # neurons not contained in martins lists have zeror gn and are deleted
        
        mask = global_neurons>0
        
        for k in list(toc.keys()):
            toc[k] = toc[k][mask]
        
        pickle.dump(toc,open(os.path.join(dpath,'toc'),'w'),protocol = 2)  
    
    
    for key,mode,value in filters+extra_filters:
        if key not in list(toc.keys()):
            warn('toc does not have column '+key)
            continue
        if mode=='=':
            mask = toc[key]==value
            
        
        elif mode=='in':
            mask = toc[key]==value[0]
            for v in value[1:]:
                mask += toc[key]==v
        elif mode == '<':
            mask = toc[key]<value
        elif mode == '>':
            mask = toc[key]>value
        elif mode == '>=':
            mask = toc[key]>=value
        elif mode == '<=':
            mask = toc[key]<=value
        
        

        for k in list(toc.keys()):
            toc[k] = toc[k][mask]

        
    
    # now make sure every neuron is present for all conditions
    
    good_inds = []
    
    for gn in pylab.unique(toc['global_neuron']):
        inds = find(toc['global_neuron'] == gn)
        if len(inds)==3:
            good_inds+=inds.tolist()
    
    for k in list(toc.keys()):
        toc[k] = toc[k][good_inds]       
                
    return toc


if __name__ == '__main__':
    
    toc = get_toc()
