import sys;sys.path.append('src')
from matplotlib import pylab
import pandas as pd
import analyse_experiment as analyses
import joe_and_lili
from scipy.stats import wilcoxon
import pickle as pickle
import analyse_model
from reaction_times_functions import (
    reaction_time_plot, get_reaction_time_analysis)
import numpy as np
from GeneralHelper import (
    find, nice_figure, ax_label1, simpleaxis1)
path = 'preprocessed_and_simulated_data/'
def do_plot(extra_filters = [],min_count_rate = 5,min_trials  =10,tlim = [0,2000],
                alignment ='TS',ff_ax = None,cv2_ax = None,dir_score_ax = None,
                pop_score_ax=None,rate_ax=None,RTs_ax=None,
            condition_colors = ['0','0.3','0.6'],
            ff_test_interval = None,ff_test_point = 1000.,
            ff_test_ys = [0.1,2.],textsize=6,lw=3,lw_line=0.5):
    toc = joe_and_lili.get_toc(extra_filters = extra_filters)
    gns = pylab.unique(toc['global_neuron'])
    # find the gns and directions where criteria are met across conditions
    count_rate_block = pylab.zeros((len(gns),3,6))
    trial_count_block = pylab.zeros((len(gns),3,6))
    
    for i,gn in enumerate(gns):
        for j,condition in enumerate([1,2,3]):
            for k,direction in enumerate([1,2,3,4,5,6]):
                count_rate_block[i,j,k] =  analyses.get_mean_direction_counts(
                    gn,condition,direction,tlim  =tlim,alignment = alignment)
                trial_count_block[i,j,k]  =analyses.get_trial_count(
                    gn,condition,direction)
    
    enough_counts = pylab.prod(count_rate_block>=min_count_rate,axis=1)
    enough_trials = pylab.prod(trial_count_block>=min_trials,axis=1)

    good_directions = enough_counts * enough_trials

    monkey = extra_filters[0][-1].decode('UTF-8')
    if rate_ax is not None:
        pylab.sca(rate_ax)
        try:
            rate_gns,rate_conditions,rate_directions,rates,trate = pickle.load(
                open(path+'experiment_'+monkey+'_rate_file_'+alignment,'rb'))
            mask = rates < np.max(rates[:,:400])
        except:
            rate_gns = []
            rate_conditions = []
            rate_directions = []
            rates = []
            for i,gn in enumerate(gns):
                for j,condition in enumerate([1,2,3]):
                    for k,direction in enumerate([1,2,3,4,5,6]):
                        if good_directions[i,k]:
                            rate,trate = analyses.get_rate(
                                gn, condition, direction,
                            alignment = alignment,tlim  =tlim)
                            rates.append(rate[0])
                            rate_gns.append(gn)
                            rate_conditions.append(condition)
                            rate_directions.append(direction)
            rates = pylab.array(rates)
            rate_conditions = pylab.array(rate_conditions)
            mask = rates < np.max(rates[:,:400])
            pickle.dump((rate_gns,rate_conditions,
                         rate_directions,rates,trate),
                        open(path+'experiment_'+monkey+'_rate_file_'+alignment,
                             'wb'),protocol =2)

        for (condition,color) in zip([1,2,3],condition_colors):
            pylab.plot(trate, pylab.nanmean(
                rates[rate_conditions==condition],axis=0),
            color = color,label = 'condition '+str(condition))
    
    if ff_ax is not None:
        pylab.sca(ff_ax)
        try:
            ffs,tff,ff_conditions,ff_gns,ff_directions = pd.read_pickle(
                path+'experiment_'+monkey+'_ff_file_'+alignment)
        except:
        
            ff_gns = []
            ff_conditions = [] 
            ff_directions = []
            ffs = []
            for i,gn in enumerate(gns):
               for j,condition in enumerate([1,2,3]):
                    for k,direction in enumerate([1,2,3,4,5,6]):
                        if good_directions[i,k]:
                            ff,tff = analyses.get_ff(
                                gn, condition, direction,alignment = alignment,tlim  =tlim)
                            ffs.append(ff)
                            ff_gns.append(gn)
                            ff_conditions.append(condition)
                            ff_directions.append(direction)
            ffs = pylab.array(ffs)
            ff_conditions = pylab.array(ff_conditions)
            pickle.dump((ffs,tff,ff_conditions,ff_gns,ff_directions),open(
                path+'experiment_'+monkey+'_ff_file_'+alignment,'wb'),protocol = 2)

        for (condition,color) in zip([1,2,3],condition_colors):
            avg_ff = pylab.nanmean(ffs[ff_conditions==condition],axis=0)
            pylab.plot(tff, avg_ff-avg_ff[0],color = color,label = 'condition '+str(condition))

        if ff_test_interval is not None:
            
            for ntest,test_conditions in enumerate([[1,2],[2,3]]):
                interval_mask = (tff>ff_test_interval[0]) * (tff<ff_test_interval[1])
                test_time = tff[interval_mask]
                test_vals = ffs[:,interval_mask]
                test_vals1 = test_vals[ff_conditions == test_conditions[0]]
                test_vals2 = test_vals[ff_conditions == test_conditions[1]]
                scores = pylab.zeros_like(test_time)
                for i in range(len(test_time)):
                    s,p = wilcoxon(test_vals1[:,i],test_vals2[:,i])
                    scores[i] = p
                sigplot=pylab.zeros_like(scores)*pylab.nan
                sigplot[scores <0.05] = ff_test_ys[ntest]
                print('test_time' ,test_time)                
                pylab.plot(test_time,sigplot,lw = lw)
        if ff_test_point is not None:
            for ntest,test_conditions in enumerate([[1,2],[2,3]]):
                test_ind = pylab.argmin(pylab.absolute(tff-ff_test_point))
                test_time = tff[test_ind]
                test_vals = ffs[:,test_ind]
                test_vals1 = test_vals[ff_conditions == test_conditions[0]]
                test_vals2 = test_vals[ff_conditions == test_conditions[1]]
                s,p = wilcoxon(test_vals1[:],test_vals2[:])
                bottom_val = pylab.nanmean(test_vals1) - pylab.nanmean(
                    ffs[ff_conditions==test_conditions[0]],axis=0)[0]
                top_val = pylab.nanmean(test_vals2) - pylab.nanmean(
                    ffs[ff_conditions==test_conditions[1]],axis=0)[0]
                center = 0.5 * (bottom_val+top_val) 

        pylab.text(-1400,-0.58,'Experiment',ha = 'center',va ='bottom',
            size = labelsize+2,rotation=90,weight='bold')  

    if cv2_ax is not None:
        pylab.sca(cv2_ax)

        try:
            cv2_gns,cv2_conditions,cv2_directions,cv2s,tcv2 = pd.read_pickle(
                path+'experiment_'+monkey+'_cv2_file_'+alignment)
        except:
            cv2_gns = []
            cv2_conditions = [] 
            cv2_directions = []
            cv2s = []
            for i,gn in enumerate(gns):
                for j,condition in enumerate([1,2,3]):
                    for k,direction in enumerate([1,2,3,4,5,6]):
                        if good_directions[i,k]:
                            cv2,tcv2 = analyses.get_cv_two(gn, condition, direction,
                                alignment = alignment,tlim  =tlim)
                            cv2s.append(cv2)
                            cv2_gns.append(gn)
                            cv2_conditions.append(condition)
                            cv2_directions.append(direction)
            cv2s = pylab.array(cv2s)

            cv2_conditions = pylab.array(cv2_conditions)
            pickle.dump((cv2_gns,cv2_conditions,cv2_directions,cv2s,tcv2),open(
                path+'experiment_'+monkey+'_cv2_file_'+alignment,'wb'),protocol =2)

        for (condition,color) in zip([1,2,3],condition_colors):
            pylab.plot(tcv2, pylab.nanmean(cv2s[cv2_conditions==condition],axis=0),
            color = color,label = 'condition '+str(condition))

    if RTs_ax is not None:
        pylab.sca(RTs_ax)
        reaction_time_plot(extra_filters[0][2], 
                           condition_colors = condition_colors)
        pylab.ylabel('p.d.f')
        pylab.axvline(1500,linestyle = '-',color = 'k',lw = 0.5)
        pylab.ylim(0,0.015)    
        pylab.yticks([0,0.004,0.008,0.012])
        pylab.legend(frameon = False,fontsize = labelsize,loc = 'upper right', 
                     bbox_to_anchor=(1.5, 1.1))
        pylab.xticks([1500,1600,1700,1800,1900,2000])
        pylab.gca().set_xticklabels(['RS', '100','200','300','400','500'])



###################################
###########MODEL#################
condition_colors = ['navy','royalblue','lightskyblue']
redo = False

def plot_ffs(params,sig_time = 1000,plot = True,lw_line=0.5, 
                redo=False, save=False):
    ffs = analyse_model.get_fanos(params, redo=redo, save=save,
                                  datafile='supplfig3_model_ffs')

    if not plot:
        return
    time = ffs.pop('time') + 500
    conditions = params['sim_params']['conditions']
    if sig_time is not None:
        unit_set = set()
        for condition in conditions:
            unit_set = unit_set.union(set(ffs[condition]))
        units = pylab.array(list(unit_set))
        test_vals = [[] for c in conditions]
        test_ind = pylab.argmin(pylab.absolute(sig_time-time))
        for i,c in enumerate(conditions):
            test_vals[i] = [ffs[c][unit][test_ind] for unit in units]
        test_vals = pylab.array(test_vals)
        nan_rows = pylab.isnan(test_vals).sum(axis=0)
        good_inds = find(nan_rows==0)
        test_vals = test_vals[:,good_inds]
        good_units = units[good_inds]

    offset_lst = []
    for cond_cnt, condition in enumerate(conditions):
        condition_ffs = ffs[condition]
        all_ffs = []
        for u in good_units:
            all_ffs.append(condition_ffs[u])
        all_ffs = pylab.array(all_ffs)
        mean_ffs = pylab.nanmean(all_ffs,axis=0)
        offset_lst.append(pylab.nanmean(mean_ffs[:220]))

        pylab.plot(time,mean_ffs - offset_lst[cond_cnt],
            color = condition_colors[cond_cnt],label = 'condition '+str(condition))

    if sig_time is not None:
        sigs = []
        for i,c in enumerate(conditions[:-1]):
            try:
                s,p = wilcoxon(test_vals[i,:],test_vals[i+1,:])
            except:
                s,p =0,0
            sigs.append(p)
            if p<0.05:
                sig_symbol = '*'
                if p<0.01:
                    sig_symbol = '**'
                if p<0.001:
                    sig_symbol = '***'
                bottom_val = pylab.nanmean(test_vals[i,:]) -offset_lst[i]
                top_val = pylab.nanmean(test_vals[i+1,:]) - offset_lst[i+1]
                center = 0.5 * (bottom_val+top_val)

    pylab.ylabel(r'$\Delta$FF',rotation=90)
    pylab.xlabel('time [ms]')

def plot_cv2s(params,sig_time = 1000,plot = True,
              redo=False, save=False,lw_line=0.5):
    cv2s = analyse_model.get_cv_two(
        params,redo=redo,save=save, datafile='supplfig3_model_cv_twos')

    if not plot:
        return
    time = cv2s.pop('time') + 500
    conditions = params['sim_params']['conditions']
    if sig_time is not None:
        unit_set = set()
        for condition in conditions:
            unit_set = unit_set.union(set(cv2s[condition]))
        units = pylab.array(list(unit_set))
        test_vals = [[] for c in conditions]
        test_ind = pylab.argmin(pylab.absolute(sig_time-time))
        for i,c in enumerate(conditions):
            test_vals[i] = [cv2s[c][unit][test_ind] for unit in units]
        test_vals = pylab.array(test_vals)
        nan_rows = pylab.isnan(test_vals).sum(axis=0)
        good_inds = find(nan_rows==0)
        test_vals = test_vals[:,good_inds]
        good_units = units[good_inds]

    print('good units: ', len(good_units))    
    for condition in conditions:
        condition_cv2s = cv2s[condition]
        all_cv2s = []
        for u in good_units:
            all_cv2s.append(condition_cv2s[u])
        all_cv2s = pylab.array(all_cv2s)
        pylab.plot(time,pylab.nanmean(all_cv2s,axis=0),
                   color = condition_colors[condition-1],
            label = 'condition '+str(condition))

    pylab.ylabel("CV$_2$",math_fontfamily='dejavusans')
    pylab.xlabel('time [ms]')



def plot_rates(params,plot,redo=False, save=False):
    scores = analyse_model.get_rates(params,redo  =redo,save=save,
                                     datafile='supplfig3_model_rates')
    if not plot:
        return 
    time = scores.pop('time') + 500

    for condition in params['sim_params']['conditions']:
        condition_scores = scores[condition]
        rates_arr = np.array(list(map(lambda x: condition_scores[x], 
                                      condition_scores.keys())))
        pylab.plot(time,pylab.nanmean(rates_arr,axis=0),
                   color = condition_colors[condition-1])
    pylab.ylabel('rate', rotation=90)


def plot_RTs(params, redo=False,save=False):
    for tau in [50.]:
        for threshold_per_condition in [False]:
            for integrate_from_go in [False]:
                for min_count_rate in [7.5]:
                    for align_ts in [False]:
                        result = get_reaction_time_analysis(
                            params,tlim  =tlim,redo = redo,
                            tau  =tau,integrate_from_go = integrate_from_go,
                            normalise_across_clusters=True,
                            threshold_per_condition = threshold_per_condition,
                            save=save, 
                            fname='supply_fig3_reaction_time_analysis')
                        rts = result['rts']
                        conditions = result['conditions']
                        directions = result['directions']
                        predictions = result['predictions']
                        correct= directions == predictions
                        correct_inds = np.where(correct)[0]
                        incorrect_inds = np.where(correct==False)[0]             
                        for condition in [1,2,3]:
                            rt = rts[(conditions == condition)*correct]
                            bins = pylab.linspace(0,500,15)
                            pylab.hist(
                                rt,bins,histtype = 'step',
                                facecolor = condition_colors[condition-1],
                                density = True,
                                edgecolor  = condition_colors[condition-1],
                                label = 'condtion '+str(condition))
                            pylab.xlim(1400,2000)
                        min_len = min(len(rts[(conditions == 2)*correct]),
                                      len(rts[(conditions == 3)*correct]))

                            
                        pylab.xlim(-100,500)
                        pylab.xticks([0,100,200,300,400,500])
                        pylab.gca().set_xticklabels(
                            ['RS', '100','200','300','400','500'])
                        pylab.ylim(0,0.01)
                        pylab.yticks([0,0.004,0.008])                            
                        pylab.ylabel('p.d.f')
                        pylab.xlabel('reaction time [ms]')
                        pylab.axvline(0,linestyle = '-',color = 'k',lw = 0.5)  

###########
##########
    
if __name__ == '__main__':
    labelsize = 8
    labelsize1 = 7 
    ticksize =2.
    size = 6
    scale=1.5
    lw= 0.3
    rcparams = {'axes.labelsize': size*scale,
                'xtick.major.size': ticksize,
                'ytick.major.size': ticksize,            
              'xtick.labelsize':size,
                'ytick.labelsize': size,
                'lines.linewidth':0.5,
                'axes.linewidth':0.2}

    fig = nice_figure(ratio  =.6,rcparams = rcparams)
    fig.subplots_adjust(hspace = .9,wspace = 0.5,bottom  =0.14,
                        top =0.9,left=0.15, right=0.9)
    tlim = [0,2000]
    xticks = [0,500,1000,1500,2000]
    nrow,ncol = 4, 3
    pad=.3
    x_label_val=-0.5
    size_cond = 15
    condition_colors_exp = ['navy','royalblue','lightskyblue']    
    for monkey in ['lili']:
        extra_filters = [('monkey','=',str.encode(monkey))]
        rate_ax = ax_label1(simpleaxis1(pylab.subplot2grid(
            (nrow,ncol),(0,1)),labelsize,pad=pad),'b',
                x=x_label_val,size=labelsize,y=1.25)
        ff_ax = ax_label1(simpleaxis1(
            pylab.subplot2grid(
                (nrow,ncol),(0,0),rowspan=2),labelsize,pad=pad),'a',
                x=x_label_val,size=labelsize)
        cv2_ax = ax_label1(simpleaxis1(pylab.subplot2grid(
            (nrow,ncol),(1,1)),labelsize,pad=pad),'c',
                x=x_label_val,size=labelsize)
        RTs_ax = ax_label1(simpleaxis1(
            pylab.subplot2grid(
                (nrow,ncol),(0,2),rowspan=2),labelsize,pad=pad),'d',
                x=x_label_val,size=labelsize,y=1.05)
        do_plot(extra_filters = extra_filters,ff_ax = ff_ax,rate_ax = rate_ax, 
                    cv2_ax=cv2_ax,RTs_ax=RTs_ax,textsize=size,
                    lw=1,lw_line=0.3, condition_colors=condition_colors_exp)


    pylab.sca(rate_ax)
    pylab.xticks(xticks)
    pylab.ylabel('rate',rotation=90)
    pylab.xlim(tlim)
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)

    pylab.ylim(0.,30)

    pylab.sca(cv2_ax)
    pylab.ylim(0.4,1.3)
    pylab.xticks(xticks)
    pylab.yticks([0.4,0.8,1.2])    
    pylab.ylabel('CV$_2$',rotation=90,math_fontfamily='dejavusans')

    pylab.xlim(tlim)
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
    
    pylab.sca(ff_ax)
    pylab.xlim(tlim)
    pylab.xticks([])
    pylab.ylabel(r'$\Delta$FF',rotation=90)

    pylab.ylim(-0.7,0.1)
    pylab.yticks([-0.5,0])

    
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.text(500, pylab.ylim()[1],'PS',va = 'bottom',
               ha = 'center',size = labelsize1)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.text(1500, pylab.ylim()[1],'RS',va = 'bottom',
               ha = 'center',size = labelsize1)

    ########################################
    ###################MODEL###############
    params = {'randseed':8721,'trials':150,'N_E':1200,'N_I':300,
                    'I_th_E':1.25,'I_th_I':0.78,'Q':6,'rs_stim_amp':0,
                    'n_jobs':4,'conditions':[1,2,3]}

    settings = [{'randseed':7745,'jep':3.2,'jipratio':0.75,
        'condition_stim_amps':[0.05,.05,.05],'rs_stim_amp':0.1,
        'rs_length':400,'trials':400}]

    plot = True
    redo_model = False
    save=True
    for setno,setting in enumerate(settings):
        for k in list(setting.keys()):
            params[k] = setting[k]
        min_count_rate = 5.
        plot_params = {'sim_params':params,
                        'min_count_rate':round(float(min_count_rate),4), 
                         'cvtwo_win':400}

        plot_params['timestep'] = 5

        # plot rates
        ax_label1(simpleaxis1(pylab.subplot2grid(
            (nrow,ncol),(2,1)),labelsize,pad=pad),'f',
                  x=x_label_val,size=labelsize)      
        plot_rates(plot_params,plot = plot,redo=redo_model,save=save)
        pylab.ylim(0,40)
        
        ax_label1(simpleaxis1(
            pylab.subplot2grid((nrow,ncol),(2,0),rowspan=2),
            labelsize,pad=pad),'e',x=x_label_val,
                  size=labelsize, y=1.05)            
        
        plot_ffs(plot_params,plot = plot,redo=redo_model,save=save)
        pylab.axvline(500,linestyle = '-',color = 'k',lw = lw)
        pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw)
        pylab.xlim(0,2000)
        pylab.yticks([-0.5,0])
        pylab.ylim(-.7,0.2)
        pylab.xlabel('')
        pylab.xlabel('time [ms]') 
        pylab.text(-1300,-0.6,'E/I clustered\n model',
                   ha = 'center',va ='bottom',
                size = labelsize+2,rotation=90,weight='bold')  
        ax_label1(simpleaxis1(pylab.subplot2grid(
            (nrow,ncol),(3,1)),labelsize,pad=pad),'g',
            x=x_label_val,size=labelsize)          
    
        plot_cv2s(plot_params,plot = plot,redo=redo_model,save=save)
        pylab.axvline(500,linestyle = '-',color = 'k',lw = lw)
        pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw)
        pylab.xlim(0,2000)
        pylab.ylim(0.4,1.3)
        pylab.yticks([0.4,0.8,1.2])
        ax_label1(simpleaxis1(
            pylab.subplot2grid((nrow,ncol),(2,2),rowspan=2),
            labelsize,pad=pad),'h',x=x_label_val,
                  size=labelsize, y=1.0) 
        plot_RTs(plot_params,redo=redo_model,save=save)

    pylab.savefig('suppl_fig3.pdf')
    pylab.close()
