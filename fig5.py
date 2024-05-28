import sys;sys.path.append('src')#;sys.path.append('../data')
from matplotlib import pylab
import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon
import pickle as pickle
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

# Local modules (not installed packages)
import analyse_model
import analyse_experiment as analyses
import joe_and_lili
from network_schematic import schematic_2d_model
from GeneralHelper import (
    nice_figure, ax_label1, simpleaxis1, find
)


path = 'preprocessed_and_simulated_data/'

########################################
########### EXPERIMENT #################
########################################

# function to plot the results of the experiment analysis
def do_plot(extra_filters = [],min_count_rate = 5,
            min_trials  =10,tlim = [0,2000],
            alignment ='TS',ff_ax = None,cv2_ax = None,
            dir_score_ax = None,pop_score_ax=None,
            rate_ax=None, condition_colors = ['0','0.3','0.6'],
            ff_test_interval = None,
            ff_test_point = 1000.,ff_test_ys = [0.1,2.],
            textsize=6,lw=3,lw_line=0.5):
    """plot the results of the experiment analysis"""
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


    if rate_ax is not None:
        print('plotting firing rates of experiments ... ')
        pylab.sca(rate_ax)

        try:
            rate_gns,rate_conditions,rate_directions,rates,trate = pd.read_pickle(
                path+'experiment_'+monkey+'_rate_file_'+alignment)
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
                                gn, condition, direction,alignment = alignment,
                                tlim  =tlim)
                            rates.append(rate[0])
                            rate_gns.append(gn)
                            rate_conditions.append(condition)
                            rate_directions.append(direction)
            rates = pylab.array(rates)
            rate_conditions = pylab.array(rate_conditions)
            pickle.dump(
                (rate_gns,rate_conditions,rate_directions,rates,trate),
                open(
                    path+'experiment_'+monkey+'_rate_file_'+alignment,
                    'wb'),protocol =2)
        for (condition,color) in zip([1,2,3],condition_colors):
            pylab.plot(
                trate, 
                pylab.nanmean(
                    rates[rate_conditions==condition],axis=0),
                color = color,label = 'condition '+str(condition))
    if ff_ax is not None:
        print('plotting fano factors of experiments ... ')
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
                                gn, condition, direction,alignment = alignment,
                                tlim  =tlim)
                            ffs.append(ff)
                            ff_gns.append(gn)
                            ff_conditions.append(condition)
                            ff_directions.append(direction)
            ffs = pylab.array(ffs)
            ff_conditions = pylab.array(ff_conditions)
            pickle.dump(
                (ffs,tff,ff_conditions,ff_gns,ff_directions),
                open(path+'experiment_'+monkey+'_ff_file_'+alignment,
                     'wb'),protocol = 2)
        for (condition,color) in zip([1,2,3],condition_colors):
            avg_ff = pylab.nanmean(ffs[ff_conditions==condition],axis=0)
            pylab.plot(tff, avg_ff-avg_ff[0],
                       color = color,label = 'condition '+str(condition))
        if ff_test_interval is not None:
            for ntest,test_conditions in enumerate([[1,2],[2,3]]):
                interval_mask = (tff>ff_test_interval[0]) * (tff<ff_test_interval[1])
                test_time = tff[interval_mask]
                test_vals = ffs[:,interval_mask]
                test_vals1 = test_vals[ff_conditions == test_conditions[0]]
                test_vals2 = test_vals[ff_conditions == test_conditions[1]]
                scores = pylab.zeros_like(test_time)
                for i in range(len(test_time)):
                    s,p = wilcoxon(test_vals1[:,i],test_vals2[:,i],
                                   nan_policy='omit')
                    scores[i] = p
                sigplot=pylab.zeros_like(scores)*pylab.nan
                sigplot[scores <0.05] = ff_test_ys[ntest]            
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
                pylab.plot([test_time]*2,[bottom_val+0.02,top_val-0.02],
                           '-_k',lw =lw_line,ms = 2.)
                pylab.text(test_time-10, center+0.05, '*',
                           va = 'top',ha ='right',size = 6)



    if cv2_ax is not None:
        print('plotting cv2 of experiments ... ')
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
                            cv2,tcv2 = analyses.get_cv_two(
                                gn, condition, direction,
                                alignment = alignment,tlim  =tlim)
                            cv2s.append(cv2)
                            cv2_gns.append(gn)
                            cv2_conditions.append(condition)
                            cv2_directions.append(direction)
            cv2s = pylab.array(cv2s)

            cv2_conditions = pylab.array(cv2_conditions)

            pickle.dump(
                (cv2_gns,cv2_conditions,cv2_directions,cv2s,tcv2),
                open(path+'experiment_'+monkey+'_cv2_file_'+alignment,'wb'),
                protocol =2)

        for (condition,color) in zip([1,2,3],condition_colors):
            pylab.plot(
                tcv2, 
                pylab.nanmean(cv2s[cv2_conditions==condition],axis=0),
                color = color,label = 'condition '+str(condition))


    if pop_score_ax is not None:
        print('plotting decoding accuracy of experiments ... ')
        pylab.sca(pop_score_ax)
        try:
            pop_score_gns,pop_score_conditions,pop_scores,tpop_score = pd.read_pickle(
                path+'experiment_'+monkey+'_pop_score_file_'+alignment)
        except:
            pop_score_gns = []
            pop_score_conditions = [] 
            pop_scores = []
            for i,gn in enumerate(gns):
                for j,condition in enumerate([1,2,3]):
                    result= analyses.get_population_decoding(
                        gns, condition,alignment = alignment,
                        tlim=tlim,n_jobs = 12,reps=10,save=False)

                    pop_score = pylab.array([r[0] for r in result]).mean(axis=0)
                    tpop_score = result[0][1]
                    
                    pop_scores.append(pop_score)
                    pop_score_gns.append(gn)
                    pop_score_conditions.append(condition)
                    
            pop_scores = pylab.array(pop_scores)
            pop_score_conditions = pylab.array(pop_score_conditions)
            pickle.dump(
                (pop_score_gns,pop_score_conditions,pop_scores,tpop_score),
                open(path+'experiment_'+monkey+'_pop_score_file_'+alignment,'wb'),
                protocol =2)

        for (condition,color) in zip([1,2,3],condition_colors):
            pylab.plot(
                tpop_score, 
                pylab.nanmean(pop_scores[pop_score_conditions==condition],
                              axis=0),color = color,
                label = 'condition '+str(condition))
        
        pylab.ylim(0,1)
        pylab.axhline(1/6.,linestyle = '--',lw = lw_line,color = '0.4',zorder = -1)
        pylab.text(2000, 1/6., r'$1/6$',va= 'center',ha = 'left',size = textsize)
        pylab.axhline(1/3.,linestyle = '--',lw = lw_line,color = '0.4',zorder = -1)
        pylab.text(2000, 1/3., r'$1/3$',va= 'center',ha = 'left',size = textsize)
        pylab.axhline(1/2.,linestyle = '--',lw = lw_line,color = '0.4',zorder = -1)
        pylab.text(2000, 1/2., r'$1/2$',va= 'center',ha = 'left',size = textsize)



###################################
########### MODEL #################
###################################
condition_colors = ['0','0.3','0.6']
condition_colors = ['navy','royalblue','lightskyblue']
redo = False

# functions to plot fano factor, cv2, 
# population decoding, and firing rates of models

def plot_ffs(params,sig_time = 1000,plot = True,lw_line=0.5, redo=False):
    """plot fano factors of the model"""
    ffs = analyse_model.get_fanos(params, redo=redo,datafile='fig5_model_fanos')

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
    for condition in conditions:
        condition_ffs = ffs[condition]
        all_ffs = []
        for u in good_units:
            all_ffs.append(condition_ffs[u])
        all_ffs = pylab.array(all_ffs)
        mean_ffs = pylab.nanmean(all_ffs,axis=0)
        offset_lst.append(pylab.nanmean(mean_ffs[:220]))
        pylab.plot(time,mean_ffs - offset_lst[condition-1],
                   color = condition_colors[condition-1],
                   label = 'condition '+str(condition))

    if sig_time is not None:
        sigs = []

        for i,c in enumerate(conditions[:-1]):
            s,p = wilcoxon(test_vals[i,:],test_vals[i+1,:],nan_policy='omit')
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
                pylab.plot([sig_time]*2,[bottom_val+0.02,top_val-0.02],'-_k',
                           lw =lw_line/2,ms = 2.)
                pylab.text(sig_time-10, center+0.05, sig_symbol,
                           va = 'top',ha ='right',size = 6)

    pylab.ylabel(r'$\Delta$FF',rotation=90)

def plot_cv2s(params,sig_time = 1000,plot = True,redo=False, lw_line=0.5):
    """plot cv2s of the model"""
    cv2s = analyse_model.get_cv_two(params,redo=redo,
                                    datafile='fig5_model_cv_twos')

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


def plot_population_decoding(params,plot, redo=False):
    """plot population decoding of the model"""
    scores = analyse_model.get_population_decoding(
        params,redo  =redo, datafile='fig5_model_popdecoding')
    if not plot:
        return 
    
    time = scores.pop('time') + 500

    for condition in params['sim_params']['conditions']:
        condition_scores = scores[condition]
        
        pylab.plot(time,pylab.nanmean(condition_scores,axis=0),
                   color = condition_colors[condition-1])
    
    pylab.ylabel('DA', rotation=90)

def plot_rates(params,plot,redo=False):
    """plot firing rates of the model"""
    scores = analyse_model.get_rates(params,redo  =redo,
                                     datafile='fig5_model_rates')
    if not plot:
        return 
    time = scores.pop('time') + 500

    for condition in params['sim_params']['conditions']:
        condition_scores = scores[condition]
        rates_arr = pylab.array(list(map(lambda x: condition_scores[x], 
                                      condition_scores.keys())))
        pylab.plot(time,pylab.nanmean(rates_arr,axis=0),
                   color = condition_colors[condition-1])
    
    pylab.ylabel('rate', rotation=90)
    pylab.xlabel('time [ms]')

    
def draw_hex_array(center,size=0.3,colors = [[0.5]*3]*6,
                   axes = None,radius = 0.1,add = True,
                   show_numbers = False,edgecolor='k'):
    """draw a hexagonal array of circles for the model sketch"""
    angles = pylab.array([30,90,150,210,270,330])*pylab.pi/180
    Y = size*pylab.cos(angles)
    X = size*pylab.sin(angles)
    i = 0
    circs= []
    coords = []
    number = 6
    for x,y in zip(X,Y):
        coords.append((x+center[0],y+center[1]))
        circ = pylab.Circle((x+center[0],y+center[1]), radius=radius,  
                            fc=colors[i], edgecolor=edgecolor)
        if axes is None:
            axes = pylab.gca()
        if add:
            axes.add_patch(circ)
        circs.append(circ)
        if show_numbers:
            pylab.text(x+center[0],y+center[1],str(number),size = 6,
                       ha ='center',va = 'center')
            if number == 6:
                number =1
            else:
                number+=1
        i+=1
    return circs,coords


def model_plot(ax, net_factor = 1.6,net_offset = 1.,
               offset_dir = [1.2,0.4],net_sigma = 0.12,N = 1200,
               decoder_factor = 0.7,decoder_offset = 1,randseed=0,colors = None):
    """plot the model sketch"""
    pylab.seed(randseed)
    stim_center = [0,0]
    stim_size  =0.3
    if colors is None:
        colors = [red]+5*[off_color]
    circs,coords = draw_hex_array(stim_center,size  =stim_size,colors =colors)
    for circ in circs:
        circ.set_zorder(4)
    Q = len(circs)
    cluster_centers = []
    positions = pylab.zeros((0,2))
    for coord in coords:
        new_pos =pylab.randn(int(N/Q),2)*net_sigma
        new_pos[:,0] += coord[0]*net_factor + net_offset*offset_dir[0]
        new_pos[:,1] += coord[1]*net_factor + net_offset*offset_dir[1]
        
        cluster_centers.append((coord[0]*net_factor + net_offset*offset_dir[0],
                                coord[1]*net_factor + net_offset*offset_dir[1]))
        positions = pylab.append(positions, new_pos,axis=0)
    ax.plot(positions[:,0],positions[:,1],'ok',ms = 0.2,zorder = 2 )
    connectivity = pylab.rand(N,N)<0.006
    connections = []
    for i in range(N):
        js = find(connectivity[i])
        for j in js:
            connections.append([positions[i],positions[j]])
    
    color = [0.9]*3 + [0.05]
    colors = [color]*len(connections)
    lines = LineCollection(connections,colors = colors,linewidths =0.04,zorder =2)
    ax.add_collection(lines)
    for i in range(6):
        ax.annotate('', xy=cluster_centers[i], xycoords='data',
                xytext=coords[i], textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple,tail_width=0.06,head_width=0.15",
                                fc=[0,0,0,1.], ec="none",
                                connectionstyle="arc3,rad=-0.3"),zorder = 3
                )
    cluster_center_center = pylab.array(cluster_centers).mean(axis=0)
    edge_length = stim_size * decoder_factor
    decoder_center = pylab.array(
        (cluster_center_center[0] + decoder_offset*offset_dir[0],
         cluster_center_center[1] + decoder_offset*offset_dir[1]))
    rectangle = Rectangle(decoder_center-0.5*edge_length, edge_length, 
                          edge_length,ec ='k',fc ='w')

    for i in range(6):
        ax.annotate('', xy=decoder_center, xycoords='data',
                xytext=cluster_centers[i], textcoords='data',
                size=20,
                arrowprops=dict(
                    arrowstyle="-",
                    fc=[0,0,0,1], ec="k",
                    connectionstyle="arc3,rad=0.3",
                    shrinkB=0),zorder = 2,
                )
    rectangle.set_zorder(0)
    ax.add_patch(rectangle)
    pylab.text(stim_center[0],stim_center[1]-stim_size*1.5,
               'stimulus',ha = 'center',va = 'top',size = 20)
    pylab.text(decoder_center[0],decoder_center[1]+edge_length,
               'decoder',ha = 'center',va ='bottom',size = 20)
    pylab.text(stim_center[0]-.79,(decoder_center[1] - stim_center[1])/2-1.16,
               'E/I Clustered model',ha = 'center',va ='bottom',
               size = 27,rotation=90,weight='bold') 
    pylab.axis('equal')


  
    

labelsize = 8
labelsize1 = 6    
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

fig = nice_figure(ratio  =.5,rcparams = rcparams)
fig.subplots_adjust(
    hspace = .5,wspace = 0.5,bottom  =0.14,top  =0.95)
tlim = [0,2000]
xticks = [0,500,1000,1500,2000]
nrow,ncol = 7, 3
pad=.3
x_label_val=-0.5
size_cond = 20
##########################
###EXPERIMENT PROTOCOL####
##########################
exp_ax = simpleaxis1(pylab.subplot2grid((nrow,ncol),(0,0),rowspan=3),
                                labelsize,pad=pad)
ax_label1(exp_ax,'a',x=x_label_val,size=labelsize,y=1.08)

pylab.sca(exp_ax)
pylab.axis('off')
condition_colors_exp = ['navy','royalblue','lightskyblue']    


for monkey in ['joe']:
    
    extra_filters = [('monkey','=',str.encode(monkey))]
    pop_score_ax = ax_label1(simpleaxis1(
        pylab.subplot2grid((nrow,ncol),(0,1),rowspan=2),labelsize,pad=pad),
                             'b',x=x_label_val,size=labelsize)
    rate_ax = simpleaxis1(pylab.subplot2grid((nrow,ncol),(2,1)),labelsize,pad=pad)
    ff_ax = ax_label1(simpleaxis1(
        pylab.subplot2grid((nrow,ncol),(0,2),rowspan=2),labelsize,pad=pad),
                      'c',x=x_label_val,size=labelsize)
    cv2_ax = simpleaxis1(pylab.subplot2grid((nrow,ncol),(2,2)),labelsize,pad=pad)
    do_plot(extra_filters = extra_filters,ff_ax = ff_ax,rate_ax = rate_ax, 
               pop_score_ax = pop_score_ax,cv2_ax=cv2_ax,textsize=size,
               lw=1,lw_line=0.3, condition_colors=condition_colors_exp)


pylab.sca(cv2_ax)
pylab.ylim(0.4,1.3)
pylab.xticks(xticks)
pylab.yticks([0.4,0.8,1.2])    
pylab.ylabel('CV$_2$',rotation=90, math_fontfamily='dejavusans')
pylab.xlabel('time [ms]')    
pylab.xlim(tlim)
pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
#pylab.text(1500, pylab.ylim()[1],'RS',va = 'bottom',ha = 'center',size = labelsize1)
#pylab.fill_betweenx([0.4,1.2],1500,2000,color='gray',alpha=0.5)

pylab.sca(ff_ax)
pylab.xlim(tlim)
pylab.xticks([])
pylab.ylabel(r'$\Delta$FF',rotation=90)
pylab.ylim(-0.7,0.1)
pylab.yticks([-0.5,0])
pylab.legend(frameon = False,fontsize = labelsize-1,loc = 'upper center',
                bbox_to_anchor=(1.2, 1.1))

pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
pylab.text(500, pylab.ylim()[1],'PS',va = 'bottom',ha = 'center',size = labelsize-1)
pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
pylab.text(1500, pylab.ylim()[1],'RS',va = 'bottom',ha = 'center',size = labelsize-1)
#pylab.fill_betweenx([-.7,0.1],1500,2000,color='gray',alpha=0.5)

pylab.sca(pop_score_ax)
pylab.xlim(tlim)
pylab.xticks([])
pylab.ylabel('DA',rotation=90)
pylab.ylim(0.1,1.)
pylab.yticks([0.1,0.4,0.7,1.])
#pylab.fill_betweenx([0.1,1],1500,2000,color='gray',alpha=0.5)
pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
pylab.text(500, pylab.ylim()[1],'PS',va = 'bottom',ha = 'center',size = labelsize)
pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
pylab.text(1500, pylab.ylim()[1],'RS',va = 'bottom',ha = 'center',size = labelsize)
# FIRING RATES
pylab.sca(rate_ax)
pylab.xlim(tlim)
pylab.xticks(xticks)
pylab.ylim(0,40)
pylab.yticks([10,30])
pylab.ylabel('rate')
pylab.xlabel('time [ms]')


########################################################################
# network model and simulation parameters ##############################
########################################################################
params = {'randseed':8721,'trials':150,'N_E':1200,'N_I':300,'I_th_E':1.25,
            'I_th_I':0.78,'Q':6,'rs_stim_amp':0,'n_jobs':12,'conditions':[1,2,3]}

settings = [{'randseed':8721,'jep':3.2,'jipratio':0.75,
                'condition_stim_amps':[0.1,0.1,0.1],
                'rs_stim_amp':0.1,'rs_length':400, 
                'trials':150}]  
#######################################################################

print('PLOT MODEL ...')
ax_model = ax_label1(simpleaxis1(
    pylab.subplot2grid((nrow,ncol),(4,0),rowspan=3),labelsize,pad=pad),
                                'd',x=x_label_val, 
                                size=labelsize,y=1.08)  
# ax_model.text(0.1,.2,
#         'E/I Clustered model',ha = 'center',va ='bottom',
#         size = labelsize-2,rotation=90,weight='bold')       
pylab.box('off')
pylab.axis('off')


plot = True

for setno,setting in enumerate(settings):
    for k in list(setting.keys()):
        params[k] = setting[k]
    min_count_rate = 5.
    plot_params = {'sim_params':params,
                    'min_count_rate':round(float(min_count_rate),4), 
                        'cvtwo_win':400}
    ax_label1(simpleaxis1(
        pylab.subplot2grid((nrow,ncol),(4,1),rowspan=2),
        labelsize,pad=pad),'e',x=x_label_val, size=labelsize)            

    plot_params['timestep'] = 5
    plot_population_decoding(plot_params,plot = True,redo=redo)
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw)
    #pylab.axvline(1900,linestyle = '--',color = 'k',lw = lw)
    pylab.xlim(0,2000)
    pylab.axhline(1/6.,linestyle = '--',lw = lw,color = '0.4',zorder = -1)
    pylab.text(2000, 1/6., r'$1/6$',va= 'center',ha = 'left',size = size)
    pylab.axhline(1/3.,linestyle = '--',lw = lw,color = '0.4',zorder = -1)
    pylab.text(2000, 1/3., r'$1/3$',va= 'center',ha = 'left',size = size)
    pylab.axhline(1/2.,linestyle = '--',lw = lw,color = '0.4',zorder = -1)
    pylab.text(2000, 1/2., r'$1/2$',va= 'center',ha = 'left',size = size)
    plot_params.pop('timestep')
    pylab.yticks([0.1,0.4,0.7,1.])
    pylab.xticks([])
    #pylab.fill_betweenx([0.1,1],1500,2000,color='gray',alpha=0.5)

    # plot rates
    simpleaxis1(pylab.subplot2grid((nrow,ncol),(6,1)),labelsize,pad=pad)
    plot_rates(plot_params,plot = True,redo=redo)
    pylab.ylim(0,40)
    pylab.yticks([10,30])
    
    ax_label1(simpleaxis1(
        pylab.subplot2grid((nrow,ncol),(4,2),rowspan=2),labelsize,pad=pad),'f',x=x_label_val,size=labelsize)            
    
    plot_ffs(plot_params,plot = True,redo=redo)
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw)
    pylab.xlim(0,2000)
    pylab.yticks([-1,-0.5,0])
    pylab.ylim(-1.5,0.2)
    #pylab.fill_betweenx([-1.5,0.2],1500,2000,color='gray',alpha=0.5)
    #pylab.yticks([-1.,0])
    pylab.xticks([])
    simpleaxis1(pylab.subplot2grid((nrow,ncol),(6,2)),labelsize,pad=pad)            

    plot_cv2s(plot_params,plot = True,redo=redo)
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw)
    pylab.xlim(0,2000)
    pylab.ylim(0.4,1.3)
    pylab.yticks([0.4,0.8,1.2])
    #pylab.fill_betweenx([0.4,1.3],1500,2000,color='gray',alpha=0.5)
pylab.savefig('compound_data_fig_cv2_corrected0.eps')
pylab.close()



###############################
# plot experiment protocol ####
###############################

off_color = '0.8'
green = 'green'#'0.5'
red = 'red'#'0.3'
gray = '0.3'
fig = nice_figure()
fig.subplots_adjust(bottom  =0.01,top  =0.95)    
hsep = 2. 
vsep = 1. 
draw_hex_array((0,0),colors = [green]+5*[off_color],edgecolor=None)
draw_hex_array((hsep,0),colors = [red]+5*[off_color],show_numbers= True,
                edgecolor=None)
pylab.text(-0.7,0,'condition 1',va = 'center',ha='right',color=condition_colors[0],
            size = size_cond)
draw_hex_array((0,-vsep),colors = [green]*2+4*[off_color],edgecolor=None)
draw_hex_array((hsep,-vsep),colors = [red]+5*[off_color],edgecolor=None)
pylab.text(-0.7,-vsep,'condition 2',va = 'center',ha='right',color=condition_colors[1],
            size = size_cond)

draw_hex_array((0,-2*vsep),colors = [green]*3+3*[off_color],edgecolor=None)
draw_hex_array((hsep,-2*vsep),colors = [red]+5*[off_color],edgecolor=None)
pylab.text(-0.7,-vsep*2,'condition 3',va = 'center',ha='right',color=condition_colors[2],
            size = size_cond)

pylab.text(-2.3,-vsep+0.2,'Behaving monkey',va = 'center',ha='right',
            color='k',rotation=90,size=size_cond+2)#,weight='bold')

txth = 0.8
lw = 1.5
pylab.plot([-0.6*hsep,1.6*hsep],[0.7]*2,'k',lw = lw)
epoch_size = 14
pylab.arrow(0.4*hsep,0.57,0.3,0, width=0.004, head_width=0.03,fc ='k',lw=0.5)
pylab.text(0.48*hsep,0.45,'time',ha='center',va = 'top',size=10)
pylab.plot([-0.5*hsep]*2,[0.65,0.75],'k',lw = lw)
pylab.text(-0.5*hsep,txth,'TS',va = 'bottom',ha='center',size=epoch_size)
pylab.plot([0.0]*2,[0.65,0.75],'k',lw = lw)
pylab.text(0,txth,'PS',va = 'bottom',ha='center',size=epoch_size)
pylab.plot([hsep]*2,[0.65,0.75],'k',lw = lw)
pylab.text(hsep,txth,'RS',va = 'bottom',ha='center',size=epoch_size)
pylab.plot([hsep+0.7]*2,[0.65,0.75],gray,lw = lw)
pylab.text(hsep+0.7,txth,'MO',va = 'bottom',ha='center',color = gray, size=epoch_size)
pylab.axis('scaled')
pylab.axis('off')
pylab.subplots_adjust(left=0.1, bottom=0, right=1, top=1, wspace=0, hspace=0)    
pylab.savefig('./experiment.eps')
pylab.close()


## plot model sketch
fig = nice_figure(fig_width=1)
ax = pylab.subplot(1,1,1)
model_plot(ax, colors=['gray']+5*['gray'])
pylab.axis('off')
pylab.box('off')
pylab.subplots_adjust(left=0.1, bottom=0, right=0.95, top=1, wspace=0, hspace=0)        
pylab.savefig('./model.eps')
pylab.savefig('./model.jpg',dpi=600)
pylab.close()

######################################
# merge different plots into one #####
######################################
import pyx
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0.0, "compound_data_fig_cv2_corrected0.eps"))
c.insert(pyx.epsfile.epsfile(.35, 4.9,"experiment.eps",scale=0.3))
c.writeEPSfile("compound_data_fig_cv2_corrected1.eps")  
#inser model
c.insert(pyx.epsfile.epsfile(0, 0.0, "compound_data_fig_cv2_corrected1.eps"))
# c.insert(pyx.epsfile.epsfile(.8, 1.2,"model.eps",scale=0.3))
i = pyx.bitmap.jpegimage("model.jpg")
c.insert(pyx.bitmap.bitmap(.7, 1.2,i, compressmode=None,
                           width=3.7*1.2, height=2.5*1.2))
c.writePDFfile('fig5.pdf')
# remove intemittent files
os.remove('compound_data_fig_cv2_corrected0.eps')
os.remove('experiment.eps')
os.remove('compound_data_fig_cv2_corrected1.eps')
os.remove('model.eps')
os.remove('model.jpg')

