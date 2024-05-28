import sys;sys.path.append('src')
from matplotlib import pylab
import pandas as pd
import numpy as np
import analyse_experiment as analyses
import joe_and_lili
from scipy.stats import wilcoxon
import pickle as pickle
from GeneralHelper import nice_figure, ax_label1, simpleaxis1

path = '../data/preprocessed_and_simulated_data/'

def do_plot(extra_filters = [],min_count_rate = 5,min_trials  =10,
            tlim = [0,2000],alignment ='TS',ff_ax = None,
            count_dist_ax = None,
            condition_colors = ['0','0.3','0.6'],
            ff_test_interval = None,ff_test_point = 1000.,
            ff_test_ys = [0.1,2.],textsize=6,lw=3,lw_line=0.5,
            mean_matching_ff = False):
    toc = joe_and_lili.get_toc(extra_filters = extra_filters)
    gns = pylab.unique(toc['global_neuron'])
    # find the gns and directions where criteria are met across conditions
    count_rate_block = pylab.zeros((len(gns),3,6))
    trial_count_block = pylab.zeros((len(gns),3,6))
    colors_hbar = ['r', 'g', 'y']
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



    if count_dist_ax is not None:
        try:
            (spike_counts,tspike_counts,spike_counts_conditions,
             spike_counts_gns,spike_counts_directions) = pd.read_pickle(
                 path+'suppl_fig2_spike_counts_'+alignment)
        except:
            print('calculating spike counts ...')
            spike_counts_gns = []
            spike_counts_conditions = [] 
            spike_counts_directions = []
            spike_counts = []
            for i,gn in enumerate(gns):
               for j,condition in enumerate([1,2,3]):
                    for k,direction in enumerate([1,2,3,4,5,6]):
                        if good_directions[i,k]:
                            spike_count,tspike_count = analyses.get_spike_counts(
                                gn, condition, direction,alignment = alignment,
                                tlim  =tlim)
                            spike_counts.append(spike_count)
                            spike_counts_gns.append(gn)
                            spike_counts_conditions.append(condition)
                            spike_counts_directions.append(direction)

            spike_counts_conditions = pylab.array(spike_counts_conditions)
            pickle.dump((spike_counts,tspike_count,spike_counts_conditions,
                         spike_counts_gns,spike_counts_directions),
                        open(path+'suppl_fig2_spike_counts_'+alignment,'wb'),
                        protocol = 2)

        for cnt, spk in enumerate(spike_counts):
            if cnt==0:
                means= np.mean(spk,0)
                vars = np.var(spk,0)
            else:
                means = np.vstack((means, np.mean(spk,0)))
                vars = np.vstack((vars, np.var(spk,0)))
        slices = [200,800,1400]
        max_count_mask_per_cond = []
        for (condition,color) in zip([1,2,3],condition_colors):
            means_sel = means[spike_counts_conditions==condition]
            vars_sel = vars[spike_counts_conditions==condition]
            greatest_dist = np.min(means_sel,1)
            max_count = np.max(greatest_dist)
            max_count_mask_per_cond.append(means_sel<max_count)
            row= condition -1
            
            for col in [0,1,2]:
                pylab.sca(count_dist_ax[row*3+col])

                if mean_matching_ff:
                    mask = means_sel[:,slices[col]]<max_count
                    pylab.hist(means_sel[mask,slices[col]],
                               bins=np.arange(0,50,1),color=color)
                    if condition==1 and col==1:
                        pylab.gca().set_title('Mean Matched Count Distribution')
                else:
                    pylab.hist(means_sel[:,slices[col]],
                               bins=np.arange(0,50,1),color=color)
                    if condition==1 and col==1:
                        pylab.gca().set_title('Count Distribution')
                if condition==1:
                    pylab.fill_betweenx(
                        [60,64],10,40,color=colors_hbar[col],alpha=0.4)
                pylab.xlim(0,50)      
                pylab.ylim(0,60)     

    if ff_ax is not None:
        pylab.sca(ff_ax)
        if mean_matching_ff:
            pylab.gca().set_title('Mean Matched Fano Factor')
        else:
            pylab.gca().set_title('Fano Factor')
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
                                gn, condition, direction,
                                alignment = alignment,tlim  =tlim)
                            ffs.append(ff)
                            ff_gns.append(gn)
                            ff_conditions.append(condition)
                            ff_directions.append(direction)
            ffs = pylab.array(ffs)

            ff_conditions = pylab.array(ff_conditions)
            pickle.dump((ffs,tff,ff_conditions,ff_gns,ff_directions),
                        open(path+'experiment_'+monkey+'_ff_file_'+alignment,
                             'wb'),protocol = 2)
        for (condition,color) in zip([1,2,3],condition_colors):
            if mean_matching_ff:
                ffs_cond = ffs[ff_conditions==condition]
                avg_ff=[]
                for i,j in enumerate(max_count_mask_per_cond[condition-1].T):
                    avg_ff.append(np.nanmean(ffs_cond[j,i]))
            else:
                avg_ff = pylab.nanmean(ffs[ff_conditions==condition],axis=0)
            offset = pylab.nanmean(ffs[ff_conditions==condition],axis=0)[0]
            pylab.plot(tff, avg_ff-offset,
                       color = color,label = 'condition '+str(condition))
            pylab.fill_betweenx([-.7,-0.68],
                                slices[condition-1],slices[condition-1]+400,
                                color=colors_hbar[condition-1],alpha=0.4)
        if ff_test_interval is not None:
            
            for ntest,test_conditions in enumerate([[1,2],[2,3]]):
                interval_mask = (
                    tff>ff_test_interval[0]) * (tff<ff_test_interval[1])
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
                pylab.plot(test_time,sigplot,lw = lw)
        if ff_test_point is not None:
            for ntest,test_conditions in enumerate([[1,2],[2,3]]):
                test_ind = pylab.argmin(pylab.absolute(tff-ff_test_point))
                test_time = tff[test_ind]
                test_vals = ffs[:,test_ind]
                test_vals1 = test_vals[ff_conditions == test_conditions[0]]
                test_vals2 = test_vals[ff_conditions == test_conditions[1]]
                s,p = wilcoxon(test_vals1[:],test_vals2[:])
                bottom_val = pylab.nanmean(
                    test_vals1) - pylab.nanmean(
                        ffs[ff_conditions==test_conditions[0]],axis=0)[0]
                top_val = pylab.nanmean(
                    test_vals2) - pylab.nanmean(
                        ffs[ff_conditions==test_conditions[1]],axis=0)[0]
                center = 0.5 * (bottom_val+top_val) 
                pylab.plot([test_time]*2,[bottom_val+0.02,top_val-0.02],'-_k',
                           lw =lw_line,ms = 2.)
                pylab.text(test_time-10, center, '*',va = 'top',ha ='right')



###################################
###########MODEL#################
condition_colors = ['0','0.3','0.6']
condition_colors = ['navy','royalblue','lightskyblue']
    

if __name__ == '__main__':
    abc_fontsize = 10
    labelsize = 8
    labelsize1 = 6
    ticksize =2.
    size = 7
    scale=1.5
    lw= 0.3
    rcparams = {'axes.labelsize': size*scale,
                'xtick.major.size': ticksize,
                'ytick.major.size': ticksize,            
                'lines.linewidth':0.5,
                'axes.linewidth':0.2}

    fig = nice_figure(fig_width= 1.,ratio  =.7,rcparams = rcparams)
    fig.subplots_adjust(hspace = .5,wspace = 0.9,bottom  =0.14,top  =0.9)
    tlim = [0,2000]
    xticks = [0,500,1000,1500,2000]
    nrow,ncol = 7, 6
    pad=.3
    x_label_val=-0.6
    size_cond = 12

    condition_colors_exp = ['navy','royalblue','lightskyblue']
    for monkey in ['joe']:
        
        extra_filters = [('monkey','=',str.encode(monkey))]
        ff_ax = ax_label1(simpleaxis1(
            pylab.subplot2grid((nrow,ncol),(0,0),rowspan=3, colspan=3),
            labelsize,pad=pad),'a',x=x_label_val/5,size=abc_fontsize)

        mean_matched_ff_ax = ax_label1(simpleaxis1(
            pylab.subplot2grid((nrow,ncol),(0,3),rowspan=3, colspan=3),
            labelsize,pad=pad),'b',x=x_label_val/5,size=abc_fontsize)
        count_dist_ax_list=[]
        count_dist_ax_mm_list=[]
        for i in range(9):
            row = int(i/3) + 4
            col = i%3
            label_axis, label_axis_mm ='', ''
            if row == 4 and col ==0:
                label_axis = 'c'
                label_axis_mm = 'd'                
            count_dist_ax = ax_label1(simpleaxis1(
                pylab.subplot2grid((nrow,ncol),(row,col),rowspan=1, colspan=1),
                labelsize,pad=pad),label_axis,x=x_label_val,size=abc_fontsize)
            count_dist_ax_list.append(count_dist_ax)
            count_dist_ax_mm = ax_label1(simpleaxis1(
                pylab.subplot2grid((nrow,ncol),(row,col+3),rowspan=1, colspan=1),
                labelsize,pad=pad),label_axis_mm,x=x_label_val,size=abc_fontsize)
            count_dist_ax_mm_list.append(count_dist_ax_mm) 


        do_plot(extra_filters = extra_filters,ff_ax = ff_ax,
                count_dist_ax=count_dist_ax_list, textsize=size,
                    lw=1,lw_line=0.3, condition_colors=condition_colors_exp,
                    mean_matching_ff = False)
        do_plot(extra_filters = extra_filters,ff_ax = mean_matched_ff_ax,
                count_dist_ax=count_dist_ax_mm_list,textsize=size,lw=1,lw_line=0.3, 
                condition_colors=condition_colors_exp,mean_matching_ff = True)


    pylab.sca(ff_ax)
    pylab.xlim(tlim)
    pylab.xticks([])
    pylab.ylabel(r'$\Delta$FF',rotation=90)
    pylab.ylim(-0.7,0.1)
    pylab.yticks([-0.5,0])
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.text(500, pylab.ylim()[0]-0.1,'PS',va = 'bottom',ha = 'center',size = labelsize)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.text(1500, pylab.ylim()[0]-0.1,'RS',va = 'bottom',ha = 'center',size = labelsize)


    pylab.sca(mean_matched_ff_ax)
    pylab.xlim(tlim)
    pylab.xticks([])
    pylab.ylim(-0.7,0.1)
    pylab.yticks([-0.5,0])
    pylab.legend(frameon = False,fontsize = labelsize,
                 loc = 'upper center',bbox_to_anchor=(1., 1.1))
    
    pylab.axvline(500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.text(500, pylab.ylim()[0]-0.1,'PS',va = 'bottom',ha = 'center',size = labelsize)
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = lw/2)
    pylab.text(1500, pylab.ylim()[0]-0.1,'RS',va = 'bottom',ha = 'center',size = labelsize)
  
    pylab.sca(count_dist_ax_list[6])
    pylab.xlabel('Spike Count')#
    pylab.ylabel('#')
    pylab.savefig('suppl_fig2.pdf')
    pylab.show()

