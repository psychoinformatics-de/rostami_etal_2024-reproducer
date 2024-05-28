import sys;sys.path.append('src')
import pylab
from simulate_experiment import get_simulated_data
import spiketools
from GeneralHelper import (
    find, nice_figure, ax_label1, 
    simpleaxis, ax_label_title
    )
import scipy.stats as ss
from reaction_times_functions import (
    reaction_time_plot, get_reaction_time_analysis
)

########################################################################
# network model and simulation parameters ##############################
########################################################################
sim_params = {'randseed':7745,'trials':2000,'N_E':1200,'N_I':300,
              'I_th_E':1.25,'I_th_I':0.78,'Q':6,'rs_stim_amp':0,
              'n_jobs':12,'conditions':[1,2,3], 'jep':3.2,
                'jipratio':0.75,'condition_stim_amps':[0.1,0.1,0.1],
                'rs_stim_amp':0.1,'rs_length':400}
params = {'sim_params':sim_params}
# prams for reaction time analysis
tau = 50.
threshold_per_condition =False
integrate_from_go = False
min_count_rate = 7.5
align_ts = False
########################################################################

x_label_val=-0.3      
abc_fontsize = 10*0.7              
fig = nice_figure(fig_width=0.489, ratio = 1.3)
nrows = 2
ncols = 2
gs = pylab.GridSpec(nrows,ncols,top=0.85,bottom=0.15,hspace = 0.4,
                    wspace = 0.5,left = 0.15,right = 0.85,height_ratios = [2,1])
subplotspec = gs.new_subplotspec((1,1), colspan=1,rowspan=1)
ax2 = ax_label1(simpleaxis(pylab.subplot(subplotspec)),
                         'c',x=x_label_val, size=abc_fontsize)
ax2.set_title('  Behaving monkey',size=8, loc='center')
labelsize=5
cond_colors = ['navy','royalblue','lightskyblue']
reaction_time_plot(b'joe', condition_colors = cond_colors)
pylab.xlabel('reaction time [ms]')
pylab.ylabel('p.d.f')
pylab.axvline(1500,linestyle = '-',color = 'k',lw = 0.5)
pylab.ylim(0,0.015)    
pylab.yticks([0,0.004,0.008,0.012])
pylab.legend(frameon = False,fontsize = 6,loc = 'upper right', 
             bbox_to_anchor=(1.45, 1.1))
pylab.xticks([1500,1700,1900])
ax2.set_xticklabels(['RS','200','400'])
condition_alpha = 1.
            
condition_colors = [[0,0,0,condition_alpha],
                    [0.4,0.4,0.4,condition_alpha],
                    [0.6,0.6,0.6,condition_alpha]]
tlim = [-500,2000]


# get the reaction time analysis
result = get_reaction_time_analysis(params,tlim  =tlim,redo = False,
                        tau  =tau,integrate_from_go = integrate_from_go,
                        normalise_across_clusters=True,
                        threshold_per_condition = threshold_per_condition,
                        fname='fig6_reaction_time_analysis')

rts = result['rts']
conditions = result['conditions']
directions = result['directions']
predictions = result['predictions']
correct= directions == predictions
correct_inds = find(correct)
incorrect_inds = find(correct==False)
subplotspec = gs.new_subplotspec((0,0), colspan=2,rowspan=1)
ax1 = ax_label1(simpleaxis(
    pylab.subplot(subplotspec)),'a',x=x_label_val/3, size=abc_fontsize)
ax1.text(-500,1500,   
               'Example trial of motor cortical attractor model', size=8)
#pylab.suptitle()

plot_trial = find(correct*(conditions==3))[7]#[6]#[8]
pylab.xticks([-500,0,1000])
pylab.gca().set_xticklabels(['0', '500','1500'])

data  = get_simulated_data(params['sim_params'],
                datafile = 'fig6_simulated_data')

cut_window = [-500,2000]
trial_starts = data['trial_starts']
spiketimes = spiketools.cut_spiketimes(
    data['spiketimes'],
    tlim = pylab.array(cut_window)+trial_starts[plot_trial])
spiketimes[0] -= trial_starts[plot_trial]
pylab.plot(spiketimes[0],spiketimes[1],'.',ms =0.5,color = '0.5')
pylab.show()
pylab.xlim(cut_window)
Q = params['sim_params']['Q']
N_E = params['sim_params']['N_E']
cluster_size = N_E/Q
direction_clusters =  data['direction_clusters']
integrals = result['integrals']
time = result['time']
threshold = result['condition_thresholds'][conditions[plot_trial]]
pylab.axvline(1000,linestyle = '--',color ='k',lw = 0.5)
direction_clusters = pylab.array(direction_clusters).flatten()
for cluster in range(6):
    pylab.text(2000,(cluster+0.5)*cluster_size,str(cluster+1),
                va = 'center',ha = 'left',weight='bold')
    direction = find(direction_clusters == cluster)[0]
    pylab.plot(
        time,
        cluster*cluster_size +integrals[direction,plot_trial]*cluster_size*0.8,
        color = 'k')
    pylab.plot([1000,2000],
               [cluster*cluster_size +threshold*cluster_size*0.8]*2,
               '--k',lw =0.5)
    if (direction+1) == directions[plot_trial]:
        try:
            crossing = find(
                (integrals[direction,plot_trial]>threshold)*(time>1000))[0]
            print(crossing)
            pylab.plot(
                time[crossing],
                cluster*cluster_size +threshold*cluster_size*0.8,
                'ok',ms = 4)
        except:
            print('no crossing')
        
pylab.xlim(time.min(),time.max())
pylab.ylim(0,N_E)
pylab.ylabel('unit')
pylab.xlabel('time [ms]')
pylab.text(-50,1250,'PS')
pylab.text(950,1250,'RS')


subplotspec = gs.new_subplotspec((1,0), colspan=1,rowspan=1)
ax2 = ax_label1(simpleaxis(
    pylab.subplot(subplotspec)),'b',x=x_label_val, size=abc_fontsize)
ax2.set_title('Attractor model', size=8)
for condition in [1,2,3]:
    
    rt = rts[(conditions == condition)*correct]
    mask = (rts > 100)*(rts < 400)
    bins = pylab.linspace(0,500,15)
    pylab.hist(rt,bins,histtype = 'step',
               facecolor = cond_colors[condition-1],
                density = True,edgecolor  = cond_colors[condition-1],
                label = 'condtion '+str(condition))
    pylab.xlim(1400,2000)

min_len = min(len(rts[(conditions == 2)*correct]),
              len(rts[(conditions == 3)*correct]))

pylab.xlim(-100,500)
pylab.xticks([0,200,400])
pylab.gca().set_xticklabels(['RS','200','400'])
pylab.ylim(0,0.015)
pylab.yticks([0,0.004,0.008,0.012])                            
pylab.ylabel('p.d.f')
pylab.xlabel('reaction time [ms]')
pylab.axvline(0,linestyle = '-',color = 'k',lw = 0.5)             
# save figure
pylab.savefig('fig6.pdf')
pylab.show()



