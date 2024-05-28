import sys;sys.path.append('src')
import pylab
# Local modules (not installed packages)
import analyse_model
import defaultSimulate as default
from GeneralHelper import (
    nice_figure, make_color_list, 
    ax_label1, ax_label_title,
    simpleaxis, find)


def do_plot(params,file_name,axes=None,redo = False,plot = True,markersize = 0.5,
            spikealpha = 1,box_color = 'k',cmap = 'jet',lw = 0.8,show_clusters = 4,
            legend = False,current_limits = [-16,18],
            voltage_limits = [-22,42],ylabel = False,
            rate_ticks = [],V_ticks = [],I_ticks = [], rcparams = {}):
    """plot the results of the simulation and analysis of the model"""
    # get the results of the simulation
    pylab.rcParams.update(rcparams)
    result = analyse_model.get_simulate_and_analyse_fig3(params, file_name)
    spiketimes = result['spiketimes']
    if plot:
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[0]
        pylab.sca(ax)
        pylab.xlabel('time [ms]')
        # draw a box around the focus cluster/interval
        bottom = min(result['focus_cluster_inds'])
        top = max(result['focus_cluster_inds'])
        left = params['focus_interval'][0]
        right = params['focus_interval'][1]
        
        x_margin = 2.
        y_margin = 2.
        pylab.fill([left,right,right,left,left],[bottom,bottom,top,top,bottom],
                   color = "r",lw = 0,alpha=0.1)
        pylab.plot(spiketimes[0],spiketimes[1],'.k',markersize = markersize,
                   alpha = spikealpha)
        pylab.xlim(0,params['simtime'])
        pylab.xticks(list(range(0,int(params['simtime'])+50,200)),
                     size = rcparams['xtick.labelsize'])
        pylab.plot([left-x_margin,right+x_margin,right+x_margin,left-x_margin,
                    left-x_margin],[bottom-y_margin,bottom-y_margin,
                                    top+y_margin,top+y_margin,
                                    bottom-y_margin],'--',
                    color = box_color,lw = 1.5)
        
        # show only the clusters below and above the focus cluster

        cluster_size = params['N_E']/params['Q']
        for i in range(params['Q']):
            pylab.axhline(i*cluster_size,linestyle = '-',color = 'k',alpha = 0.8)
        pylab.ylim((params['focus_cluster']-(show_clusters))*cluster_size,
                   (params['focus_cluster']+show_clusters+1)*cluster_size)
        pylab.yticks([])
        if ylabel:
            pylab.ylabel('unit')
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[1]
        pylab.sca(ax)
        pylab.xlabel('time [ms]')
        cluster_rates = result['cluster_rates']
        t_rate = result['t_rate']
        if ylabel:
            pylab.ylabel(r'rate [1/s]')
        colors = make_color_list(2*show_clusters+1,cmap = cmap)
        for i,q in enumerate(range(params['focus_cluster']-show_clusters,
                                   params['focus_cluster']+show_clusters+1)):
            pylab.plot(t_rate,cluster_rates[q],lw= lw,color = colors[i])
            if q == params['focus_cluster']:
                # find bounding box of that line in the focus interval
                focus_piece = find((t_rate>=left)*(t_rate<=right))
                bottom = cluster_rates[q][focus_piece].min()
                top = cluster_rates[q][focus_piece].max()
                pylab.plot(t_rate[focus_piece],cluster_rates[q][focus_piece],'--r',lw=1.5)
        
        pylab.xlim(0,params['simtime'])
        pylab.xticks(list(range(0,int(params['simtime'])+50,200)),
                size = rcparams['xtick.labelsize'])
        pylab.yticks(rate_ticks, size = rcparams['ytick.labelsize'])
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[2]
        pylab.sca(ax)
        bottom = voltage_limits[0]
        top = voltage_limits[1]
        pylab.fill([left,right,right,left,left],[bottom,bottom,top,top,bottom],
                   color = "r",lw = 0, alpha=0.1)        
        pylab.xticks(list(range(left,right+50,100)), 
                     size = rcparams['xtick.labelsize'])
        pylab.xlabel('time [ms]')
        ex_current = result['ex_current'][0]+result['Ixe']
        inh_current = result['inh_current'][0]
        time = result['current_times']
        focus_piece = find((time>=left)*(time<=right))        
        pylab.plot(time[focus_piece],ex_current[focus_piece],color = '0.4',
                   label = r'$\mathrm{I_{E}} + \mathrm{I_{x}}$')
        pylab.plot(time[focus_piece],inh_current[focus_piece],color = '0.65',
                   label = r'$\mathrm{I_{I}}$')
        pylab.plot(
            time[focus_piece],ex_current[focus_piece]+inh_current[focus_piece],
            color = 'k',label = r'$\mathrm{I_{tot}}$')
        pylab.axhline(0,linestyle ='--',color = '0.7')
        if params['jipfactor'] == 0.:
            current_limits = [current_limits[0]+3, current_limits[1]-3]
            I_ticks = [I_ticks[0]+4, I_ticks[1], I_ticks[2]-4]
        pylab.ylim(current_limits)
        pylab.xlim(left,right)
        pylab.yticks(I_ticks, size = rcparams['ytick.labelsize'])
        if ylabel:
            pylab.ylabel(r'$\mathrm{I_{syn}}$ [pA]')
        if legend:
            pylab.legend(loc = 'upper center',frameon = False,
                         fontsize= 8,ncol = 3,handlelength = 1.5,
                         columnspacing = 1.,handletextpad = 0.5,
                         borderaxespad = 0.,borderpad = 0.)
        
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[3]
        pylab.sca(ax)
        bottom = voltage_limits[0]
        top = voltage_limits[1]
        pylab.fill([left,right,right,left,left],[bottom,bottom,top,top,bottom],
                   color = "r",lw = 0, alpha=0.1)        
        pylab.xlabel('time [ms]')
        if ylabel:
            pylab.ylabel('$V_{m}$ [mV]')
        V_m = result['V_m'][0]
        pylab.plot(time[focus_piece],V_m[focus_piece],'k')
        V_th_E = params.get('V_th_E',default.V_th_E)
        pylab.axhline(V_th_E,linestyle = '--',color = 'k')
        spikeheight = 20.
        for spike in result['focus_spikes']:
            pylab.plot([spike]*2,[V_th_E,V_th_E+spikeheight],'k')
        pylab.ylim(voltage_limits)
        pylab.xlim(left,right)
        pylab.rcParams.update(rcparams)
        pylab.xticks(list(range(left,right+50,100)),
                     size = rcparams['xtick.labelsize'])
        pylab.yticks(V_ticks,
                     size = rcparams['ytick.labelsize'])
        

########################################################################
# network model and simulation parameters ##############################
########################################################################
params = {'simtime':1000.,'n_jobs':12,'Q':50,'rate_kernel':50,'N_E':4000}
settings = [{'warmup':0, 'jipfactor':0.,'jep':3.7,'randseed':3,
             'focus_cluster':8,'focus_interval':[200,600],'focus_unit':6}, 
            {'warmup':0, 'jipfactor':0.75,'jep':8.,'randseed':5,
             'focus_cluster':15,'focus_interval':[700,1000],'focus_unit':12}]  
file_name = 'fig3_simulated_data'
########################################################################
rc_params = {'axes.labelsize': 10,
             'lines.linewidth':2,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8}
fig = nice_figure(ratio = .9, rcparams = rc_params)
abc_size = 10
ncols =2
nrows = 7
hspace = 0.6
gs = pylab.GridSpec(nrows,ncols,top=0.95,bottom=0.08,
                    hspace = 0.01,left = 0.1,right = 0.9,
                    height_ratios = [1.8,hspace,0.7,hspace,1.2,hspace,.7])
x_label_val = -0.17
subplotspec = gs.new_subplotspec((0,0), colspan=1,rowspan=1)
ax1 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax1, 'a', x=x_label_val, size=abc_size)
ax_label_title(ax1, 'E clustered network', size=abc_size+4)
subplotspec = gs.new_subplotspec((0,1), colspan=1,rowspan=1)
ax2 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax2, 'b', x=x_label_val, size=abc_size)
ax_label_title(ax2, 'E/I clustered network', size=abc_size+4)

subplotspec = gs.new_subplotspec((2,0), colspan=1,rowspan=1)
ax3 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax3, 'c', x=x_label_val, size=abc_size)

subplotspec = gs.new_subplotspec((2,1), colspan=1,rowspan=1)
ax4 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax4, 'd',x=x_label_val, size=abc_size)

subplotspec = gs.new_subplotspec((4,0), colspan=1,rowspan=1)
ax5 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax5, 'e', x=x_label_val, size=abc_size)

subplotspec = gs.new_subplotspec((4,1), colspan=1,rowspan=1)
ax6 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax6, 'f',x=x_label_val, size=abc_size)

subplotspec = gs.new_subplotspec((6,0), colspan=1,rowspan=1)
ax7 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax7, 'g',x=x_label_val, size=abc_size)

subplotspec = gs.new_subplotspec((6,1), colspan=1,rowspan=1)
ax8 =simpleaxis(pylab.subplot(subplotspec))
ax_label1(ax8, 'h',x=x_label_val, size=abc_size)

axes = [[ax1,ax3,ax5,ax7],[ax2,ax4,ax6,ax8]]
for setno,setting in enumerate(settings):
    for k in list(setting.keys()):
        params[k] = setting[k]
    if setno == 0:
        legend = True
        ylabel = True
        rate_ticks = [0,30,60,90]
    else:
        legend = False
        ylabel = False
        rate_ticks = [0,10,20,30]
    do_plot(params,file_name=file_name,redo = False,
            axes = axes[setno],box_color = 'r',
            cmap = 'Greys',legend = legend,ylabel = ylabel,
            rate_ticks = rate_ticks,
            V_ticks = [-20,0,20,40],I_ticks = [-15,0,15],
            rcparams=rc_params)
# save the figure
pylab.savefig('fig3.pdf')

