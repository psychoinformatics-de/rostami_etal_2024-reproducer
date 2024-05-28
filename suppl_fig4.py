import sys;sys.path.append('src/')
import ClusterModelNEST
import pylab
import spiketools
from GeneralHelper import (Organiser,
    nice_figure, simpleaxis,
    ax_label, psth_plot)
import defaultSimulate as default


datafile = 'suppl_fig4_balanced_state'

def simulate_and_analyse(params):
    pylab.seed(None)
    N_E = params.get('N_E',default.N_E)
    EI_Network = ClusterModelNEST.ClusteredNetwork(default, params)
    # Creates object which creates the EI clustered network in NEST
    
    result = EI_Network.get_simulation() 
    if params['record_voltage']:
        voltage_results = EI_Network.get_voltage_recordings() 
        result.update(voltage_results)
    spiketimes = result['spiketimes']
    results = {}
    if params.get('calc_synchrony',False):
        print('calculating synchrony')
        counts,time = spiketools.spiketimes_to_binary(
            spiketimes,tlim = params['synchrony_tlim'],
            dt = params.get('count_bin',20.))
        synchrony = (counts.mean(axis=0).var()/counts.var(axis=1).mean())**0.5
        
        results['synchrony']=synchrony

    results['e_rate'] = result['e_rate']
    results['i_rate'] = result['i_rate']
    if params.get('return_spiketimes',True):
        results['spiketimes'] = spiketimes
    try:
        results['times'] = result['times']
        results['V_m'] = result['V_m']
        results['input_currents_ex'] = result['I_syn_ex']
        results['input_currents_in'] = result['I_syn_in']
    except:
        pass 




    spikelist = spiketools.spiketimes_to_list(spiketimes)
    cv2s = []

    for unit,spikes in enumerate(spikelist):
        intervals = pylab.diff(pylab.sort(spikes))
        if len(intervals)>1:
            mean_interval = intervals.mean()
            ot = params['simtime']/mean_interval
            cv2 = intervals.var(ddof = 1)/intervals.mean()**2
            if round(cv2,2) >0 and ot<30:
                
                corrected_cv2 = spiketools.unbiased_cv2(cv2,ot,2)
                if pylab.isnan(corrected_cv2):
                    print('corrected_cv2 is nan')
                else:
                    cv2  =corrected_cv2[0]

        else:
            cv2 = pylab.nan
        
        cv2s.append(cv2)



    cv2 = pylab.nanmean(cv2s)
    results['cv2'] = cv2
    results['I_xE'] = result['params']['I_xE']

    return results

def do_g_plot(params,redo = False,g_range = pylab.linspace(0.5,1.5,26),
              reps  =5,lw = 1.5,final_g=1.2,n_jobs = 1):
    # original_grange =pylab.linspace(0.5,1.5,26)
    g_range = pylab.array([ 0.5 ,  0.54,  0.58,  0.62,  0.66,  0.7 , 
                           0.74,  0.78,  0.82,
        0.86,  0.9 ,  0.94,  0.98,  1.02,  1.06,  1.1 ,  1.14,  1.18,
        1.22,  1.26,  1.3 ,  1.34,  1.38,  1.42,  1.46,  1.5 ])
    cvs = []
    syncs = []
    e_rates = []
   
    for g in g_range:
        params['ge'] = round(g,4)
        ORG = Organiser(params, datafile, redo=redo, reps=reps, n_jobs=n_jobs)
        results = ORG.check_and_execute(simulate_and_analyse)
        cvs.append([result['cv2'] for result in results])
        syncs.append([result['synchrony'] for result in results])
        e_rates.append([result['e_rate'] for result in results])
    cvs = pylab.array(cvs).mean(axis=1)
    syncs = pylab.array(syncs).mean(axis=1)
    e_rates = pylab.array(e_rates).mean(
        axis=1)*params.get('t_ref',default.t_ref)/1000.
    pylab.plot(g_range,cvs,color = '0.',label = r'$\mathrm{CV^2}$',lw = lw)
    pylab.plot(g_range,syncs,color = '0.4',label = r'$\mathrm{\chi}$',lw = lw)
    pylab.plot(g_range,e_rates,color = '0.65',
               label = r'$\mathrm{\nu_{E} [1/\tau_{r}]}$',lw = lw)
    pylab.legend(loc = 'upper center',frameon = False,
                 fontsize= 6,ncol = 3,handlelength = 1.5,
                 columnspacing = 1.,handletextpad = 0.5,
                 borderaxespad = 0.,borderpad = 0.)
    pylab.xlim(g_range.min(),g_range.max())
    ymax = max(cvs.max(),syncs.max(),e_rates.max())
    pylab.ylim(0,ymax*1.15)
    pylab.yticks([0,0.3,0.6,0.9])
    final_ind = pylab.argmin(pylab.absolute(g_range-final_g))
    return cvs[final_ind],syncs[final_ind]
def do_s_plot(params,redo = False,
              s_range = pylab.linspace(0.5,2.,31),reps  =5):
    cvs = []
    syncs = []
    e_rates = []
    for s in s_range:
        params['s'] = s
        ORG = Organiser(params, datafile, redo=redo, reps=reps)
        results = ORG.check_and_execute(simulate_and_analyse)
        cvs.append([result['cv2'] for result in results])
        syncs.append([result['synchrony'] for result in results])
        e_rates.append([result['e_rate'] for result in results])
    cvs = pylab.array(cvs).mean(axis=1)
    syncs = pylab.array(syncs)
    e_rates = pylab.array(e_rates)
    pylab.plot(s_range,cvs)
    pylab.plot(s_range,syncs)



def do_plot(paramsm,redo  =False):
    ORG = Organiser(params, datafile, redo=redo)
    results = ORG.check_and_execute(simulate_and_analyse)
    spiketimes = results['spiketimes']
    pylab.plot(spiketimes[0],spiketimes[1],'.k',markersize=1,alpha = 0.5)



if __name__ == '__main__':
    target_rates = [(3,5)]
    for target_rate in target_rates:
        final_s = 1.
        final_ge = 1.2
        reps = 1
        params = {'n_jobs':12,'warmup':200,'simtime':10000,
                  'record_voltage':False,'ge':1.2,
                  'return_spiketimes':False,'calc_synchrony':True,
                  'pool':False,'synchrony_tlim':[0,5000]}

        if target_rate == (3,5):
            params['I_th_E']=2.13 # 3.26
            params['I_th_I']=1.24 # 1.49
        rc_params = {'axes.labelsize': 10,
                    'lines.linewidth':2,
                    'xtick.labelsize': 8,
                    'ytick.labelsize': 8}
        fig = nice_figure(ratio = .9, rcparams = rc_params)
        ncols = 2
        nrows = 5
        gs = pylab.GridSpec(nrows,ncols,hspace =0.5,
                            top=0.95,bottom=0.06,
                            height_ratios = [1.,3,0.5,1.2,0.8])

        # plot sync,cv,rate vs ge
        subplotspec = gs.new_subplotspec((0,0), colspan=1,rowspan=1)
        ax1 =simpleaxis(pylab.subplot(subplotspec))
        ax_label(ax1,'a', weigth='bold')
        
        
        params['s'] = final_s
        final_cv,final_sync = do_g_plot(
            params,reps = reps,final_g = final_ge,n_jobs = 1)
        pylab.xlabel('$g$')
        params['ge'] = final_ge
        

        # plot interval distribution of excitatory units

        subplotspec = gs.new_subplotspec((0,1), colspan=1,rowspan=1)
        ax3 =simpleaxis(pylab.subplot(subplotspec))
        ax_label(ax3,'b', weigth='bold')
        params['return_spiketimes'] = True
        params['calc_synchrony'] =False
        params['record_voltage'] = True
        params['record_from']=  10
        params['randseed'] = 1
        params['simtime'] = 1000
        params['V_th_E'] = 20.
        params['V_th_I'] = 20.
        ORG = Organiser(params, datafile+'_voltage', redo=False)
        results = ORG.check_and_execute(simulate_and_analyse)
        spiketimes = results['spiketimes']
        N_E = params.get('N_E',default.N_E)
        spikelist = spiketools.spiketimes_to_list(spiketimes)[:N_E]
        intervals = []
        for sl in spikelist:
            intervals += pylab.diff(sl).tolist()
        bins = pylab.linspace(0,max(intervals),100)
        pylab.hist(intervals,bins, histtype='step',  
                   fill=True, facecolor = '0.8', edgecolor='0.')
        xmax = pylab.xlim()[1]
        ymax = pylab.ylim()[1]
        pylab.text(
            0.8*xmax,0.8*ymax,r'$\mathrm{CV^{2} = }'+str(round(final_cv,2))+'$',
            ha = 'right',va = 'top',size = 7)
        pylab.text(
            0.8*xmax,0.65*ymax,r'$\mathrm{\chi =}'+str(round(final_sync,2))+'$',
            ha = 'right',va = 'top',size = 7)
        pylab.yticks([])
        pylab.ylabel(r'count')
        pylab.xlabel(r'ISI [ms]')
        pylab.xlim(0,1000)
        # raster plot
        subplotspec = gs.new_subplotspec((1,0), colspan=2,rowspan=1)
        ax4 =simpleaxis(pylab.subplot(subplotspec))
        ax_label(ax4,'c', weigth='bold')

        pylab.plot(spiketimes[0],spiketimes[1],'.k',markersize = 1,alpha = 0.5)
        pylab.xlabel(r'time [ms]')
        pylab.ylabel(r'unit')
        pylab.xlim(0,1000)
        # psth
        subplotspec = gs.new_subplotspec((2,0), colspan=2,rowspan=1)
        ax5 =simpleaxis(pylab.subplot(subplotspec))
        ax_label(ax5,'d', weigth='bold')
        psth_plot(spiketimes[:,spiketimes[1]<N_E],lw =1.,binsize = 10.)
        pylab.xlabel(r'time [ms]')
        pylab.ylabel(r'$\mathrm{\nu_{E}}$')
        if target_rate == (3,5):
            pylab.ylim(0,10)
            pylab.yticks([0,3,6,9])
        else:
            pylab.ylim(0,15)
            pylab.yticks([0,5,10,15])
        pylab.xlim(0,1000)

        # plot input currents
        # select a unit with high rate
        counts = [len(s) for s in spikelist[:params['record_from']]]
        unit = pylab.argmax(counts)
        subplotspec = gs.new_subplotspec((3,0), colspan=2,rowspan=1)
        ax6 =simpleaxis(pylab.subplot(subplotspec))
        ax_label(ax6,'e', weigth='bold')
        ex_current = results['input_currents_ex'][unit]
        in_current = results['input_currents_in'][unit]
        times = results['times']
        Ie = results['I_xE']
        pylab.plot(times,ex_current+Ie,color = '0.4',label = r'$\mathrm{I_{E} + I_{x}}$')
        pylab.plot(times,in_current,color = '0.65',label = r'$\mathrm{I_{I}}$')
        pylab.plot(times,ex_current+in_current+Ie,color = 'k',label = r'$\mathrm{I_{\text{tot}}}$')
        pylab.legend(loc = 'upper center',frameon = False,fontsize= 8,ncol = 3,
                     handlelength = 1.5,columnspacing = 1.,handletextpad = 0.5,
                     borderaxespad = 0.,borderpad = 0.)
        pylab.axis('tight')
        pylab.ylim(ymax = 1.25*pylab.ylim()[1])
        pylab.xlim(0,1000)
        pylab.xlabel(r'time [ms]')
        pylab.ylabel(r'$\mathrm{I_{\text{syn}}}$ [pA]')
        # voltage
        subplotspec = gs.new_subplotspec((4,0), colspan=2,rowspan=1)
        ax7 =simpleaxis(pylab.subplot(subplotspec))
        ax_label(ax7,'f', weigth='bold')
        pylab.plot(results['times'],results['V_m'][unit],'k')
        threshold = params.get('V_th_E',default.V_th_E)
        pylab.axhline(threshold,linestyle = '--',color = 'k')
        spikes = spiketimes[0,spiketimes[1]==unit]
        spikeheight= 20.
        for spike in spikes:
            pylab.plot([spike]*2,[threshold,threshold+spikeheight],'k')
        pylab.xlabel(r'time [ms]')
        pylab.ylabel(r'$\mathrm{V_{m}}$ [mV]')
        pylab.xlim(0,1000)
        pylab.savefig('suppl_fig4.pdf')

