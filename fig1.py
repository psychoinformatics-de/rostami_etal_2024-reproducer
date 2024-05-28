import sys;sys.path.append('../src')
import os
import pylab
import numpy as np
import matplotlib.image as mimage

# Local modules (not installed packages)
from analyse_experiment import get_stats, plot_experiment
from analyse_model import get_analysed_spiketimes
from GeneralHelper import (
    nice_figure, ax_label_fig1, ax_label_title,
    simpleaxis,colors,text_width_pts)
import network_schematic

data_path = '../data/preprocessed_and_simulated_data/'

# experimental data statistics for Moneky Joe (M1)
try:
    monkey = b'joe'
    fn = 'experiment_'+monkey.decode("utf-8") +'_ff_cv_twos.npy'
    d = np.load(data_path + fn, allow_pickle=True).item()
    locals().update(d)
except:
    tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, count_rates = get_stats(
        gns=None,monkey=monkey, save_interim=False)
    dict_ff_cv_twos = {'tff':tff, 'ffs':ffs, 'tcv_two':tcv_two,
                       'cv_twos':cv_twos}
    np.save(data_path + fn, dict_ff_cv_twos)

rcparams = {'axes.labelsize': 9,
            'lines.linewidth':1.5}
fig = nice_figure(ratio  =0.55,
                latex_page = 1.2*text_width_pts,
                rcparams=rcparams)


########################################################################
# plotting parameters ##################################################
########################################################################
ff_plotargs = {'color':colors['red']}
cv2_plotargs = {'color':(0,0,0)}
nrows = 5
ncols  =9
xlim = [0,2000]
ff_ylim = [0,2.5]
hspace = 0.05
plot = True
########################################################################

gs = pylab.GridSpec(nrows,ncols,top=0.9,bottom=0.1,hspace = 0.1,
                    wspace = 3.,left = 0.08,right = 0.94,
                    height_ratios = [0.5,0.3,0.4,hspace,0.4])


abc_size = 10
# load monkey drawing
if os.path.exists(data_path+'drawing_small.png'):
    print('drawing exists')
else:
    raise ValueError('drawing file not found1 Please download drawing_small.png file, following the istructions in the README.md file.')
subplotspec = gs.new_subplotspec((0,0), colspan=int(ncols/3),rowspan=1)
ax1 = pylab.subplot(subplotspec)
ax_label_fig1(ax1, 'a', size=abc_size)
ax_label_title(ax1, 'Behaving monkey')
drawing = mimage.imread(data_path+'drawing_small.png')
drawing = drawing[:-2]
pylab.imshow(drawing, cmap='gray')
pylab.axis('off')


# schematic of experimental condition
subplotspec = gs.new_subplotspec((1,0), colspan=int(ncols/3),rowspan=1)
ax2 = pylab.subplot(subplotspec)
pylab.axis('off')
plot_experiment(150, 40,lw =0.5,write_epoch=True)
pylab.axis('equal')
pylab.xlim(xlim)


# fano factor for experimental data
subplotspec = gs.new_subplotspec((2,0), colspan=int(ncols/3),rowspan=11)
ax3 = simpleaxis(pylab.subplot(subplotspec))
pylab.plot(tff,pylab.nanmean(ffs,axis = 0),**ff_plotargs)
pylab.xlim(xlim)
pylab.axvline(500,linestyle = '--',color = (0,0,0),lw = 0.5)
pylab.axvline(1500,linestyle = '--',color = (0,0,0),lw = 0.5)

# cv twos for experimental data
pylab.plot(tcv_two,pylab.nanmean(cv_twos,axis = 0),**cv2_plotargs)
pylab.xlim(xlim)
pylab.ylim(ff_ylim)
pylab.axvline(500,linestyle = '--',color = (0,0,0),lw = 0.5)
pylab.axvline(1500,linestyle = '--',color = (0,0,0),lw = 0.5)
pylab.axhline(1,linestyle = '--',color = (169/255,169/255,169/255),lw = 0.5)
pylab.ylabel(r'CV$_2$, FF', math_fontfamily='dejavusans')
pylab.xlabel('time [ms]')

# plot networ schematic E/I
subplotspec = gs.new_subplotspec((0,ncols-int(ncols/3)), colspan=int(ncols/3),
                                 rowspan=1)
ax4 = pylab.subplot(subplotspec)
ax_label_fig1(ax4, 'c', size=abc_size)
ax_label_title(ax4, 'E/I clustered network')
pylab.axis('off')
network_schematic.draw_EI_schematic()

# plot networ schematic EE
subplotspec = gs.new_subplotspec((0,ncols-2*int(ncols/3)), colspan=int(ncols/3),
                                 rowspan=1)
ax41 = pylab.subplot(subplotspec)
ax_label_fig1(ax41, 'b', size=abc_size)
ax_label_title(ax41, 'E clustered network')
pylab.axis('off')
network_schematic.draw_EE_network(I_radius = 60,I_position = [-170,-50], 
                                  y_offset  =-15)

# plot stimulus amplitude profile
subplotspec = gs.new_subplotspec((1,int(ncols/3)), colspan=int(ncols/3),rowspan=1)
ax5 = simpleaxis(pylab.subplot(subplotspec))
time = pylab.arange(xlim[0],xlim[1])
signal = (time>500)*(time<1500)
pylab.plot(time,signal*0.5,color = (0.6, 0.6, 0.6))
pylab.plot(time,signal,color = (0.4, 0.4, 0.4))
pylab.plot(time,signal*1.5,color =(0.1, 0.1, 0.1))
pylab.text(300, 1.9, "Stimulus Amplitude")
pylab.ylim(0,2.5)
pylab.axis('off')

subplotspec = gs.new_subplotspec((1,2*int(ncols/3)), colspan=int(ncols/3),
                                 rowspan=1)
ax51 = simpleaxis(pylab.subplot(subplotspec))
pylab.text(300, 1.9, "Stimulus Amplitude")
pylab.plot(time,signal*0.5,color = (0.6, 0.6, 0.6))
pylab.plot(time,signal,color = (0.4, 0.4, 0.4))
pylab.plot(time,signal*1.5,color =(0.1, 0.1, 0.1))
pylab.ylim(0,2.5)
pylab.axis('off')



subplotspec = gs.new_subplotspec((2,int(ncols/3)), colspan=int(ncols/3),rowspan=1)
ax6 = simpleaxis(pylab.subplot(subplotspec))
subplotspec = gs.new_subplotspec((2,2*int(ncols/3)), colspan=int(ncols/3),rowspan=1)
ax7 = simpleaxis(pylab.subplot(subplotspec))

########################################################################
# network model and simulation parameters ##############################
########################################################################
params = {'N_E':4000,'N_I':1000,'I_th_E':2.14,'I_th_I':1.26,
          'ff_window':400,'min_vals_cv2':1,
              'stim_length':1000,'isi':1000,'isi_vari':200,
              'cut_window':[-500,1500],
          'rate_kernel':50.,'warmup':500,'trials':20}
stim_range = [0,1,2]
settings = [{'randseed':24,'Q':50,'jipfactor':0.,'jep':3.45, 
             'stim_clusters':stim_range,'stim_amp':0.2, 'portion_I':50},
            {'randseed':24,'Q':50,'jipfactor':0.,'jep':3.45, 
             'stim_clusters':stim_range,'stim_amp':0.25, 'portion_I':50},
            {'randseed':24,'Q':50,'jipfactor':0.,'jep':3.45, 
             'stim_clusters':stim_range,'stim_amp':0.3, 'portion_I':50},
            {'randseed':0,'Q':50,'jipfactor':0.75,'jep':11.,
             'stim_clusters':stim_range,'stim_amp':0.2,'portion_I':1},
            {'randseed':0,'Q':50,'jipfactor':0.75,'jep':11.,
             'stim_clusters':stim_range,'stim_amp':0.25,'portion_I':1},
            {'randseed':0,'Q':50,'jipfactor':0.75,'jep':11.,
             'stim_clusters':stim_range,'stim_amp':0.3,'portion_I':1}]

params['fixed_indegree'] = False
params['trials'] = 20
params['n_jobs'] = 20
save = True
filename = 'fig1_simulated_data'
########################################################################


print('MODEL PLOT...')
def make_plot_ff_cv2(params,axes = None,plot = True,datafile=filename,
                        ff_plotargs={},cvtwo_plotargs = {},
                            calc_cv2s = True,t_offset  =0,save= False,
                            split_ff_clusters = False,split_cv2_clusters = False, 
                            ylim_ff=[0.,2.5], ylim_cv2 = [0.,1.3], 
                            xlim = [0,2000]):
    """plot fano factor and cv2s from simulation data"""
    # get data (if not exist simulate and generate data)
    result = get_analysed_spiketimes(params,datafile, calc_cv2s=calc_cv2s,
                                     save =save)
    # plot fano factor
    stim_clusters = params['stim_clusters']
    non_stim_clusters = [i for i in range(params['Q']) if i not in stim_clusters]
    axes[0].plot(result['t_ff']+t_offset,pylab.nanmean(
        result['ffs'][stim_clusters],axis=0),**ff_plotargs)
    axes[0].set_ylim(ylim_ff)
    axes[0].set_xlim(xlim)
    if split_ff_clusters:
        axes[0].plot(result['t_ff']+t_offset,pylab.nanmean(
            result['ffs'][stim_clusters],axis=0),linestyle = '--',**ff_plotargs)
        axes[0].plot(result['t_ff']+t_offset,pylab.nanmean(
            result['ffs'][non_stim_clusters],axis=0),linestyle = ':',**ff_plotargs)
    # plot cv2s
    if calc_cv2s:
        axes[1].plot(result['t_cv2']+t_offset,pylab.nanmean(
            result['cv2s'][stim_clusters],axis=0),label = 'all',**cvtwo_plotargs)
        axes[1].set_ylim(ylim_cv2)
        axes[1].set_xlim(xlim)
        if split_cv2_clusters:
            axes[1].plot(result['t_cv2']+t_offset,pylab.nanmean(
                result['cv2s'][stim_clusters],axis=0),
                         linestyle = '--',label = 'stim',**cvtwo_plotargs)
            axes[1].plot(result['t_cv2']+t_offset,pylab.nanmean(
                result['cv2s'][non_stim_clusters],axis=0),
                         linestyle = ':',label = 'non stim',**cvtwo_plotargs)
        #return result


# plot model data
for setno,setting in enumerate(settings):
    subplotspec = gs.new_subplotspec((2,(int(setno/3)+1)*int(ncols/3)), 
                                     colspan=int(ncols/3),rowspan=1)
    ax6 = simpleaxis(pylab.subplot(subplotspec))
    subplotspec = gs.new_subplotspec((4,(int(setno/3)+1)*int(ncols/3)), 
                                     colspan=int(ncols/3),rowspan=1)
    ax7 = simpleaxis(pylab.subplot(subplotspec))    
    for k in setting.keys():
        params[k] = setting[k]
    axes = [ax6, ax7] 
    if setno == 0:
        ax6.set_ylabel('FF')
        ax7.set_ylabel('CV$_2$', math_fontfamily='dejavusans')
    ff_plotargs = {'color':colors['red'], 'alpha':.5 + setno%3/4.}
    cv2_plotargs = {'color':(0,0,0), 'alpha':0.5+setno%3/4.}
    
    make_plot_ff_cv2(params,axes = axes,save = save,plot = plot,
                     split_ff_clusters=False,ff_plotargs=ff_plotargs,
                     cvtwo_plotargs = cv2_plotargs, t_offset = 500)
    
    for ax in axes:
        ax.axvline(500,linestyle = '--',color = (0,0,0),lw = 0.5)
        ax.axvline(1500,linestyle = '--',color = (0,0,0),lw = 0.5)
    axes[1].set_xlabel('time [ms]')

pylab.savefig('fig1.pdf',dpi = 300)
pylab.show()
