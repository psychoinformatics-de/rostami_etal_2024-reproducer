from types import ModuleType
import copy
import pylab
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.markers import TICKDOWN
import matplotlib
import networkx as nx
import ast 
import json
import pickle
import defaultSimulate as default
from joblib import Parallel,delayed
import collections
# ******************************************************************************
#                                    Classes
# ******************************************************************************


class TimeoutException(Exception):   # Custom exception class
    pass


class Organiser:
    """Organiser class for managing function results and data storage."""
    def __init__(self, params, datafile, datapath='preprocessed_and_simulated_data/',
                n_jobs=1, ignore_keys=[''], 
                 reps=None, redo=None, save=True):
        """Initialise the Organiser with the given parameters.
        
        Args:
            params (dict): Dictionary of parameters to be used in the function.
            datafile (str): Name of the datafile to save the results.
            datapath (str): Path to the datafile.
            n_jobs (int): Number of parallel jobs to run.
            ignore_keys (list): List of keys to ignore in the parameters.
            reps (int): Number of repetitions to run.
            redo (bool): Flag to redo the function execution.
            save (bool): Flag to save the results to the datafile.
        """
        self.params = params
        self.datafile = os.path.join(datapath, datafile)
        self.save = save
        self.n_jobs = n_jobs
        self.ignore_keys = ignore_keys
        self.reps = reps
        self.redo = redo

    def _load_results(self):
        """Load results from the datafile if it exists, 
        otherwise return an empty dictionary."""
        try:
            if os.path.exists(self.datafile):
                return pd.read_pickle(self.datafile)
        except Exception as e:
            print(f"Error loading datafile: {e}")
        return {}

    def _save_results(self, results_dict):
        """Save results dictionary to the datafile."""
        if self.save:
            try:
                with open(self.datafile, 'wb') as file:
                    pickle.dump(results_dict, file, protocol=2)
                    print(f"Results saved to '{self.datafile}'")
            except Exception as e:
                print(f"Error saving results: {e}")

    def _execute_function(self, func, params):
        """Execute the specified function with the given parameters."""
        try:
            return func(copy.deepcopy(params))
        except Exception as e:
            print(f"Error executing function: {e}")
        return None

    def _execute_parallel(self, func, params_list):
        """Execute tasks in parallel using joblib."""
        try:
            if type(params_list) != list and self.n_jobs > 1:
                print('running single job in parallel mode')
                return Parallel(n_jobs=self.n_jobs)(delayed(
                    self._execute_function)(func, params_list))
            elif type(params_list) != list and self.n_jobs == 1:
                return self._execute_function(func, params_list)
            else:
                return Parallel(n_jobs=self.n_jobs)(delayed(
                    self._execute_function)(
                        func, params) for params in params_list)
        except Exception as e:
            print(f"Error executing parallel function: {e}")
        return []

    def check_and_execute(self, func):
        """Check if results exist in the datafile, 
        otherwise execute the function and save the results."""
        key = key_from_params(self.params, self.reps, self.ignore_keys)
        results_dict = self._load_results()
        # insert literal newline string if exists
        updated_results_dict = {key.replace(
            '\n', '\\n'): value for key, value in results_dict.items()}

        if self.redo:
            print("Redo flag is set. Ignoring saved results.")
            results_dict = {}

        rerun = False
        if isinstance(key, list):
            for k in key:
                if k in updated_results_dict.keys() or k.replace(
                    '\n', '\\n') in updated_results_dict.keys() or k.strip(
                        ) in results_dict.keys() or k in results_dict.keys():
                    pass
                else:
                    rerun = True
                    break
        else:
            if key not in results_dict:
                rerun = True
        if rerun:
            print(f"Key '{key}' not found in results. Executing function '{func.__name__}'.")
            if self.reps is None:
                results_dict[key] = self._execute_parallel(func, self.params)
            else:
                params_list = [
                    copy.deepcopy(self.params) for _ in range(self.reps)]
                if self.n_jobs == 1:
                    results = []
                    for i in range(self.reps):
                        results.append(self._execute_function(
                            func, params_list[i]))
                else:
                    results = self._execute_parallel(func, params_list)
                results_dict.update(zip(key, results))
            self._save_results(results_dict)
        if self.reps is None:
            return results_dict.get(key)
        else:
            return [results_dict.get(k) for k in key]


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            print('uncacheable: ',args)
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

class memoized_but_forgetful(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args,**kwargs):

        key = (args, frozenset(sorted(kwargs.items())))
        if not isinstance(key, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args,**kwargs)
        if key in self.cache:
            return self.cache[key]
        else:
            # forget old value
            self.cache = {}
            value = self.func(*args,**kwargs)
            self.cache[key] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


class LineText(Text):
    """ Inline label for Line2D objects. Since the line behind the label 
        is filled with nans to avoid showing it, it works best with lines 
        of densely spaced points.

        line: handle of a Line2D object as returned by plot 

        label: the label string

        pad:  padding between line and label as fraction of text height

        x_pos,y_pos: position of the label in data coordinates. 
                     If both are None, x_pos is taken to be line.xdata.mean()
        
        kwargs as in matplotlib.text.Text
        """

    def __init__(self,line,label=None,pad=0.1,x_pos=None,y_pos=None,*args,**kwargs):
        if label is None:
            label = line.get_label()
        kwargs['color'] = kwargs.get('color',line.get_color())
        Text.__init__(self,text=label,ha = 'center',va = 'center',rotation_mode = 'anchor',**kwargs)
        self.line = line
        self.pad = pad
        self.xdata = line.get_xdata()
        self.ydata = line.get_ydata()

        
        self.rotation_initialised  =False
        self.nearest_ind = None

        self._init_position(x_pos,y_pos)

        

    def _init_position(self,x_pos,y_pos):
        """ find a position for the label. """
        if (x_pos is None) and (y_pos is None):
            x_pos = self.xdata.mean()
        if x_pos is None:
            # find closest y point
            self.nearest_ind = np.argmin(np.absolute(self.ydata-y_pos))
        elif y_pos is None:
            # find nearest x point
            self.nearest_ind = np.argmin(np.absolute(self.xdata-x_pos))
        else:
            # find nearest point to specified x,y point
            self.nearest_ind = np.argmin((self.xdata-x_pos)**2+(self.ydata-y_pos)**2)
         
        x = self.xdata[self.nearest_ind]
        y = self.ydata[self.nearest_ind]
        self.set_position([x,y])

    def _get_line_segment(self,renderer):
        """ find the left and right indices of the line segment 
            that is covered by the label. """
        # the size of the label in data coordinates   
        textheight,textwidth = self._get_text_height_width(renderer)
        x,y = self.get_position()
        # find the closest points on both sides of x and y where (dx**2+dy**2)**0.5 ~ (0.5*textwidth+pad*textheight)
        left_ind = np.argmin(np.absolute(((self.xdata[:self.nearest_ind]-x)**2+(self.ydata[:self.nearest_ind]-y)**2)**0.5-(0.5*textwidth+self.pad*textheight)))
        right_ind = np.argmin(np.absolute(((self.xdata[self.nearest_ind:]-x)**2+(self.ydata[self.nearest_ind:]-y)**2)**0.5-(0.5*textwidth+self.pad*textheight)))+self.nearest_ind
        
        return left_ind,right_ind
    
    def _get_text_height_width(self,renderer):
        """ calculate the extent of the text box in data coordinates. """
        tbox = self.get_window_extent(renderer)
        dbox = tbox.transformed(self.axes.transData.inverted())
        return dbox.height,dbox.width

    def _init_rotation(self,renderer):
        """ calculate the angle of the line connecting the ends 
            of the line segment covered by the label. """

        # get the segment of the line that is covered by the text
        left_ind,right_ind = self._get_line_segment(renderer)
        # calculate the angle in data coordinates
        angle = np.degrees(np.arctan2(self.ydata[right_ind]-self.ydata[left_ind],self.xdata[right_ind]-self.xdata[left_ind]))
        self.set_rotation(angle)
    
    def _blank_out_covered_segment(self,renderer):
        """ fill the line segment covered by the label with 
            nans so that a gap arises for the label. """
        left_ind,right_ind = self._get_line_segment(renderer)
        newydata = self.ydata.copy()
        newydata[left_ind:right_ind] = np.nan
        self.line.set_ydata(newydata)
        self.line.draw(renderer = renderer)
        
    def draw(self,renderer):
        """ draw the label. """
        # to calculate the rotation, the extent of the label needs to be known
        # this can only be calculated when the label is first drawn
        if not self.rotation_initialised:
            self._init_rotation(renderer)
            Text.draw(self, renderer)
            self.rotation_initialised = True

        # blank out the line segment covered by the text
        self._blank_out_covered_segment(renderer)
        Text.draw(self, renderer)
        
    def get_rotation(self):
        """
        Unlike the ordinary text, the get_rotation returns an updated
        angle in the pixel coordinate assuming that the input rotation is
        an angle in data coordinate (or whatever transform set).

        (copied from contour.ClabelText)
        """
        angle = Text.get_rotation(self)
        trans = self.get_transform()
        x, y = self.get_position()
        new_angles = trans.transform_angles(np.array([angle]),
                                            np.array([[x, y]]))
        return new_angles[0]
    




# ******************************************************************************
#                                    Functions
# ******************************************************************************


def mergeParams(params, defaultValues):
    """
    Updates default Values defined in a module with params-dictionary.
    The returned dictionary contains the parameters used for simulation.
    """
    return nested_update(moduleVarToDict(defaultValues), params)


def timeout_handler(signum, frame):
    """
    # Custom signal handler for timeout. 
    Should be raised if a set of instructions takes too long
    """   
    raise TimeoutException


def moduleVarToDict(module):
    """
    Creates dict of explicit variables in module which are not imported modules.
    Key names equal the variable name.
    """
    ModuleDict = {}
    if module:
        ModuleDict = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_') or isinstance(value, ModuleType))}
    return ModuleDict

def nested_update(d, d2):
    """
    Updates values in d with values of d2. Adds new keys if they are not present.
    Returns Changed dict.
    """
    d_local=copy.deepcopy(d)
    for key in d2:
        if isinstance(d2[key], dict) and key in d_local:
            d_local[key] = nested_update(d_local[key], d2[key])
        else:
            d_local[key] = d2[key]
    return d_local


def find(condition):
    """Old pylab.find function"""
    res, = np.nonzero(np.ravel(condition))
    return res

def round_to_resolution(x, dt):
    """Rounds x to the nearest multiple of dt"""
    rounded_value = round(x / dt) * dt
    return rounded_value

def random_pick(x,n=1):
    """Randomly picks n elements from x"""
    inds = pylab.arange(len(x))
    pylab.shuffle(inds)
    inds = inds[:n]
    if n==1:
        inds= inds[0]
    return x[inds] 

def extract_info_from_keys(params, keys):
    """Extract information from string keys 
    and update the params dictionary."""
    
    keys_dict_str = keys[keys.find('{') + 1:keys.find('}')]
    keys_dict_str = keys_dict_str.replace("'", '"')
    
    # Evaluate the cleaned string as a dictionary and update params
    params.update(ast.literal_eval(f"{{{keys_dict_str}}}"))
    return params


def key_from_params(params, reps=None, ignore_keys=None):
    """Create string keys from parameters dictionary."""
    if ignore_keys is None:
        ignore_keys = ['']

    # Filter keys to include based on sorted order and ignore list
    filtered_keys = [k for k in sorted(params.keys()) if k not in ignore_keys]

    # Generate key strings
    key_parts = []
    for key in filtered_keys:
        key_parts.append(params[key].__repr__().replace(' ', ''))

    # Join key parts with '_' to form the final key
    key = '_'+'_'.join(key_parts)

    # Handle repetitions if specified
    if reps is not None:
        key = [f"{key}_{i}" for i in range(reps)]#[key + '_' + str(r) for r in range(reps)]

    return key


def compare_key(all_keys, params):
    """Compare dictionary keys in string format with a given 
    parameter dictionary."""
    for key_string in all_keys:
        try:
            # Extract the dictionary part from the key string
            dict_str = key_string.split('{', 1)[1].rsplit('}', 1)[0]
            key_dict = json.loads('{' + dict_str + '}')
            
            # Compare the parsed dictionary with the given params dictionary
            if key_dict == params:
                return key_string
        except:
            pass

# ******************************************************************************
#                         global functions for plotting
# ******************************************************************************

def math_string(text):
    return r'$'+text+'$'
def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k',zorder  =None):
    # draw a line with downticks at the ends
    pylab.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize,zorder = zorder)
    # draw the text with a bounding box covering up the line
    pylab.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize,zorder  =zorder)


def line_intersections(A, B):

    # min, max and all for arrays
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)
    aall = lambda abools: np.dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))

    x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = np.meshgrid(slope(A), slope(B))
    m1inv, m2inv = 1/m1, 1/m2

    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12), 
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    return xi[aall(xconds)], yi[aall(yconds)]

def contour_intersections(c1,c2):

    points = [[],[]]
    for col1 in c1.collections:
        for path1 in col1.get_paths():
            for col2 in c2.collections:
                for path2 in col2.get_paths():
                    if path1.intersects_path(path2)==1:
                        # get the vertices
                        x,y = line_intersections(path1.vertices,path2.vertices)
                        points[0]+=x.tolist()
                        points[1]+=y.tolist()

    return np.array(points).T


def make_graph(weights):
    graph = nx.DiGraph()
    edges = []
    edge_sign =  []
    node_sign = []
    for i in range(weights.shape[0]):
        node_sign.append(pylab.sign(weights[:,i].mean()))
        for j in range(weights.shape[1]):
            if weights[i,j]!=0:
                edge_sign.append( pylab.sign(weights[i,j]) )
                
                edges.append((j,i,abs(weights[i,j])))
    graph.add_weighted_edges_from(edges)
    
    return graph,edge_sign,node_sign

def make_graph_positions(graph,scale=1):
    pos = nx.spring_layout(graph,iterations  =100,scale = scale)
    return pos
def draw_network(weights,alpha = 0.04,node_size = 25,pos = None,e_col = 'k',i_col = 'r',node_colors = None,edge_colors = None,zorder = None):
    
    graph,edge_sign,node_sign = make_graph(weights)       
    if edge_colors is None:
        edge_colors = []
        for e in edge_sign:
            
            if e>0:
                edge_colors.append(e_col)
            else:
                edge_colors.append(i_col)
    if node_colors is None:
        node_colors = []
        for n in node_sign:
            if n>0:
                node_colors.append(e_col)
            else:
                node_colors.append(i_col) 
    
    if pos is None:
        pos = make_graph_positions(graph)
    nx.draw_networkx(graph, pos = pos,
                     alpha = alpha,arrows = False,node_size = node_size,
                     with_labels = False,edge_color = edge_colors,node_color = node_colors,linewidths = 0.,zorder = zorder)
 
def psth_plot(spiketimes,binsize=2.,tlim = None,color = 'k',lw = 2.):
    if tlim is None:
        tlim = [spiketimes[0].min(),spiketimes[0].max()]
    bins = pylab.linspace(int(tlim[0]), int(tlim[1]),int((tlim[1]-tlim[0])//binsize))
    h,b = pylab.histogram(spiketimes[0],bins)
    x = pylab.tile(b,(2,1)).T.reshape(len(b)*2)
    y = pylab.array([0] + pylab.tile(h,(2,1)).T.reshape(len(h)*2).tolist()+[0])
    fillcolor = list(col_conv.to_rgb(color))+[0.2]
    N_units = len(pylab.unique(spiketimes[1]))
    pylab.fill_between(x,y/binsize*1000./float(N_units),pylab.ones_like(y)*-1,color = color,lw = lw,facecolor = fillcolor)
    pylab.ylim(ymin = 0.)

def raster_plot(output,plot_args={'color':'k','markersize':0.5},xscale = None,yscale = None,time_axis = None):

    u,t = pylab.where(output)
    if time_axis is not None:
        t = time_axis[t]
    if xscale is not None:
        t= t.astype(float)
        t = t/t.max() * xscale
    if yscale is not None:
        u= u.astype(float)
        u = u/u.max() * yscale

    

    pl =  pylab.plot(t,u,'.',**plot_args)
    pylab.gca().invert_yaxis()
    return pl
def scatter_weights(weights,sizes = 5.,cmap  =pylab.cm.RdBu):
    x,y = pylab.where(weights!=0)
    ws = weights[x,y]
    maxval = pylab.absolute(ws).max()
    
    pylab.scatter(x,y,s=sizes,c = ws,edgecolors = 'none',cmap = cmap,vmin = -maxval,vmax = maxval)


def draw_box(box,box_target,box_color):
    xy = [[box[0],box[1]],
              [box[0],box[1]+box[3]],
              [box[0]+box[2],box[1]+box[3]],
              [box[0]+box[2],box[1]],
               box_target]
        
    pol = Polygon(xy,facecolor = box_color,edgecolor = 'none')
    pylab.gca().add_patch(pol)
def small_ax(signals,box,ylim,plotargs,box_target=None,box_color = None):
    
    if box_color is not None and box_target is not None:
        draw_box(box,box_target,box_color)
    
    for s,pl in zip(signals,plotargs):
        x,y,dx,dy = box
        ploty = s.copy()
        ploty -=float(ylim[0])
        ploty/= float(ylim[1])
        ploty = ploty*dy +y
        plotx = pylab.linspace(x,x+dx,len(ploty))
        pylab.plot(plotx,ploty,**pl)
    
def subplot_grid(x,y):
    axes = []
    for i in range(x):
        axes.append([])
        for j in range(y):
            axes[-1].append(pylab.subplot(y,x,j*x+i+1))
    return axes  

def make_color_list(n,cmap = 'jet',minval = 0.0,maxval = 1.):
    if n==1:
        return make_color_list(2,cmap,minval,maxval)[:1]
    if n==3:
        colors = make_color_list(4,cmap,minval,maxval)
        colors.pop(1)
        return colors
    cmap = pylab.cm.get_cmap(cmap)
    return [cmap(val) for val in pylab.linspace(minval,maxval,n)]      

col_conv= matplotlib.colors.ColorConverter()

def ax_label(ax,label,size= 10, family ='sans-serif', weigth = 'normal'):
    pylab.sca(ax)
    pylab.title(label,loc = 'left',
                fontsize = size,family =family, weight = weigth)
    return ax

def ax_label_title(ax,label,size= 10,
                   family ='sans-serif'):
    pylab.sca(ax)
    pylab.title(label,loc = 'center',
                fontsize = size,family =family)
    return ax


def ax_label1(ax,label,x=-0.5, size= 8,y=1.1):
    pylab.sca(ax)
    ax.text(x, y, label,transform=ax.transAxes, 
            size=size, weight='bold')
    return ax
def ax_label2(ax,label,size= 10):
    ax.annotate(label, xy=(0.9, 0.9), xycoords="axes fraction")
    return ax

def ax_label_fig1(ax,label,size= 8):
    pylab.sca(ax)
    ax.text(-0.3, 1.08, label,transform=ax.transAxes, 
            size=size, weight='bold',font='DejaVu Sans')
    return ax


def simpleaxis(ax,labelsize = 5,pad=1.5):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(which = 'minor',direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(which = 'minor',direction='out')
    ax.tick_params('both', length=2, width=0.5, which='major',pad = pad,labelsize=labelsize)
    return ax

def simpleaxis1(ax,labelsize = 5,pad=1.5):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(which = 'minor',direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(which = 'minor',direction='out')
    return ax

def nice_figure(fig_width= 1,ps = False,square = False,
                ratio = None,latex_page = 510.,rcparams = {}):
    # A4 size in pt 595 × 842 pts
    # A4 size in mm 210 x 297 mm
    # A4 size in inches 8.27 x 11.69 inches
    
    inches_per_pt = 1.0/(72.27)  
    if not square and ratio is None:        # Convert pt to cm
        golden_mean = (pylab.sqrt(5)-1.0)/2.0
        ratio = golden_mean# Aesthetic ratio
    elif ratio is None:
        ratio = 0.9         # 1. doesn't look right
    fig_width = fig_width*latex_page*inches_per_pt  # width in inches
    fig_height = fig_width*ratio      # height in inches
    fig_size =  [fig_width,fig_height]
           
    params = {'axes.labelsize': 7,
              'font.size':7,
              'font.family':'sans-serif',
              'legend.fontsize': 7,
              'xtick.labelsize':7,
              'ytick.labelsize': 7,
              'mathtext.fontset': 'cm',
              'figure.figsize' : fig_size,
            'xtick.major.size': 3,
            'ytick.major.size': 3.,          
            'lines.linewidth':1.,
            'axes.linewidth':0.2}

    params.update(rcparams)
    
    
    if ps:
        params['backend']= 'ps'
        params['savefig.dpi']=300
            
              
    pylab.rcParams.update(params)
    return pylab.figure()
def std_plot(x,y,color):
    color = list(col_conv.to_rgb(color))+[0.2]
    mean = pylab.nanmean(y,axis = 0)
    std = pylab.nanvar(y,axis = 0)**0.5
    
    not_nan_inds = pylab.find(pylab.isnan(mean)==False)
    mean = mean[not_nan_inds]
    std  = std[not_nan_inds]
    x = x[not_nan_inds]
    pylab.fill_between(x,mean+std,mean-std,color = color)

def sem_plot(x,y,color,lw = 1,line_style = '-',fill = False):
    
    mean = pylab.nanmean(y,axis = 0)
    std = pylab.nanvar(y,axis = 0)**0.5
    
    not_nan_inds = pylab.find(pylab.isnan(mean)==False)
    mean = mean[not_nan_inds]
    std  = std[not_nan_inds]
    x = x[not_nan_inds]
    std/=pylab.sqrt(float(std.shape[0]))
    if fill:
        pylab.fill_between(x,mean+std,mean-std,color = list(
            col_conv.to_rgb(color))+[0.2])
    else:
        
        pylab.plot(x,mean+std,line_style,lw = lw,color = color)
        pylab.plot(x,mean-std,line_style,lw = lw,color = color)
    
def percentile_plot(x,y,ps = [25,75],color='b'):
    color = list(col_conv.to_rgb(color))+[0.2]
   
    y.sort(axis = 0)
    nan_counts = pylab.isnan(y).astype(int).sum(axis =0)
    not_nan_counts = y.shape[0]-nan_counts
    bottom_inds = pylab.floor((not_nan_counts-1)*ps[0]/100.).astype(int)
    top_inds = pylab.floor((not_nan_counts-1)*ps[1]/100.).astype(int)
    bottom = y[bottom_inds,list(range(y.shape[1]))]
    top =  y[top_inds,list(range(y.shape[1]))]
    
    not_nan_inds = pylab.find(pylab.isnan(bottom)==False)
    pylab.fill_between(x[not_nan_inds],bottom[
        not_nan_inds],top[not_nan_inds],color = color)
    


def get_custom_cmap(name = 'custom_colors',cdict = None):
    if cdict is None:
        cdict = {'red':   ((0.0, 0., 0.), 
                           (0.6, 0.1,0.1),  # red 
                           (0.9, 1.0, 1.0),  # violet
                           (1.0, 0.7, 0.7)), # blue
            
                 'green': ((0.0, 0.0, 0.0),
                           (0.5, 0.4, 0.4),
                           (0.7, 0.9, 0.9),
                           (1.0, 0.0, 0.0)),
            
                 'blue':  ((0.0, 0.7, 0.7),
                           (0.1, 1., 1.),  # red
                           (0.4, 0.1, 0.1),  # violet
                           (1.0, 0.,0.))  # blue
                  }
    return pylab.matplotlib.colors.LinearSegmentedColormap(name,cdict)


def labeled_line(x,y,label,label_x=None,label_pad = 0.5,
                 lw = 1.,color = 'k',fontsize = 10.,alpha = 1.,
                 textcolor = None,textalpha = None,angle_width = None):
    pylab.plot(x,y,lw = lw,color = color,alpha = alpha)
    if label_x is None:
        label_x= 0.5*(x.min()+x.max())

    closest_x_ind = pylab.argmin(pylab.absolute(x-label_x))
    if angle_width is None:
        offset= 1
    else:
        offset = int(angle_width/(x[1]-x[0]))
    y_before = y[max(closest_x_ind-offset,0)]
    y_after = y[min(closest_x_ind+offset,len(x))]
    x_before = x[max(closest_x_ind-offset,0)]
    x_after = x[min(closest_x_ind+offset,len(x))]
    
    

    # angle needs to be calculated in axis coordinates
    points = pylab.array([[x_before,y_before],[x_after,y_after]])
    transformed_points = points
    angle = pylab.arctan(
        (transformed_points[1,1]-transformed_points[0,1])/(
            transformed_points[1,0]-transformed_points[0,0]))
    angle = pylab.degrees(angle)
    print(angle)
    
    xtext = x[closest_x_ind]
    ytext = y[closest_x_ind]
    if textcolor is None:
        textcolor  =color
    if textalpha is  None:
        textalpha = alpha
    pylab.text(
        xtext, ytext, label,fontsize = fontsize,ha='center',va = 'center',
        bbox=dict(facecolor='1.', edgecolor='none',
        boxstyle='square,pad='+str(label_pad),clip_on=True),
        rotation_mode = None,rotation = angle,color = textcolor,
        alpha = textalpha,clip_on = True)


def line_label(line,**kwargs):
    linetext = LineText(line,**kwargs)
    line.axes.add_artist(linetext)
    pylab.draw()
    return linetext
def line_label_old(line,label,x_pos = None,y_pos = None,
                   color = None,alpha = None,fontsize = 8,pad = 0.5):
    
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if color is None:
        color = line.get_color()
    if alpha is None:
        alpha = line.get_alpha()

    if x_pos is None and y_pos is None:
        x_pos = xdata.mean()
    
    if x_pos is None:
        # find closest y point
        nearest_ind = pylab.argmin(pylab.absolute(ydata-y_pos))
    elif y_pos is None:
        # find nearest x point
        nearest_ind = pylab.argmin(pylab.absolute(xdata-x_pos))
    else:
        # find nearest point to specified x,y point
        nearest_ind = pylab.argmin((xdata-x_pos)**2+(ydata-y_pos)**2)
     
    x = xdata[nearest_ind]
    y = ydata[nearest_ind]

    ax = line.get_axes()
    # add a text object to the axis without rotation
    txt = LineText(line,x,y,label,fontsize = fontsize,color =color,
                   alpha = alpha,ha = 'center',va = 'center',
                   rotation_mode = None,clip_on = True)
    ax.add_artist(txt)
    # get the width of the text
    # for this to work the figure needs to be drawn first...
    pylab.draw()
    renderer = ax.get_renderer_cache()
    tbox = txt.get_window_extent(renderer)
    dbox = tbox.transformed(ax.transData.inverted())
    textwidth = dbox.width
    textheight = dbox.height
    print(textwidth,textheight)
    # find the closest points on both sides of x and y where (dx**2+dy**2)**0.5 ~ (0.5+pad)*textwidth
    left_ind = pylab.argmin(pylab.absolute(((
        xdata[:nearest_ind]-x)**2+(
            ydata[:nearest_ind]-y)**2)**0.5-(0.5*textwidth+pad*textheight)))
    right_ind = pylab.argmin(pylab.absolute(((
        xdata[nearest_ind:]-x)**2+(
            ydata[nearest_ind:]-y)**2)**0.5-(0.5*textwidth+pad*textheight)))+nearest_ind
    # calculate the angle in data coordinates
    angle = pylab.degrees(
        pylab.arctan2(ydata[right_ind]-ydata[left_ind],
                      xdata[right_ind]-xdata[left_ind]))
    txt.set_rotation(angle)
    # set the y values in the label range to nan so they become invisible
    ydata[left_ind:right_ind]=pylab.nan
    line.set_ydata(ydata)
    



# ******************************************************************************
#                        global parameters for plotting
# ******************************************************************************

text_width_pts = 400
cycle_cmap = 'jet'

colors = {'FF':'k',
          'CV2':'b',
          'off_gray': [0.7]*3,
          'yellow': pylab.array([228,215,35])/255.,
          'green': pylab.array([100,200,0])/255.,
          'red': pylab.array([200,35,35])/255.,
          'orange':pylab.array([233,105,44])/255.}

off_gray = colors['off_gray']
yellow = colors['yellow']
green = colors['green']
red = colors['red']

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00',
                  '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


