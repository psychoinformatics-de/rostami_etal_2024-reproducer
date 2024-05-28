import pylab
from matplotlib.patches import Polygon,Circle,Rectangle
from scipy.spatial.distance import pdist,squareform
from GeneralHelper import Organiser, yellow, red, find
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from GeneralHelper import nice_figure, find
# def get_weight_matrix(N_E = 4000,N_I=1000,Q = 20,jplus = 5,
#                       jip_factor = 3/4.,ps = None,g = 1.2):

#     # generate weights
#     if ps is None:
#         ps = pylab.ones((2,2))*0.5
#         ps[0,0] = 0.1
#     jplus = pylab.ones((2,2))*jplus
    
#     newNs,newps,newjs,newTs,newtaus = BiNet.weights.EI_jplus_cluster_specs(
#         [N_E,N_I], ps, pylab.ones((2,)), pylab.ones((2,)), 
#         g, Q, jplus,jip_factor)
    
#     w = BiNet.weights.generate_weight_matrix(newNs,newps,newjs,None)

#     # make list of cluster indices
#     E_cluster_size = N_E/Q
#     I_cluster_size = N_I/Q

#     clusters = [list(range(e,e+E_cluster_size)) for e in range(0,N_E,E_cluster_size)] +\
#                [list(range(i,i+I_cluster_size)) for i in range(N_E,N_E+N_I,I_cluster_size)]

    
#     #assert len(clusters) == 2*Q
#     #assert set([item for sublist in clusters for item in sublist])==set(range(N_E+N_I))

#     return w,clusters


def _rotate_around_origin(pos,degrees):
    radians = pylab.deg2rad(degrees)
    sin  =pylab.sin(radians)
    cos = pylab.cos(radians)
    mat = pylab.array([[-sin,cos],[cos,sin]])
    return pylab.dot(mat,pos.T).T

def _circle_slice_patch(pos= [0,0],radius = 10,fraction = 0.1,rotation = 10,ec = (0,0,0),fc = (1,1,1),lw=2,resolution  =100):
    
    h = radius*(2*fraction -1)

    alpha = pylab.arccos(-h/radius)

    angles = pylab.linspace(alpha,-alpha, resolution)

    x =radius* pylab.sin(angles)
    y =radius* pylab.cos(angles)
    
    positions = pylab.append(x[:,None],y[:,None],axis=1)

    positions = _rotate_around_origin(positions, rotation)
    
    
    positions[:,0] += pos[0]
    positions[:,1] += pos[1]

    
    
    #ts = pylab.gca().transData
    #coords = ts.transform(pos)
    #tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], rotation)
    #t= ts + tr
    #print tr
    patch = Polygon(positions,ec = ec,fc = fc,lw = lw)
    
    return patch




def _draw_subnetwork(pos = [0,0],fraction  =0.25 ,radius=10,orientation=0,colors = [(0,0,0),red],strings = ['',''],lw = 5,gap = 0.04,text = False,fontsize = 6):
    
    ax = pylab.gca()
    p = _circle_slice_patch(pos,radius  =radius,fraction = fraction*(1-gap),ec=colors[1],rotation = orientation-90,lw = lw)
    ax.add_patch(p)

    
    p = _circle_slice_patch(pos,radius  =radius,fraction = (1-fraction)*(1-gap),ec=colors[0],rotation = orientation+90,lw = lw)
    ax.add_patch(p)
    
    # add another patch to ensure the background is white in the gap
    p = _circle_slice_patch(pos,radius  =radius,fraction = 1 ,ec='none',rotation = orientation+90,lw = lw)
    ax.add_patch(p)
    p.set_zorder(-1)
   
    pylab.text(pos[0], pos[1]+0.2*radius, strings[0],color = colors[0],size = fontsize,va = 'center',ha = 'center')
    pylab.text(pos[0], pos[1]-0.8*radius, strings[1],color = colors[1],size = fontsize,va = 'center',ha = 'center')

def _circular_distribution_wrapper(params):
    return _circular_distribution(**params)
def _circular_distribution_old(n=20,radius=100,min_dist = 0,randseed=None):
    if randseed is not None:
        pylab.seed(randseed)


    if min_dist>0:
        positions = _circular_distribution(n*500,radius,min_dist = 0)
        distances = squareform(pdist(positions))
        distances[list(range(len(positions))),list(range(len(positions)))] = pylab.nan
        rep = 0
        while pylab.nanmin(distances)< min_dist:
            rep+=1
            
            order = pylab.argsort(pylab.nanmin(distances,axis=1))
            
            new_len = int(len(positions)*9/10.)
            print(rep,new_len,n)
            if new_len>n:

                positions = positions[order[len(positions)/10:]]
            else:
                positions /= pylab.nanmin(distances)/(1.1*min_dist)
                print('scaled ',min_dist,pylab.nanmin(distances))

            distances = squareform(pdist(positions))
            distances[list(range(len(positions))),list(range(len(positions)))] = pylab.nan
            

        
        if len(positions)>n:
            order = pylab.arange(len(positions))
            pylab.shuffle(order)
            order = order[:n]
            positions = positions[order]


        
    else:
        radii = (pylab.rand(n)*radius**2)**0.5
        angles = pylab.rand(n)*2*pylab.pi

        x = radii * pylab.sin(angles)
        y = radii * pylab.cos(angles)

        positions = pylab.append(x[:,None],y[:,None],axis=1)

    

    return positions

def _circular_distribution(n=20,radius=100,min_dist = 0,randseed=None):
    if randseed is not None:
        pylab.seed(randseed)


    if min_dist>0:
        positions = _circular_distribution(n*10,radius,min_dist = 0)
        distances = squareform(pdist(positions))
        distances[list(range(len(positions))),list(range(len(positions)))] = pylab.nan
      
        while pylab.nanmin(distances)< min_dist:
           
            
            if (len(positions)-1)==n:
                positions /= pylab.nanmin(distances)/(1.1*min_dist)
                print('rescaled')
                
            else:
                d_min = pylab.nanmin(distances)
                row,col = pylab.where(distances == d_min)
                distances[row,col] = pylab.nan 
                unitmins = pylab.nanmin(distances[row],axis=1)
                remove = row[pylab.argmin(unitmins)]
                keep = [r for r in range(len(positions)) if r!= remove]
                positions = positions[keep]

            distances = squareform(pdist(positions))
            distances[list(range(len(positions))),list(range(len(positions)))] = pylab.nan
            

        
        if len(positions)>n:
            order = pylab.arange(len(positions))
            pylab.shuffle(order)
            order = order[:n]
            positions = positions[order]


        
    else:
        radii = (pylab.rand(n)*radius**2)**0.5
        angles = pylab.rand(n)*2*pylab.pi

        x = radii * pylab.sin(angles)
        y = radii * pylab.cos(angles)

        positions = pylab.append(x[:,None],y[:,None],axis=1)

    

    return positions


def draw_network(n = 20,radius =11,lw = 2,
                 gap = 0.05,randseed = None,dist_radius = 100,
                 connection_lw = 0.1,fraction = 0.25,
                 connection_alpha=0.1,x_offset = 0,y_offset  = 0):
    """draw clustered network of excitatory and inhibitory neurons"""
    position_params = {'n':n,'radius':dist_radius,
                       'min_dist':2.02*radius,'randseed':randseed}
    ORG = Organiser(position_params, 'circular_positions')
    positions = ORG.check_and_execute(_circular_distribution_wrapper)
    positions[:,0] += x_offset
    positions[:,1] += y_offset
    orientations = pylab.rand(n)*360
    
    for i in range(len(positions)):
        for j in range(len(positions)):
            pylab.plot(positions[[i,j],0],positions[[i,j],1],zorder = -3,color = (0,0,0),alpha = connection_alpha,lw= connection_lw)
    for position,orientation in zip(positions,orientations):
        
        _draw_subnetwork(pos = position,orientation=orientation,radius  = radius,lw = lw,gap= gap,fraction = fraction)


    return positions

def arc_coords(center,radius,angle):
    theta= pylab.deg2rad(angle)
    
    y = radius *pylab.cos(theta)
    x = radius *pylab.sin(theta)
    return pylab.array((x,y))+pylab.array(center)

def _add_self_conntection(position,radius,dotsize,dot_offset,central_angle,angle_offset,connection_width,color):

    center = arc_coords(position, radius, central_angle)
    start = arc_coords(position, radius, central_angle-angle_offset)
    end = arc_coords(position, radius+dot_offset, central_angle+angle_offset)
    connection_radius = pdist(pylab.append(center[None,:],start[None,:],axis =0))
    
    connection = Circle(center,connection_radius,fc = 'none',ec = color,lw = connection_width,zorder = -2)
    pylab.gca().add_patch(connection)
    pylab.plot(end[0],end[1],'o',ms = dotsize,mec = color,mfc = color,zorder = -2)


def _add_external_connection(position1,position2,radius,angle1,angle2,dotsize,dot_offset,connection_width,color,linestyle,rad):

    start = arc_coords(position1,radius,angle1) 
    end = arc_coords(position2,radius+dot_offset,angle2) 

    pylab.gca().annotate('', xy=end, xycoords='data',
                xytext=start, textcoords='data',
                size=8,
                # bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="-",
                                fc=[0,0,0,1], ec=color,
                                connectionstyle="arc3,rad="+str(rad),shrinkB=0,shrinkA =0,lw = connection_width,linestyle = linestyle),zorder = -1,
                )

    
    pylab.plot(end[0],end[1],'o',ms = dotsize,mec = color,mfc = color,zorder= -2)
def draw_detail(radius=40,distance=100,position = [0,0],lw=3,gap = 0.08,colors = [(0,0,0),red],connection_width=2,dotsize=10,dot_offset=1,angle_offset = 17,dashes = (0,(3,3)),fraction = 0.25):


    position1 = pylab.array(position)
    position2 = position1.copy()
    position2[0]+= distance
    

    _draw_subnetwork(pos = position1,radius = radius,lw = lw,gap  =gap,strings = ['E','I'],fraction = fraction)
    _draw_subnetwork(pos = position2,radius = radius,lw = lw,gap  =gap,strings = ['E','I'],fraction = fraction)


    
    # now draw the self connections
    _add_self_conntection(position1,radius,dotsize,dot_offset,central_angle=0,angle_offset=angle_offset,connection_width=connection_width,color=colors[0])
    _add_self_conntection(position1,radius,dotsize,dot_offset,central_angle=180,angle_offset=angle_offset,connection_width=connection_width,color=colors[1])

    _add_self_conntection(position1,radius,dotsize,dot_offset,central_angle=-120,angle_offset=-angle_offset*1.5,connection_width=connection_width,color=colors[0])
    _add_self_conntection(position1,radius,dotsize,dot_offset,central_angle=-120,angle_offset=angle_offset*0.8,connection_width=connection_width,color=colors[1])

    # population 2
    _add_self_conntection(position2,radius,dotsize,dot_offset,central_angle=0,angle_offset=angle_offset,connection_width=connection_width,color=colors[0])
    _add_self_conntection(position2,radius,dotsize,dot_offset,central_angle=180,angle_offset=angle_offset,connection_width=connection_width,color=colors[1])

    _add_self_conntection(position2,radius,dotsize,dot_offset,central_angle=120,angle_offset=angle_offset*1.5,connection_width=connection_width,color=colors[0])
    _add_self_conntection(position2,radius,dotsize,dot_offset,central_angle=120,angle_offset=-angle_offset*0.8,connection_width=connection_width,color=colors[1])

    # add cross connections
    # EE out
    _add_external_connection(position1, position2, radius, 40, -40, dotsize, dot_offset, connection_width, colors[0], dashes,-0.3)
    _add_external_connection(position2, position1, radius, -55, 55, dotsize, dot_offset, connection_width, colors[0], dashes,0.3)
    
    # II out
    _add_external_connection(position2, position1, radius, -152, 152, dotsize, dot_offset, connection_width, colors[1], dashes,-0.3)
    _add_external_connection(position1, position2, radius, 140, -140, dotsize, dot_offset, connection_width, colors[1], dashes,0.3)
    
    # EI

    _add_external_connection(position2, position1, radius, -133, 100, dotsize, dot_offset, connection_width, colors[1], dashes,-0.3)
    _add_external_connection(position1, position2, radius, 133, -100, dotsize, dot_offset, connection_width, colors[1], dashes,0.3)
    

    # IE

    _add_external_connection(position1, position2, radius, 110, -123, dotsize, dot_offset, connection_width, colors[0], dashes,0.2)
    _add_external_connection(position2, position1, radius, -110, 123, dotsize, dot_offset, connection_width, colors[0], dashes,-0.2)
  


def draw_detail_box(positions,target,lw=1,dashes=(3,2),color = (0,0,0),radius=11,offset = 2):
    box_bottom = positions[:,1].min(axis=0)-radius-offset
    box_top = positions[:,1].max(axis=0)+radius+offset
    box_left = positions[:,0].min(axis=0)-radius-offset
    box_right = positions[:,0].max(axis=0)+radius+offset

    box_x = [box_left,box_right,box_right,box_left,box_left]
    box_y = [box_bottom,box_bottom,box_top,box_top,box_bottom]
    
    pylab.plot(box_x,box_y,lw = lw,dashes=dashes,color = color)


    pylab.gca().annotate('', xy=target, xycoords='data',
                xytext=[0.5*(box_right+box_left),box_top], textcoords='data',
                size=8,
                # bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="-|>",
                                fc=color, ec=color,
                                connectionstyle="arc3,rad="+str(-0.8),shrinkB=0,shrinkA =0,lw = lw),zorder = -1,
                )

def _line_with_half_circle(start,end,line_args = {},dot_args={},line_z=-1,patch_size = 15):
    
    pylab.plot([start[0],end[0]],[start[1],end[1]],zorder =line_z,**line_args)
    pylab.plot(end[0],end[1],'o',zorder = line_z,**dot_args)
    rect = Rectangle([end[0]+0.1*patch_size,end[1]-0.5*patch_size], patch_size, patch_size,fc = (1,1,1),ec = 'none',zorder = line_z+1)
    pylab.gca().add_patch(rect)

def draw_legend(position,linelength,hspace = 0.4,vspace =0.3, strings = ['excitatory','inhibitory','within','across'],plotargs = [{},{},{},{}],dot_args = [{},{},{},{}],textargs = {}):
    
    l1_start = pylab.array(position)
    l1_end = l1_start.copy()
    l1_end[0]+= linelength
    
    _line_with_half_circle(l1_start, l1_end,plotargs[0],dot_args[0])
    pylab.text(l1_end[0]+hspace * linelength,l1_end[1],strings[0],ha  ='left',va = 'center',**textargs)
    
    l2_start =l1_start.copy()
    l2_start[1] -= linelength*vspace
    l2_end = l2_start.copy()
    l2_end[0]+= linelength

    _line_with_half_circle(l2_start, l2_end,plotargs[1],dot_args[1])
    pylab.text(l2_end[0]+hspace * linelength,l2_end[1],strings[1],ha  ='left',va = 'center',**textargs)
    
    l3_start =l2_start.copy()
    l3_start[1] -= linelength*vspace
    l3_end = l3_start.copy()
    l3_end[0]+= linelength

    _line_with_half_circle(l3_start, l3_end,plotargs[2],dot_args[2])
    pylab.text(l3_end[0]+hspace * linelength,l3_end[1],strings[2],ha  ='left',va = 'center',**textargs)
    

    l4_start =l3_start.copy()
    l4_start[1] -= linelength*vspace
    l4_end = l4_start.copy()
    l4_end[0]+= linelength

    _line_with_half_circle(l4_start, l4_end,plotargs[3],dot_args[3])
    pylab.text(l4_end[0]+hspace * linelength,l4_end[1],strings[3],ha  ='left',va = 'center',**textargs)


def draw_EI_schematic(Q = 20,randseed = 2,detail_radius=60,detail_distance = 160,detail_position = [190,95],network_linewidth = 0.4,network_gap = 0.1,network_connection_lw = 0.2,network_radius = 11,
                   detail_lw = 1,detail_connection_lw = 0.6,detail_dotsize = 2,connection_dashes = (0,(2,1.5)),fraction  =0.25,detail_gap = 0.05,network_connection_alpha = 0.1,detail_box_offset = 5,
                   detail_box_lw = 0.5,detail_box_dashes = (1,0.8),detail_box_color=yellow,E_color = (0,0,0),I_color = red,legend_fontsize = 6):
    
    network_positions = draw_network(n=Q,randseed = randseed,lw= network_linewidth,connection_lw=network_connection_lw,fraction  =fraction,gap  =network_gap,connection_alpha = network_connection_alpha,radius = network_radius)
    
    draw_detail(position = detail_position,radius = detail_radius,distance = detail_distance,lw = detail_lw,connection_width=detail_connection_lw,
                dotsize = detail_dotsize,dashes = connection_dashes,fraction  =fraction,gap = detail_gap)
    
    pylab.axis('equal')
    
    pylab.ylim(-140,200)
    pylab.xlim(-120,380)

    
    
    top_right = pylab.argmax((network_positions).sum(axis=1))
    
    top_right_pos = pylab.tile(network_positions[top_right][None,:], (network_positions.shape[0],1))
    
    distances = pylab.sqrt(((network_positions-top_right_pos)**2).sum(axis=1))

    order =  pylab.argsort(distances)
    nearest = order[1]

    box_positions = network_positions[[top_right,nearest]]

    detail_target = pylab.array(detail_position).copy()
    detail_target[0] -= detail_radius
    detail_target[1] += detail_radius*0.8
    draw_detail_box(box_positions,detail_target,radius = network_radius,offset = detail_box_offset,lw =detail_box_lw,color = detail_box_color)
    
    legend_position = detail_position
    legend_position[0] -= detail_radius * 0.6
    legend_position[1] -= detail_radius * 1.7
    
    third_color = [0.3]*3
    line_args = [{'lw':detail_connection_lw,'c':E_color},{'lw':detail_connection_lw,'c':I_color},{'lw':detail_connection_lw,'c':third_color},{'lw':detail_connection_lw,'c':third_color,'linestyle':connection_dashes}]
    dot_args = [{'c':E_color,'mec':E_color,'markersize':detail_dotsize},{'c':I_color,'mec':I_color,'markersize':detail_dotsize},{'c':third_color,'mec':third_color,'markersize':detail_dotsize},{'c':third_color,'mec':third_color,'markersize':detail_dotsize}]
    text_args = {'size':legend_fontsize}
    draw_legend(legend_position, detail_radius*0.8,plotargs = line_args,dot_args=dot_args,textargs=text_args,vspace= 0.6)


def draw_EE_network(Q = 20,randseed = 3,network_linewidth = 0.4,network_connection_lw = 0.2,network_radius = 11,network_connection_alpha = 0.1,I_color = red,legend_fontsize = 6,I_radius = 30,I_position = [-90,-30],x_offset = 0,y_offset  =0,equal_ax  =True,I_lw = 0.3):
    if equal_ax:

        pylab.axis('equal')
    network_positions = draw_network(n = Q,randseed = randseed,lw= network_linewidth,connection_lw=network_connection_lw,fraction  =0.,gap  =0.,connection_alpha = network_connection_alpha,radius = network_radius,x_offset = x_offset,y_offset = y_offset)
   
    _draw_subnetwork(pos = I_position+pylab.array([x_offset,y_offset]),fraction  = 0. ,radius=I_radius,orientation=0,colors = [I_color,(1,1,1)],strings = ['',''],lw = I_lw,gap = 0.00,text = False,fontsize = 6)
    for pos in network_positions:
        pylab.plot([pos[0],I_position[0]+x_offset],[pos[1],I_position[1]+y_offset],(0,0,0),lw = network_connection_lw,alpha = network_connection_alpha,zorder = -10)

def draw_EI_network(Q = 20,randseed = 3,network_linewidth = 0.4,network_connection_lw = 0.2,network_radius = 11,network_connection_alpha = 0.1,legend_fontsize = 6,
                    network_gap = 0.1,fraction = 0.25,x_offset = 0,y_offset  =0,equal_ax  =True):
    
    if equal_ax:

        pylab.axis('equal')
    network_positions = draw_network(n = Q,randseed = randseed,lw= network_linewidth,connection_lw=network_connection_lw,fraction  =fraction,gap  =network_gap,connection_alpha = network_connection_alpha,radius = network_radius,x_offset = x_offset,y_offset = y_offset)
   
    
    
    

def schematic_2d_model(ax, net_factor = 1.6,net_offset = 1.,
               offset_dir = [1.2,0.4],net_sigma = 0.12,N = 1200,
               decoder_factor = 0.7,decoder_offset = 1,randseed=0,
               colors = None):
    off_color = '0.8'
    green = '0.5'
    red = '0.3'
    pylab.seed(randseed)

    stim_center = [0,0]
    stim_size  =0.3
    if colors is None:
        colors = [red]+5*[off_color]
    circs,coords = draw_hex_array(stim_center,size  =stim_size,colors =colors)

    for circ in circs:
        circ.set_zorder(4)

    Q = len(circs)
    net_positions = []

    cluster_centers = []
    positions = pylab.zeros((0,2))
    
    for coord in coords:
        new_pos =pylab.randn(N//Q,2)*net_sigma
        new_pos[:,0] += coord[0]*net_factor + net_offset*offset_dir[0]
        new_pos[:,1] += coord[1]*net_factor + net_offset*offset_dir[1]
        
        cluster_centers.append((coord[0]*net_factor + net_offset*offset_dir[0],coord[1]*net_factor + net_offset*offset_dir[1]))
        positions = pylab.append(positions, new_pos,axis=0)
    
    ax.plot(positions[:,0],positions[:,1],'ok',ms = 0.2,zorder = 2 )
    
    connectivity = pylab.rand(N,N)<0.006

    connections = []
    for i in range(N):
        js = find(connectivity[i])
        for j in js:
            connections.append([positions[i],positions[j]])
    
    color = [0.3]*3 + [0.05]
    colors = [color]*len(connections)

    lines = LineCollection(connections,colors = colors,linewidths =0.04,zorder =2)
    
    ax.add_collection(lines)
    
    for i in range(6):
        ax.annotate('', xy=cluster_centers[i], xycoords='data',
                xytext=coords[i], textcoords='data',
                size=20,
                # bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="simple,tail_width=0.06,head_width=0.15",
                                fc=[0,0,0,1.], ec="none",
                                connectionstyle="arc3,rad=-0.3"),zorder = 3
                )


    cluster_center_center = pylab.array(cluster_centers).mean(axis=0)
    
    edge_length = stim_size * decoder_factor

    decoder_center = pylab.array((cluster_center_center[0] + decoder_offset*offset_dir[0],cluster_center_center[1] + decoder_offset*offset_dir[1]))
    rectangle = Rectangle(decoder_center-0.5*edge_length, edge_length, edge_length,ec ='k',fc ='w')

    for i in range(6):
        ax.annotate('', xy=decoder_center, xycoords='data',
                xytext=cluster_centers[i], textcoords='data',
                size=20,
                # bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="-",
                                fc=[0,0,0,1], ec="k",
                                connectionstyle="arc3,rad=0.3",shrinkB=0),zorder = -1,
                )


    rectangle.set_zorder(0)
    ax.add_patch(rectangle)
    pylab.text(stim_center[0],stim_center[1]-stim_size*1.5,'stimulus',ha = 'center',va = 'top',size = '6')
    pylab.text(decoder_center[0],decoder_center[1]+edge_length,'decoder',ha = 'center',va ='bottom',size = '6')
    pylab.axis('equal')
    
    
    
def draw_hex_array(center,size=0.3,colors = [[0.5]*3]*6,axes = None,radius = 0.1,add = True,show_numbers = False):
    angles = pylab.array([30,90,150,210,270,330])*pylab.pi/180
    Y = size*pylab.cos(angles)
    X = size*pylab.sin(angles)
    
    
    i = 0
    circs= []
    coords = []
    number = 6
    for x,y in zip(X,Y):
        coords.append((x+center[0],y+center[1]))
        circ = pylab.Circle((x+center[0],y+center[1]), radius=radius,  fc=colors[i])
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
    return circs,coords
if __name__ == '__main__':
    


    #ax = pylab.subplot(1,1,1)
    
    
    
    
    draw_EE_network(randseed  =2,Q = 50,I_position = [-180,-50],I_radius = 50,equal_ax =False)

    draw_EI_network(randseed  =3,Q = 50,x_offset = 500,equal_ax =False)

    
    pylab.axis('equal')
    pylab.xlim(-200,1200)


    
    
    pylab.show()

    

