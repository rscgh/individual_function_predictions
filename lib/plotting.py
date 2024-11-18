
class bc: 
  cyan='\033[96m'; green='\033[92m'; end='\033[0m'; red='\033[91m'; under='\033[4m';
  pink='\033[95m'; blue='\033[94m'; yellow='\033[93m'; bold='\033[1m'


import brainspace, os
from nilearn import plotting
stub = os.path.dirname(brainspace.__file__) + "/datasets/surfaces/"

# new even more general version
def plot_gen_surf2(data, surfs = None, transform=None, title="", \
      plot_conf = ["img1_0_180_6.4","img1_0_0_6.4", "img2_0_0_6.4", "img2_0_180_6.4"], suptitles =None,\
      figsize=(19,6), tsize="xx-large", tweight="bold", threshold=1e-14, cmap= "viridis", auto_sym=False, show_fig=True, tight=True, **kwargs):
    # import hcp_utils and from nilearn import plotting
    # data
    datad= {}
    if isinstance(data,dict): datad = data;
    elif isinstance(data, list):
      datad = {"img"+str(i+1): data[i] for i in range(len(data))}
    else: 
      # normally should be a single numpy.ndarray in this case
      datad = {"img1": data};
    
    # data transformations
    if transform in ["HCP29to32k", "to32k"]:
      transform = {"img1": hcp.left_cortex_data, "img2": hcp.right_cortex_data}
    if transform in ["HCP54kto32kLR"]:
      transform = {"img1" : lambda img : hcp.left_cortex_data(img[hcp.struct.cortex_left]), \
                   "img2" : lambda img : hcp.right_cortex_data(img[hcp.struct.cortex_right])}
    elif transform is None:
      transform = {k:None for k in datad.keys()}
    
    #print(datad)#print(transform)
    
    # surfaces
    default_conte = {'img1': stub+"conte69_32k_lh.gii" , 'img2' : stub+"conte69_32k_rh.gii" }
    surfs = default_conte if surfs is None else surfs;
    if isinstance(surfs, list):
      surfs = {"img"+str(i+1): surfs[i] for i in range(len(surfs))}
    elif not isinstance(surfs, dict):
      surfs = {"img1" : surfs};
    
    # plotting
    n_plots = len(plot_conf);
    fig, ax = plt.subplots(1,n_plots, figsize=figsize, subplot_kw={'projection': '3d'})
    
    if not isinstance(cmap,list):
      cmap= [cmap]*n_plots;
    
    for  n, pcf in enumerate(plot_conf):
      imgid, elev, azim, dist = pcf.split("_");
      elev = int(elev); azim = int(azim); dist=float(dist);
      
      #print(imgid, flush=1)
      can_transf = not(transform[imgid] is None)
      t_img_data = transform[imgid](datad[imgid]) if can_transf else datad[imgid];
     
      #plotting.plot_surf_roi(pial[hem], roi_map=labels[hem], cmap="viridis", hemi=hem, \
      #  hemi is normally used to set elev and azim; as we manually set it, this can be neglected
      if auto_sym:
        v = np.absolute(t_img_data).max()
        kwargs.update(vmin=-v, vmax=v)
      plotting.plot_surf_roi(surfs[imgid], roi_map=t_img_data, cmap=cmap[n], hemi="left", \
            view="medial",darkness=.5, axes=ax[n], threshold=threshold, **kwargs);
      ax[n].view_init(elev, azim)
      ax[n].dist = dist
      if not(suptitles is None):
        ax[n].set_title(suptitles[n])

    y_title_pos = ax[0].get_position().get_points()[1][1]+(1/1)*0.05
    fig.suptitle(title, y=y_title_pos, size=tsize, weight=tweight)

    if tight: fig.tight_layout();
    if show_fig: fig.show()
    return fig


# this is still in mm
def plot_sep_colorbar(vmin=0, vmax=1, cmap = "viridis", orientation = "horizontal", path = None, caspect=20, cshrink=1, cfraction=0.15):
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
  img = plt.imshow(np.array([[0,1]]), aspect=0.000000000001)
  plt.gca().axis('off');
  cb = plt.colorbar(sm, orientation=orientation, aspect=caspect, fraction=cfraction, shrink=cshrink);
  if not(path is None):
     plt.savefig(path)
  return cb
        


def centered_minmax(data, p = 1):
  vmax = np.max((np.absolute(data.min()),np.absolute(data.max())))
  return -vmax*p, vmax*p;      

import sys, psutil, os
import numpy as np;
import hcp_utils as hcp

def plot_29k(data, colorbar=True, center_min_max=1,  saveas = None, **kwargs):
  plot_conf = dict(transform="to32k", plot_conf = ["img1_0_180_6.4","img1_0_0_6.4"], cbar=1, figsize=(5,3), \
    tsize="large", cmap="coolwarm", threshold=None)
  if center_min_max: 
    vmin, vmax=centered_minmax(data)
    plot_conf.update(dict(vmin=vmin, vmax=vmax))
    
  plot_conf.update(kwargs) 
  tdata = data[np.absolute(data)>plot_conf["threshold"]] if not(plot_conf["threshold"] is None) else data
  if not("vmin" in plot_conf.keys()): plot_conf["vmin"] = tdata.min()
  if not("vmax" in plot_conf.keys()): plot_conf["vmax"] = tdata.max()   
  #print(plot_conf)
  plot_gen_surf2([data], **plot_conf); #
  if not(saveas is None):
    plt.savefig(saveas, transparent=True)
    
  
  if colorbar: 
    plt.show() #flush axis
    plot_sep_colorbar(plot_conf["vmin"],plot_conf["vmax"],cshrink=1, cmap=plot_conf["cmap"], caspect=20); # , cfraction=0.03, csh =0.4
    #plot_sep_colorbar(-4,4,cshrink=0.2, cmap="coolwarm", caspect=4.5, cfraction=0.06);
    plt.tight_layout()
    if not(saveas is None):
      ext = saveas.split(".")[-1]
      plt.savefig(saveas[:-(len(ext)+1)] + ".cbar.svg" , transparent=True)


from brainspace.utils.parcellation import map_to_labels

def plot_parcellated_data(data, parcell_data, invalid=-1, fill=-1, **kwargs):#
  
  fdata=map_to_labels(data, parcell_data, mask=parcell_data!=invalid, fill=fill)
  return plot_29k(fdata,**kwargs);



############# Make plotting nicer for publication #######################
############# Use PIL to create more complicated figures #######################


from copy import deepcopy
import io
import PIL

def fig2tightPIL(fig = None, crop=True, close_fig=False, **kwargs):
  if fig is None: fig = plt.gcf();
  buf = io.BytesIO();
  save_kwargs = dict(dpi = 100, bbox_inches='tight', pad_inches=0, transparent="True",format='png')
  save_kwargs.update(kwargs)
  fig.savefig(buf, **save_kwargs);
  buf.seek(0);
  pil_img = deepcopy(PIL.Image.open(buf));
  buf.close();
  if crop:
    imageBox = pil_img.getbbox()
    pil_img = pil_img.crop(imageBox)
  if close_fig: plt.close(fig)
    
  return pil_img

def infer_colorbar(data, vmin=None, vmax=None, cminmax=False, cmap = "viridis", **kwargs):
  data = np.array(data)
  if cminmax:
     vmax = np.max((np.absolute(data.min()),np.absolute(data.max())))
     vmin = -vmax
    
  if vmin is None: vmin = data.min()
  if vmax is None: vmax = data.max()
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
  img = plt.imshow(np.array([[0,1]]), aspect=0.000000001)
  plt.gca().axis('off');
  cbar_kwargs = dict(orientation = "horizontal", aspect=20, shrink=1, fraction=0.15)
  cbar_kwargs.update(kwargs)
  cb = plt.colorbar(sm, **cbar_kwargs);
  
  return cb

def paste_centrally(img, paste, lc=0, uc=0):
  w, h = img.size
  sw, sh = paste.size
  l = int((w/2 + lc * (w/2))-(0.5*sw))
  up =int((h/2 + uc * (h/2)) -(0.5*sh))
  pi = img.copy()
  pi.paste(paste, (l, up))
  return pi

def image_grid(imgs, rows, cols, pad=0.01):
    assert len(imgs) <= rows*cols

    w, h = imgs[0].size
    pw,ph = int(w*pad), int(h*pad);
    w, h = w+pw, h+ph;
    grid = PIL.Image.new('RGBA', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        #grid.paste(img, box=(i%cols*w, i//cols*h))
        grid.paste(img, box=((i%cols*w)+pw, (i//cols*h)+ph))
    return grid


"""
# Usage:

plot_29k(mean_maps[6], title=None, cmap='cmr.wildfire', colorbar=False);
#plt.title(task_names[6],  y=1.0, x=0.1, pad=-19, loc='right', weight='bold' )
plt.title(task_names[4],  y=1.05, x=-0.99, pad=0, ha='left', va='top', weight='bold')
pil_img = fig2tightPIL(close_fig=1, dpi=300)
cb = infer_colorbar(mean_maps[6], cminmax=1, cmap='cmr.wildfire', aspect=10, shrink=0.2);
cbimg = fig2tightPIL(cb.ax.get_figure(), dpi=300, format="png", close_fig=1)
img = paste_centrally(pil_img, cbimg, uc=0.8);
#image_grid([img,img,img,img,img,img], 3,2).save('out.png')

image_grid([img]*3, 1,3)

"""
        
############# BarPlots ETC. #######################

from matplotlib import pyplot as plt

"""
def plot_bars(list_of_items, bar_labels=None, xlabels=None, plot_type="plt", fig_kwargs={}, ylim = None, **kwargs):
  bar_kwargs=dict(alpha=.5)
  bar_kwargs.update(kwargs)
  if plot_type=="plt":
    for item in list_of_items: 
      plt.bar(np.arange(len(item)), item, **bar_kwargs)
    if not(ylim is None): plt.ylim(*ylim)
    if not(bar_labels is None):plt.legend(bar_labels)
    if not(xlabels is None):
        plt.xticks(np.arange(len(xlabels)), xlabels, ma="right", rotation=90);
  return plt.gcf()"""


def plot_bars(list_of_items, bar_labels=None, xlabels=None, plot_type="plt", \
              side_by_side=False, redby=0.3, ylim = None, ax=None, cmap="viridis",
              hatches = None, rot = 90, **kwargs):
  bar_kwargs=dict(alpha=.5)
  bar_kwargs.update(kwargs)
  if plot_type=="plt":
    if ax is None:
      ax = plt.figure().gca()
    n_items = len(list_of_items)
    if side_by_side:
      bar_kwargs["width"]=(1-redby)/n_items
    
    if isinstance(cmap, str): 
       cmap = plt.get_cmap(cmap)
    #rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    
    if hatches is None:
        hatches = [None] * n_items
    
    for i, item in enumerate(list_of_items): 
      if side_by_side:
        fw = (1-redby);
        width=fw/n_items;
        hw = 0.5*width;
        bar_kwargs["width"]=width
        #ax.bar(np.arange(len(item))-(0.5)+0.5*width+i*width, item, hatch = hatches[i], color=cmap((i+1)/(n_items+1)), **bar_kwargs)
        ax.bar(np.arange(len(item))-(0.5*fw)+(i+0.5)*width, item, hatch = hatches[i], color=cmap((i+1)/(n_items+1)), **bar_kwargs)
      else:
        ax.bar(np.arange(len(item)), item,  hatch = hatches[i],  color=cmap((i+1)/(n_items+1)), **bar_kwargs)
    
    
    if not(ylim is None): ax.set_ylim(*ylim)
    if not(bar_labels is None):ax.legend(bar_labels)
    if not(xlabels is None):
        #ax.set_xticks(np.arange(len(xlabels)), xlabels, ma="right", rotation=90);
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, ma="right", rotation=rot);
        
  return ax


plt.figure();
plt.plot([1,2,3,4],[1,2,3,4], "r")
plt.plot([1,2,3,4],[0,1,2,3], "b")
leg = plt.legend(["test-restest baseline", "group average baseline"])
basic_lines = leg.get_lines();
plt.close();


def plot_corr_scores(cdict=None, items = None, ylabel="Pearson Correlation (r)", labels = None, keys= None, task_names = None, selected_ids = None, sc_retest = None, sc_group = None, side_by_side=1, **kwargs):
  assert not(cdict is None) or not(items is None) # needs either of the inputs
  assert not(not(cdict is None) and not(items is None)) # cant be given both
  
  if not(cdict is None):
    #items = [(v.corr.mean(0) if isinstance(v, dict) else v) for k,v in cdict.items()]
    if (keys is None): keys = cdict.keys()
    items = [(cdict[k].corr.mean(0) if isinstance(cdict[k], dict) else cdict[k]) for k in keys]
    if labels is None: labels =  list(keys)
  
  if not(selected_ids is None):
    print(np.array(items).shape)
    items = np.array(items)[:, selected_ids]
  
  ax = plot_bars(items, labels, xlabels = task_names , side_by_side=side_by_side, **kwargs)
  patches = ax.get_legend().get_patches()
  
  from matplotlib.lines import Line2D
  legend_labels = labels;
  redby = kwargs["redby"] if "redby" in kwargs.keys() else 0.3;
  hwidth=(1-redby)*0.5
  if not(sc_retest is None): 
    for i, v in enumerate(sc_retest): ax.plot(np.linspace(i-hwidth,i+hwidth,10), [v]*10, "r")
    legend_labels = legend_labels + ["test-restest baseline"]
    patches.append(basic_lines[0]);
  if not(sc_group is None): 
    for i, v in enumerate(sc_group): ax.plot(np.linspace(i-hwidth,i+hwidth,10), [v]*10, "b")
    legend_labels = legend_labels + ["group average baseline"]
    patches.append(basic_lines[1]);
  
  ax.set_ylabel(ylabel); ax.set_xlabel("Task Contrasts"); ax.set_ylim(0)
  ax.legend(patches, legend_labels, fontsize="small")
  return ax;

"""
# Usage:

items = [v for k,v in corr_scores.items()]
labels =  [k for k,v in corr_scores.items()]
fig = plot_bars(items, labels, \
                xlabels = task_names2,side_by_side=1)#, color = plt.get_cmap("tab20").colors); 
plt.set_cmap('viridis')

from matplotlib.lines import Line2D
for i, v in enumerate(sc_retest): fig.gca().plot(np.linspace(i-0.38,i+0.38,10), [v]*10, "r")
for i, v in enumerate(sc_gr): fig.gca().plot(np.linspace(i-0.38,i+0.38,10), [v]*10, "b")
plt.ylabel("Pearson Correlation (r)"); plt.xlabel("Task Contrasts"); plt.ylim(0)
patches = plt.gca().get_legend().get_patches()

# handles is a list, so append manual patch
patches.append(lines[0]); patches.append(lines[1]);
plt.legend(patches, labels + ["test-restest baseline", "group average baseline"], fontsize="small")
plt.savefig("results/plots/ohbm_poster/raw_correlations1.svg",bbox_inches = "tight", transparent=True)
plt.savefig("results/plots/ohbm_poster/raw_correlations1.png",bbox_inches = "tight", transparent=True)

""";




"""import pandas as pd

sidx = np.argsort(sd_av_varxp_test)[::-1]

resxx = np.zeros((3,len(task_names)))
resxx[0]= bp_av_varxp_test[sidx]
resxx[1]= sd_av_varxp_test[sidx]
resxx[2]= comb_av_varxp_test[sidx]
vnames= ["blueprint", "surface_dist","combined"]; 

resdf = pd.DataFrame(resxx, columns = [t.lower() for t in np.array(task_names)[sidx]], index = vnames).reset_index().rename(columns={"index":"model"})
resdf = pd.melt(resdf, id_vars=['model'], var_name='task', value_name='r2');
resdf[:5]"""

"""import seaborn as sb
ax = sb.barplot(data=resdf, x="r2", y="task", hue="model", palette="copper") #copper
#ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.xaxis.grid(color='gray', linestyle='dashed')

plt.gcf().set_size_inches(7, 6.5, forward=True)
ax.set_xlabel('totalCount')
ax.set_xlim(0, resxx.flatten().max()*1.05)
ax.set_ylabel(None)
ax.set_xlabel("$r^2$")
ax.tick_params(axis='both', which='major', labelsize=13)
plt.legend(loc=4, prop={'size': 13}) #alternative location 1 (rtop) or 4 (rb)
plt.savefig(f"outp/img/barplot_nn+fulllin.explained.svg");"""


#https://github.com/TheChymera/chr-helpers/blob/d05eec9e42ab8c91ceb4b4dcc9405d38b7aed675/chr_matplotlib.py
#https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib/56699813#56699813

#__author__="Paul H, Horea Christian"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, 
name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave 
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax) 
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin)) 
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave 
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin)) 
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False), 
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


# from lib.plotting import remappedColorMap
# cmap =  matplotlib.pyplot.get_cmap(name=None, lut=None)[
# cmap = remappedColorMap(