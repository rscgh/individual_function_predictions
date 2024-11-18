
import torch
import os
import numpy as np

import hcp_utils as hcp
import scipy.stats

from torch.utils import data as tdata


import scipy.stats
from tqdm import tqdm
from lib.stats import batch_resid 


########## Newer data loading function ##########
########## allowing for model combination ##########


def load_xy_data(subjs, data_dir, predictor_data, task_data, resid_task_maps=False, norm_pred=False, zscore_tasks=True):
  
  data = predictor_data 
  if isinstance(predictor_data, str):
    fn = data_dir + predictor_data
    data = np.load(fn, allow_pickle=True).item()
    
  # xdata items should be of shape (n_subj, 29696, n_pred)
  xdata = np.array([v for s,v in data.items()])# if s in train_subj_ids]) 
    
  if not(norm_pred==False):
    # if we want to additionally normalize our x-data, e.g. for usage with ridge regression
    orig_shape = xdata.shape
    xdata = scipy.stats.zscore(xdata.reshape(-1, xdata.shape[-1]), axis=norm_pred).reshape(orig_shape);

  included_subjs = [s for s,v in data.items() if s in subjs]
  missing_subjs = [s for s in subjs if not (s in included_subjs)]
  
  print(len(missing_subjs), "missing subjects: ", missing_subjs)
                 
  data = task_data 
  if isinstance(task_data, str):
    fn = data_dir + task_data
    data = np.load(fn, allow_pickle=True).item()
  
  # ydata items should be of shape (n_subj, 29696, n_tasks)
  #??training_set.ensure_z_score_task_maps // we always z-score
  ydata=None
  if zscore_tasks:
    print("z-scoring task maps along the spatial dimension")
    ydata = {subj: scipy.stats.zscore(v, axis=0) for subj, v in data.items() if subj in included_subjs} 
  else:
    ydata = {subj: v for subj, v in data.items() if subj in included_subjs} 
  assert np.all([s in data.keys() for s in included_subjs])
  print(f"All {len(included_subjs)} have task data attached to them")
  ydata = np.array([v for s,v in ydata.items()])# if s in train_subj_ids])
  if resid_task_maps: 
    from lib.stats import batch_resid    
    print("Residualizing task maps ...");
    #mean_task_maps = np.load("results/retest_hcp45.test.mean_task_maps.npy", allow_pickle=1).item()["mean_maps"]#["task_names"];
    mean_task_maps = ydata.mean(0).T; # (10, 29696)
    ydata = np.swapaxes(batch_resid(np.swapaxes(ydata,1,2), mean_task_maps),1,2)
  
  # make sure xdata "variable" columns are standartized [mean=0, var=1]; zscore_predictors kind of does that ...
  # https://stats.stackexchange.com/questions/111017/question-about-standardizing-in-ridge-regression
  return included_subjs, xdata, ydata


# combination of modalities

# before combine_modalities
def gather_modalities(subjs, pred_modalities, pred_files, data_dir = "/scratch/users/robert.scholz2/acc_dists/", zscore_predictors=False, save_as=None, modality_cache=None, v= True):
  combined = {}; count = 0;
  #load = m,c : c[m] if m in c.keys() else np.load(f"{data_dir}{pred_files[m]}")
  #datasets = [np.load(f"{data_dir}{pred_files[modality]}", allow_pickle=1).item() for modality in pred_modalities]
  datasets = []
  for modality in pred_modalities:
    if not(modality_cache is None):
      if modality in modality_cache.keys(): 
        if v: print("Loading from memory:", modality)
        datasets.append(modality_cache[modality])
      else:
        data = np.load(f"{data_dir}{pred_files[modality]}", allow_pickle=1).item()
        if zscore_predictors: data = {subj: scipy.stats.zscore(data[subj], axis=0) for subj in tqdm(data.keys(), desc="zsc-pred")}
        modality_cache[modality] = data;
        datasets.append(data)
    else: 
      data = np.load(f"{data_dir}{pred_files[modality]}", allow_pickle=1).item()
      if zscore_predictors: data = {subj: scipy.stats.zscore(data[subj], axis=0) for subj in tqdm(data.keys(), desc="zsc-pred")}
      datasets.append(data)
  #if not(modality_cache is None) and modality in modality_cache.keys(): 
  #  return modality_cache[modality];
  
  for subj in tqdm(subjs, desc="Concatenating predictors"):  
    if False in [subj in ds.keys() for ds in datasets]: continue; 
    count= count+1;
    combined[subj] = np.concatenate([ds[subj] for ds in datasets],axis=1)
  
  lsubj= list(combined.keys())[-1]
  if v: print(f"Gathered data from {count} subjects. Per subject data is of shape: {combined[lsubj].shape}")
  
  """if not(modality_cache is None):
    modality_cache[modality] = combined""";
    
  if not(save_as is None):
    fn = f"/scratch/users/robert.scholz2/acc_dists/all_combined_{save_as}_{count}_full_unrelated.L.npy"
    np.save(fn, combined)
    print(f"Saved as {fn}")
    #!ls -ash {fn}
  return combined;





########## Old task loading function ##########


def load_task_data_only(H, parcell, exp_id, n_subjs, z_score_task_maps = False):
  acc_dir = "../../bigdata/acc_dists/"
  scratch1= "/scratch/users/robert.scholz2/acc_dists/"
  ## Get task info
  fp = scratch1 + f"subj_task_maps_{exp_id}_{n_subjs}_info.npy";
  if not os.path.exists(fp):
    fp = acc_dir + f"subj_task_maps_{exp_id}_{n_subjs}_info.npy";
  
  print("Load", fp);
  itask_info = np.load(fp, allow_pickle=1).item()
  task_names = itask_info["task_info"].split("\n");
    
  fp=acc_dir + f"subj_task_maps_{exp_id}_{n_subjs}_LR54k.npy"
  print("Load", fp);
  sta = np.load(fp);
  if z_score_task_maps:
    sta=scipy.stats.zscore(sta, axis=2)
  print("subj_task_maps:", sta.shape, np.round(sta.nbytes * 0.000001,2), "MB")
  return (sta, task_names, itask_info)

from scipy.stats import zscore

def load_data_for_analysis(H, parcell, exp_id, n_subjs, remove=[], z_score_task_maps = False):
  #global H, parcell, exp_id, n_subjs, n_tasks;

  acc_dir = "../../bigdata/acc_dists/"
  scratch1= "/scratch/users/robert.scholz2/acc_dists/"
  
  ########## Loading Distance Metadata ##########
  fp = scratch1 + f"subj_distances_{exp_id}_{n_subjs}_29x{parcell}_{H}_info.npy"
  if not os.path.exists(fp):
    fp = acc_dir + f"subj_distances_{exp_id}_{n_subjs}_29x{parcell}_{H}_info.npy"

  print("Load", fp);
  interm_info = np.load(fp, allow_pickle=1).item()
  subjs = interm_info["subjs"]
  dcorrf = interm_info["dcorrf"] if "dcorrf" in interm_info else np.zeros(n_subjs); 

  n_subjs = len(subjs)
  n_regions = interm_info["n_regions"]
    
  ########## Parcellation Information ##########
  regions = interm_info["regions"]
  label_dict = interm_info["label_dict"]
  #print(np.array(subjs))
  
  label_names = [label_dict[r] for r in regions];
  #dbar = {"L_pericalcarine": "V1", "L_postcentral": "S1", "L_transversetemporal": "A1"}
  region_names = [r for r in label_names];
  region_names[:10]
  
  ########## Get the Task Contrasts Metadata ##########
  fp=scratch1 + f"subj_task_maps_{exp_id}_{n_subjs}_info.npy";
  if not os.path.exists(fp):
    fp = acc_dir + f"subj_task_maps_{exp_id}_{n_subjs}_info.npy";
    
  print("Load", fp);
  itask_info = np.load(fp, allow_pickle=1).item()
  task_names = itask_info["task_info"].split("\n");
  n_tasks = len(task_names)
    
  # make sure that subjects in both files are the same
  assert np.all(np.array(itask_info["subjs"]) == subjs)
  
  ########## Loading the task data ##########
  ## load the actual data
  fp=scratch1 + f"subj_task_maps_{exp_id}_{n_subjs}_LR54k.npy"
  if not os.path.exists(fp):
    fp = acc_dir + f"subj_task_maps_{exp_id}_{n_subjs}_LR54k.npy";
  
  print("Load", fp);
  sta = np.load(fp)
  if z_score_task_maps:
    sta=zscore(sta, axis=2)
    
  ########## Loading the distance data ##########
  fp=scratch1 + f"subj_distances_{exp_id}_{n_subjs}_29x{parcell}_{H}.npy"
  if not os.path.exists(fp):
    fp = acc_dir + f"subj_distances_{exp_id}_{n_subjs}_29x{parcell}_{H}.npy"
  
  print("Load", fp);
  sda = np.load(fp)
  print("subj_task_maps:", sta.shape, np.round(sta.nbytes * 0.000001,2), "MB")

  invalid_regions = np.isin(region_names, remove);  
  n_inv_regions = invalid_regions.sum();
  region_names = [r for r in region_names if not(r in remove)];
  sda = sda[:,:,:,~invalid_regions]
  
  print("subj_distance_maps:", sda.shape, np.round(sda.nbytes * 0.000001,2), "MB")
  
  n_regions = n_regions - n_inv_regions;
  print(f"Removed {n_inv_regions} additional regions. In total the subj_distance_maps contains {len(label_dict.keys())-n_regions} less regions than the label_dict is indexing. If nessesary, correct for this.");

  assert sda.shape[0] == n_subjs 
  assert sda.shape[3] == n_regions
  assert sta.shape[1] == n_tasks 
  return (sda, sta, region_names, n_regions, label_dict, dcorrf, subjs, task_names, itask_info)


################### Newer version +##############################
## untested 

from scipy.stats import zscore

def load_task_data(H, parcell, exp_id, n_subjs, z_score_task_maps=False, sdir = "/scratch/users/robert.scholz2/acc_dists/"):
  #### metadata ####
  fp = sdir + f"subj_task_maps_{exp_id}_{n_subjs}_info.npy";
  print("Loading:", fp);
  itask_info = np.load(fp, allow_pickle=1).item()
  task_names = itask_info["task_info"].split("\n");

  #### actual data ####
  fp = sdir + f"subj_task_maps_{exp_id}_{n_subjs}_LR54k.npy"
  print("Loading:", fp);
  sta = np.load(fp); 
  if z_score_task_maps:
    sta=scipy.stats.zscore(sta, axis=2)

  print("subj_task_maps:", sta.shape, np.round(sta.nbytes * 0.000001,2), "MB")
  return (sta, task_names, itask_info)


load_task_data_only = load_task_data; # for backwards compatibility


def load_distance_data(H, parcell, exp_id, n_subjs, remove=[], sdir = "/scratch/users/robert.scholz2/acc_dists/"):
  
  #### metadata ####
  fp = sdir + f"subj_distances_{exp_id}_{n_subjs}_29x{parcell}_{H}_info.npy"
  print("Load", fp);
  interm_info = np.load(fp, allow_pickle=1).item()
  subjs = interm_info["subjs"]
  dcorrf = interm_info["dcorrf"] if "dcorrf" in interm_info else np.zeros(n_subjs); 
  n_subjs = len(subjs)

  # parellation info
  n_regions = interm_info["n_regions"]
  regions = interm_info["regions"]
  label_dict = interm_info["label_dict"]
  label_names = [label_dict[r] for r in regions];
  region_names = [r for r in label_names];

  #### actual data ####
  fp = sdir + f"subj_distances_{exp_id}_{n_subjs}_29x{parcell}_{H}.npy"
  print("Load", fp);
  sda = np.load(fp)

  #### take out invalid regions ###
  invalid_regions = np.isin(region_names, remove);  
  n_inv_regions = invalid_regions.sum();
  region_names = [r for r in region_names if not(r in remove)];
  sda = sda[:,:,:,~invalid_regions]

  print("subj_distance_maps:", sda.shape, np.round(sda.nbytes * 0.000001,2), "MB")
  n_regions = n_regions - n_inv_regions;
  print(f"Removed {n_inv_regions} additional regions. In total the subj_distance_maps contains {len(label_dict.keys())-n_regions} less regions than the label_dict is indexing. If nessesary, correct for this.");
  return sda, subjs, n_regions, dcorrf, label_dict, region_names



def load_data_for_analysis(H, parcell, exp_id, n_subjs, remove=[], z_score_task_maps = False, sdir = "/scratch/users/robert.scholz2/acc_dists/"):

  sta, task_names, itask_info = \
     load_task_data(H, parcell, exp_id, n_subjs, z_score_task_maps=z_score_task_maps)
  sda, subjs, n_regions, dcorrf, label_dict, region_names = \
     load_distance_data(H, parcell, exp_id, n_subjs, remove=remove, sdir = "/scratch/users/robert.scholz2/acc_dists/");

  # just make sure we have correct data
  assert np.all(np.array(itask_info["subjs"]) == subjs)
  assert sda.shape[0] == len(subjs) 
  assert sda.shape[3] == n_regions
  assert sta.shape[1] == len(task_names)
  return (sda, sta, region_names, n_regions, label_dict, dcorrf, subjs, task_names, itask_info)













def data_to_variables(sta, sda, dist_corr = None, verbose=False, struct=hcp.struct.cortex_left, \
                      pred_indiv_var_only = False, dmeasure=-1):
	# sta: subjec task map array, with shape (100, 7, 59412)
	# sda: subject surface distance array, with shape (100, 3, 29696, 224)
	# 100 Subjects, 7 tasks, 3 distance measures,
	# 59k LR vertices, 29K L vertces, 224 parcels in pacellation
	n_parcels = sda.shape[3]; # i.e. 224
	n_tasks = sta.shape[1] # i.e. 7

	## DEPENDENT VARIABLES (1 Task = 1 DVar)
	## we would like to predict vertex level task activations (=y_true)
	# get the task maps for the desired hemisphere
	Y_pre = sta[: ,: , struct] # shape (n_subjs, n_tasks, n_vertices)

	# subtract the mean task maps across subject 
	# from all subject specific task maps, so that what remains are
	# just the the interindividual differences that we then can try to predict
	if pred_indiv_var_only:
		Y_pre = (Y_pre-Y_pre.mean(axis=0))     

	#y_true = np.moveaxis(Y_pre, 1, 2).reshape((-1,n_tasks))    
	# switch task and vertices axis (so task is last),
	Y_pre = np.moveaxis(Y_pre, 1, 2)   # yields shape (n_subjs, n_vertices, n_tasks)
	if verbose: print("Y_pre shape:", Y_pre.shape)
	# then flatten the first two axes, resulting in:
	# Y of shape: (n_subjs*n_vertices, n_tasks), i.e. (2969600, 7)
	Y = Y_pre.reshape((-1,n_tasks))    
	if verbose: print("Y shape:", Y.shape)

	## INDEPENDENT VARIABLES (1 Source Parcel Dist = 1 IVar)
	## we use the distances from certain source parcels
	## to the current vertex on the brain as predictors (=X)

	sda_dc = sda
	if not(dist_corr is None):
	  # move axis 0 describing the subjects to the last position (=3)
	  # then devide by subject specific correction factor: 
	  # dcorrf should be of shape (n_subjs) ~ (sda.shape[0]))
	  # then move axis back to its original position
	  # will be of shape (100, 3, 29696, 224), same as sda
	  sda_dc =  np.moveaxis((np.moveaxis(sda,0,3) / dist_corr ),3,0);

	# get distances as X_pre of 
	# shape (n_subjs, n_vertices, n_parcels), i.e. (100, 29696, 224)
	X_pre = sda_dc[: , dmeasure , struct , :] 
	if verbose: print("X_pre shape:", X_pre.shape)
	# reshape into the same number of samples as contained in y_true 
	# (aka again flatten the first two dimensions), 
	# and number of predictor variables equalling n_parcels
	X = X_pre.reshape((Y.shape[0], n_parcels))        # shape (n_subjs*n_vertices, n_parcels), i.e. (2969600, 224)
	if verbose: print("X shape:", X.shape)

	# returns unflattened [X_pre, Y_pre] and flattened (subject x vertices) arrays [X, Y]
	return X_pre, Y_pre, X, Y;



class DistDataset2(tdata.Dataset):
    
  @classmethod  
  def from_npy_files(cls, distance_file, task_contrast_file, cls_kwargs={}):
    
    return cls(XXX,YYY, **cls_kwargs)
    # date2 = from_npy_files("a.npy", "b.npy")
    
  
  def __init__(self, subject_distances, subject_task_maps, subj_ids=None, float64=False, to32k=False, switch_dims=False, tofloat=False):

    # subject_distances of i,e, shape, 
    #   with the last dimension corresponding to the number of predictor distances
    # subject_task_maps of e.g. shape: 
    
    # batches should be a divisor/multiple of 29696

    # everything is loaded into CPU-assocaited RAM
    self.n_pred = subject_distances.shape[-1]
    self.n_targ = subject_task_maps.shape[1]
    self.n_subjs = subject_distances.shape[0]

    #selfttype = torch.float32 if float32 else torch.float64; # TBD

    self.subj_ids = subj_ids
    
    if to32k:
      self.greyl = hcp.vertex_info["grayl"]
      tmp_dists = np.zeros((subject_distances.shape[0], 32492, subject_distances.shape[2]))
      tmp_dists[:, self.greyl, :] = subject_distances; 
      subject_distances = tmp_dists
      
      tmp_tasks = np.zeros((subject_task_maps.shape[0], 32492, subject_task_maps.shape[2]))
      tmp_tasks[:, self.greyl, :] = subject_task_maps; 
      subject_task_maps = tmp_tasks
      print(subject_distances.dtype)
        
    if tofloat:
      subject_distances = subject_distances.astype(np.float32)
      print(subject_distances.dtype)
      subject_task_maps = subject_task_maps.astype(np.float32)
    
    if switch_dims:
      self.subj_dists = torch.from_numpy(np.moveaxis(subject_distances, -2,-1))
      self.task_maps = torch.from_numpy(np.moveaxis(subject_task_maps, -2,-1))
    else:
      self.subj_dists = torch.from_numpy(subject_distances)
      self.task_maps = torch.from_numpy(subject_task_maps)
        

    
    print(self.subj_dists.shape)
    print(self.task_maps.shape)

    assert(self.subj_dists.shape[0] == self.task_maps.shape[0])
    if not(self.subj_ids is None): 
      assert(self.n_subjs == len(self.subj_ids))
    
  def __len__(self):
    'Denotes the total number of samples'
    return self.n_subjs

  def get_subj(self, sid, numpy=False):
    # if its a "10233"-type id
    if isinstance(sid, str):
      sid = self.subj_ids.index(sid)
    
    x = self.subj_dists[sid];
    y = self.task_maps[sid]
    return (x.detach().numpy(), y.detach().numpy()) if numpy else (x,y);

  def __getitem__(self, index):
    'Generates one sample of data'
    # Load data and get label
    x = self.subj_dists[index]
    y = self.task_maps[index]
    # optionally apply transforms
    return x, y

################### Version 3, allowing for lazy loading ... ##############################
from tqdm import tqdm

class LazyLoadingDistDataset(tdata.Dataset):
  
  def __init__(self, subjs, hcp_folder = "/scratch/users/robert.scholz2/HCP_1200/", dist_type= "centroid_dist", hemisphere= "L", parcellation="cam_laus_08s4", \
               tmap_type="tstat1", smooth_lv = "2", tmsmall = "_MSMAll", \
              contrast_info = None, cache=True, cortex_vertices=hcp.struct.cortex_left, corr_dist=False, z_score_task_maps=True): 
    # unimplemented: to32k=False, switch_dims=False, tofloat=False
    
    # subject_distances of i,e, shape, 
    #   with the last dimension corresponding to the number of predictor distances
    # subject_task_maps of e.g. shape: 
    
    # batches should be a divisor/multiple of 29696
    
    self.n_subjs = len(subjs);
    self.subjs = subjs; 
    self.hcp_folder = hcp_folder;
    
    self.H = hemisphere;
    self.dist_file_pattern = f"%s/T1w/fsaverage_LR32k/%s.{hemisphere}.midthickness.MSMAll.native.surf.29x{parcellation}.{dist_type}.npy"
    self.task_file_pattern = f"%s/MNINonLinear/Results/tfMRI_%s/tfMRI_%s_hp200_s{smooth_lv}_level2{tmsmall}.feat/GrayordinatesStats/cope%s.feat/{tmap_type}.dtseries.nii"
    task_fmri_sam   = task_fmri_sam   = "{subj}/MNINonLinear/Results/tfMRI_{task}/tfMRI_{task}_hp200_s{smooth_lv}_level2{tmsmall}.feat/GrayordinatesStats/cope{cope_num}.feat/{tmap_type}.dtseries.nii";
    
    if contrast_info is None:
       from lib.default_config import contrast_info
    
    self.contrast_info= contrast_info
    self.dist_type=dist_type;
    self.parcellation = parcellation;
    self.z_score_task_maps=z_score_task_maps;
    self.cache = cache;
    
    self.cortex_vertices = cortex_vertices
    
    self.dist_data={}
    self.task_data={}
    self.corr_dist = corr_dist;

    #self.n_pred = subject_distances.shape[-1]
    #self.n_targ = subject_task_maps.shape[1]
  
  def load_subj_task_data_files(self, subj):
    y_b = np.concatenate([nib.load( \
        os.path.join(self.hcp_folder, self.task_file_pattern % (subj, task, task, cope_num ))).get_fdata() \
                    for (task, cope_num, contr) in self.contrast_info])    
    y =  y_b[:, self.cortex_vertices].T;
    if self.z_score_task_maps:
       y=  scipy.stats.zscore(y_b[:, self.cortex_vertices].T, axis=0)
    return y;

  def load_subj_dist_data_file(self, subj):
    dist_path = self.dist_file_pattern % (subj,subj)     
    x = np.load(os.path.join(self.hcp_folder, dist_path))
    if self.corr_dist:
      x=x/x.flatten().mean()
    return x; 

  def ensure_distance_correction(self):
    self.dist_data={subj: v /v.flatten().mean() for subj, v in self.dist_data.items()}
    self.dist_corr = True
    
  def ensure_z_score_task_maps(self):
    #self.dist_data={subj: v /v.flatten().mean() for subj, v in self.dist_data.items()}
    self.z_score_task_maps = True;
    self.task_data = {subj: scipy.stats.zscore(v, axis=0) for subj, v in self.task_data.items()}
    
  def init_data_from_file(self, fn, modality="task", is_raw=True):
    data = np.load(fn, allow_pickle=True).item()
    if len([s for s in self.subjs if not(s in data.keys())]) !=0:
      print("Warning: the loaded file doesnt contain all the subject from the dataset. Maybe this is the wrong file?");
    
    # keep only the relevant subjs in memory
    data = {s:v for s,v in data.items() if s in self.subjs}
    # if is raw, apply transforms ...?
    
    if modality.startswith("task"):
      self.task_data=data  
    elif modality.startswith("dist"):
      self.dist_data=data
    
  def embed_inputs_using_pca(self, embd_obj = None, n_comps=-1):
    from sklearn.decomposition import PCA
    data = np.stack([self.get_subj(subj)[0] for subj in tqdm(self.subjs, desc="loading data")])
    print(data.shape)
    data_flat = data.reshape((-1, data.shape[-1]))
    if embd_obj is None:
       if n_comps == -1:
         n_comps = min(data_flat.shape);
       embd_obj = PCA(n_components=n_comps)
       data_red = embd_obj.fit_transform(data_flat)
    else:
       data_red = embd_obj.transform(data_flat)
    
    data = data_red.reshape(list(data.shape[:-1])+ [data_red.shape[-1]])
    self.dist_data = {subj: data[i] for i, subj in enumerate(self.subjs)}
    self.pca = embd_obj

  def acc_data_to_file(self, fn, modality="task", dtype=np.float32):
    data= None
    idx = 0 if modality.startswith("dist") else 1;
    data = {subj: self.get_subj(subj)[idx].astype(dtype) for subj in tqdm(self.subjs)}
    if not(data is None): np.save(fn, data);
    
  def __len__(self):
    'Denotes the total number of samples'
    return self.n_subjs

  def get_subj(self, subj):
    
    if self.cache and not(subj in self.dist_data.keys()): #and subj in self.task_data.keys():
      #print(subj)
      x = self.load_subj_dist_data_file(subj)
      self.dist_data[subj] = x; 
    
    else:
      x = self.dist_data[subj]
    
    if self.cache and not(subj in self.task_data.keys()):
      y = self.load_subj_task_data_files(subj)
      self.task_data[subj] = y; 
    else:
      y = self.task_data[subj]
    
    return x,y;

  def __getitem__(self, index):
    subj = self.subjs[index]
    'Generates one sample of data'
    return self.get_subj(subj);


################### Version 4, simplified, just loading from npy file ... ##############################
from tqdm import tqdm

class TaskPredictionNPYDataset(tdata.Dataset):
  
  def __init__(self, subjs, predictors_file = None, task_data_file=None, zstasks = False, \
               redsid_tasks = False, for_brainsurfCNN = False): 
    
    # subject_distances of i,e, shape, 
    #   with the last dimension corresponding to the number of predictor distances
    # subject_task_maps of e.g. shape: 
    
    self.n_subjs = len(subjs);
    self.subjs = subjs; 
    self.predictor_data={}
    self.task_data={}
    self.zscore_tasks = zstasks; 
    self.for_brainsurfCNN = for_brainsurfCNN;
    self.redsid_tasks = redsid_tasks;
    
    if not(predictors_file is None):
      self.init_data_from_file(predictors_file, modality="predictor");
    if not(task_data_file is None):
      self.init_data_from_file(task_data_file, modality="task");
    #print("Dataset Initialized.");
  
  def init_data_from_file(self, fn, modality="predictor"):
    #print(self, fn, modality)
    data = np.load(fn, allow_pickle=True).item()
    #print(data.keys())
    # keep only the relevant subjs in memory
    new_subjs = [s for s in self.subjs if s in data.keys()]
    data = {s:data[s] for s in new_subjs} # for s,v in data.items() if s in self.subjs}
    if len(new_subjs)!=len(self.subjs) or (not np.all(np.array(new_subjs)==np.array(self.subjs))):
        print(f"Warning: Subjects reduced to the {len(new_subjs)} contained in the {modality} file" + \
              f"(of previously: {self.n_subjs} subjs)")
        self.subjs = new_subjs; 
        self.n_subjs = len(new_subjs)
    # if is raw, apply transforms ...?
    
    if modality.startswith("task"):
      print(f"init task from {fn}")
      self.task_data_from_file = fn;
      if self.zscore_tasks:
        # data item standard shape is (29696, n_tasks)
        data = {subj: scipy.stats.zscore(v, axis=0) for subj, v in data.items()}
      if self.redsid_tasks:
        data = {subj: batch_resid(np.expand_dims(v.T, axis=0), mean_task_maps)[0].T for subj, v in data.items()}
      if self.for_brainsurfCNN:
        # change shape to (n_tasks, 29696) 
        data = {subj: np.array([hcp.left_cortex_data(m) for m in v.T]) for subj,v in data.items()}; 
        #data = {subj: v.T for subj, v in data.items()}
        
      self.task_data=data;
    
    elif modality.startswith("predictor"):
      print(f"init predictor from {fn}")
      # predictor item standard shape is (29696, n_predictors)
      if self.for_brainsurfCNN:
        # change shape to (n_predictors, 29696)                       
        data = {subj: np.array([hcp.left_cortex_data(m) for m in v.T]) for subj,v in data.items()};                
                
      self.predictor_data=data
      self.predictor_data_from_file = fn;
      
      # if self.distance_correction?
    
  def __len__(self):
    'Denotes the total number of samples'
    return self.n_subjs

  def get_subj(self, subj):
    x = self.predictor_data[subj]
    y = self.task_data[subj]
    
    return x,y;

  def __getitem__(self, index):
    subj = self.subjs[index]
    'Generates one sample of data'
    return self.get_subj(subj);


"""
# Usage:

from lib.DistanceDataLoading import TaskPredictionNPYDataset

dscfg = Bunch(z_score_task_maps = True, n_batch = 80, n_val="max")
dscfg.input_data_dir = "/scratch/users/robert.scholz2/acc_dists/"
dscfg.predictors_file = "all_grads_254_full_unrelated.L.fisherz.commonGroupPCA.npy"
dscfg.task_maps_file = "all_10_tasks_254_full_unrelated.raw.npy"

pred_file = dscfg.input_data_dir + dscfg.task_maps_file
task_file = dscfg.input_data_dir + dscfg.task_maps_file

training_set = TaskPredictionNPYDataset(train_subj_ids, pred_file, task_file, zstasks = dscfg.z_score_task_maps)
test_set =  TaskPredictionNPYDataset(test_subj_ids, pred_file, task_file, zstasks = dscfg.z_score_task_maps)

dscfg.predictors_file = "all_grads_254_full_unrelated.L.fisherz.commonGroupPCA.npy"
training_set = TaskPredictionNPYDataset(train_subj_ids, pred_file, task_file, zstasks = dscfg.z_score_task_maps)
print(training_set.predictor_data[subjs[0]].shape)
np.concatenate((training_set.predictor_data[subjs[0]],training_set.predictor_data[subjs[1]]), axis=1).shape

""";

################### Resting state data +##############################

import nibabel as nib
nib.imageglobals.logger.setLevel(40)

import scipy.stats
# function to load the 4 resting state time series
def load_ts_data(file_path, subj, session, v=False, bma_slice=slice(0, 29696, None), zscore=False):
  file_path = file_path.format(**{"session":session, "subj" : subj})
  if v: print(file_path)
  nimg = nib.load(file_path)
  fsdata = nimg.get_fdata()[:, bma_slice]  # e.g. of shape (4800, 29696) 
  # somehow expand to 32k????
  if zscore:
    fsdata = scipy.stats.zscore(fsdata, axis=0) # zscore over time?
  return fsdata;