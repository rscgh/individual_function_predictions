


###################################################################
## For own 2layer NN

import glob, os
from torch import nn

def simple_n_layer_nn(ncfg):
  dropout = ncfg["hl_dropout"] if "hl_dropout" in ncfg.keys() else 0.0; #
  #print(dropout.__class__) # float
  
  modules = [nn.Linear(ncfg["n_inputs"], ncfg["n_hidden"]), nn.ReLU()]
  if not((dropout == 0.0 or dropout is None)): #
    modules = [nn.Linear(ncfg["n_inputs"], ncfg["n_hidden"]), nn.Dropout(p=dropout),  nn.ReLU()] #
    
  for i in range(ncfg["n_hidden_layers"]-1):
    modules.append(nn.Linear(ncfg["n_hidden"], ncfg["n_hidden"]))
    if not((dropout == 0.0 or dropout is None)): 
      modules.append(nn.Dropout(p=dropout))
    modules.append(nn.ReLU())

  modules.append(nn.Linear(ncfg["n_hidden"], ncfg["n_tasks"]))
  model = nn.Sequential(*modules)
  return model;


import glob

def get_last_checkpoint_path(mversion, ckpt_dir="outp/dat"):
  cpaths = glob.glob(os.path.join(ckpt_dir,f"nn.v{mversion}_*"))
  cpaths.sort(key=os.path.getmtime)
  ckpt_path = None if len(cpaths) == 0 else cpaths[-1];
  return ckpt_path

"""
## Saving and loading checkpoint functions

def save_checkpoint(model, optimizer, ncfg, logs, epoch=0, mversion="001", verbose=1, odir="outp/dat", is_best_loss=0, force_save=False):
  gap= "best_loss" if is_best_loss else epoch;
  path = os.path.join(odir, f"nn.v{mversion}_train.{gap}.pt")
  sdict = {'epoch': epoch, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), "ncfg": ncfg, "logs": logs}
  if not os.path.exists(path) or force_save: 
    torch.save(sdict, path)
    if verbose: print("Saved checkpoint: ", path)
  else: 
    if verbose: print(f"Check point under path {path} already exists and will not be overwritten")

def load_checkpoint(path):
  ckpt = torch.load(path)
  return ckpt, ckpt["ncfg"], ckpt["logs"], ckpt["epoch"];
"""


###################################################################
## new version of saving checkpoints

import os
def save_checkpoint(model, conf, version="001", \
  save_globals = None, verbose=1, odir="outp/dat", detail="", force=False, context=globals(), **kwargs):
  # logs=None, history=None
  # detail can be epoch 
  path = os.path.join(odir, f"nn.v{version}_train.{detail}.pt")
  sdict = {"conf": conf, "model_state_dict": model.state_dict(), "model_type": str(type(model)), "version":version}
  
  if not(save_globals is None):
    sdict.update({"globals": {var : context[var] for var in save_globals}})
    
  for k,obj in kwargs.items():
    is_special = isinstance(obj, torch.optim.Optimizer) or isinstance(obj, torch.optim.lr_scheduler._LRScheduler);
    if is_special:
      sdict.update({k+"_state_dict":obj.state_dict(), k+"_type": str(type(obj))})
    else:
      sdict.update({k:obj})
      
    
  if not os.path.exists(path) or force: 
    torch.save(sdict, path)
    if verbose: print("Saved checkpoint: ", path)
  else: 
    if verbose: print(f"Check point under path {path} already exists and will not be overwritten")

import os, time
def get_last_modified_time(file_name, time_format='%d.%m.%Y %H:%M:%S'):
  modTimesinceEpoc = os.path.getmtime(file_name)   
  return time.strftime(time_format, time.localtime(modTimesinceEpoc))

def load_checkpoint(path, reinstate_globals=True, verbose=1, context=globals()):
  ckpt = torch.load(path)
  lmdf = get_last_modified_time(path)
  if verbose: print(f"Loading checkpoint from {path}, last modified on {lmdf}") 
                   
  if reinstate_globals and "globals" in ckpt.keys():
    if verbose: print("reinstate globals:", ckpt["globals"].keys())
    for k,v in ckpt["globals"].items(): context[k]=v
                   
  if verbose: 
    print(f"Checkpoint has keys: {ckpt.keys()}")
    ks=ckpt["conf"].keys();
    print(f"Config has keys: {ks}")
  
  return ckpt, ckpt["conf"], lmdf, ckpt["version"];


###################################################################
## For brainSurfCNN

def get_bl_within_across_subj_loss(model, train_loader, contrast_mse_loss):
  model.eval()
  wsl, asl = [], []; 

  for batch_idx, (data, target) in enumerate(train_loader):
      #data, target = data.cuda(), target.cuda()
      output = model(data)
      within_subj_loss, across_subj_loss = contrast_mse_loss(output, target)
      wsl.append(within_subj_loss.item())
      asl.append(across_subj_loss.item())

  init_within_subj_margin = np.array(wsl).mean()
  init_across_subj_margin = np.array(asl).mean()
  return init_within_subj_margin, init_across_subj_margin; 

import numpy as np
import torch 


def run_epoch(model, train_loader, contrasts, optimizer, loss_fn, \
  loss_type, within_subj_margin, across_subj_margin, train = True, ret_detailed_corrs=False):
    
    if train: 
      model.train()
    else:
      model.eval()
    
    total_loss = 0
    total_corr = []
    count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
      #data, target = data.cuda(), target.cuda()
      
      if train: 
        optimizer.zero_grad()
      
      output = model(data) # silently fails here
      
      if loss_type == 'mse':
         loss = loss_fn(output, target)
      elif loss_type == 'rc':
         within_subj_loss, across_subj_loss = loss_fn(output, target)
         loss = torch.clamp(within_subj_loss - within_subj_margin, min=0.0) + torch.clamp(within_subj_loss - across_subj_loss + across_subj_margin, min = 0.0)
      else:
         raise("Invalid loss type")
      
      if train:
        loss.backward()
        optimizer.step()
      
      total_loss += loss.item()
      
      reshaped_output = np.swapaxes(output.cpu().detach().numpy(), 0, 1)
      reshaped_target = np.swapaxes(target.cpu().detach().numpy(), 0, 1)
      corrs = np.diag(compute_corr_coeff(reshaped_output.reshape(reshaped_output.shape[0], -1), reshaped_target.reshape(reshaped_target.shape[0], -1)))
      if batch_idx == 0:
        total_corr = corrs
      else:
        total_corr = total_corr + corrs
            
    total_loss /= len(train_loader)
    total_corr /= len(train_loader)
    
    print((" Train" if train else "Val") + ': avg loss: {:.4f} - avg corr: {:.4f}'.format(total_loss, np.mean(total_corr)))
    for j in range(len(contrasts)):
        print("      %s: %.4f" % (contrasts[j], total_corr[j]))
    
    if ret_detailed_corrs:
      return total_loss, np.mean(total_corr), total_corr

    return total_loss, np.mean(total_corr)

###################################################################
from collections import OrderedDict
import os
def save_extended_checkpoint(model, optimizer, scheduler, epoch, fname, output_dir):
  state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.cuda.sparse.FloatTensor"]
  state_dict_no_sparse = OrderedDict(state_dict_no_sparse)
  checkpoint = {
    'epoch': epoch,
    'state_dict': state_dict_no_sparse,
    'optimizer': optimizer.state_dict(),
  }

  if not(scheduler is None):
    checkpoint['scheduler']= scheduler.state_dict()
    checkpoint["scheduler_type"] = str(scheduler.__class__);
    checkpoint["sched_kwargs"] = sched_kwargs;
    
  #ckpt["model_conf"] = dict(n_channels_per_hemi = 50, n_feat_channels = 64, n_output_channels = 10, min_mesh_lvl = 0, max_mesh_lvl = 2) 
  checkpoint["model_conf"] = model_conf#dict(in_ch = 50, fdim = 64, out_ch = 10, min_level = 0, max_level = 2, mesh_dir="BrainSurfCNN/data/fs_LR_mesh_templates") 
  checkpoint["train_conf"] = dict(margin_anneal_step = margin_anneal_step, n_epochs = n_epochs)
  checkpoint["train_conf"] = dict(margin_anneal_step = margin_anneal_step, n_epochs = n_epochs)

  checkpoint["optimizer_type"] = str(optimizer.__class__);
  checkpoint["optim_kwargs"] = optim_kwargs;

  # from training: val_corrs, val_corrs_mean, val_losses
  checkpoint["globals"] = {"contrast_info" : [x.split(" ") for x in task_names], "task_names":task_names, "subjs": subjs}
  checkpoint["dist_dataset_info"] = dict(H = H, hemi=hemi, parcell = parcell, exp_id = exp_id, n_subjs=n_subjs, z_score_task_maps = z_score_task_maps)
  if "prev_checkpoints" in globals().keys(): 
        checkpoint["prev_checkpoints"]= prev_checkpoints;
        checkpoint["prev_epochs"]= prev_epochs;

  checkpoint["train_info"] = dict(test_batch_size=test_batch_size, training_batch_size=training_batch_size, n_train_subjs = n_train_subjs, train_subjs = np.array(subjs)[:n_train_subjs], val_subjs = np.array(subjs)[n_train_subjs:], val_corrs=val_corrs, val_losses = val_losses, val_corrs_det= val_corrs_det);

  torch.save(checkpoint, os.path.join(output_dir, fname))



###################################################################
## Iterating over modules and collecting activations

def iter_modules(curr_module, prefix = "", mod_dict = {}, verb=False, plain_style=False, rootlv=True):
  mod_repr = prefix + curr_module._get_name()
  if plain_style: 
    mod_repr = mod_repr.replace("_Sequential", "").replace("(","").replace(")","");
    mod_repr = prefix#.replace("(","").replace(")","").replace("_","")
  mod_dict.update({mod_repr : curr_module});
  if verb: print(mod_repr);
  for key, module in curr_module._modules.items():
    pf = mod_repr+("" if (rootlv==True and plain_style==2) else ".")
    pf = pf + (f"({key})_" if not plain_style==2 else f"{key}")
    _ = iter_modules(module, prefix = pf, mod_dict = mod_dict, plain_style=plain_style, verb=verb, rootlv=False)
    #iter_modules(module, prefix = mod_repr+f".({key})_", mod_dict = mod_dict, verb=verb)
  return mod_dict;




from functools import partial
import collections, re

def save_activations(activations, name, print_stats, print_layerinout, module, inp, out):
  #global HOOKS_PRINT_MACHINE_STATS, HOOKS_PRINT_LAYER_INOUT
  if print_layerinout: 
    print('Inside ' + module.__class__.__name__ + ' forward', "in:", inp[0].shape, "out:", out.shape)
  if print_stats: 
    get_machine_stats(verb=1, gpu=0, ret=0, sep="  ", proc=1, rnd=2, per_cpu=1)
  activations[name].append(out.detach().cpu().numpy())

contains_any = lambda s, incl: np.any(np.array([re.match(x, s) for x in incl])!=None) 

def add_activation_hooks_to_layers(model, include_layers_containing, print_layerinout=False, print_stats = False, verbose=False):
   actv = collections.defaultdict(list)
   if not("hooks" in globals().keys()): globals()["hooks"] = []  
   for h in globals()["hooks"]: h.remove()
   md = iter_modules(model, plain_style=2, mod_dict={})
   hook_layer_dict = {}
   for name in md.keys():
    if not contains_any(name, include_layers_containing): continue;
    if verbose: print("Adding hook to:", name) # end=", ");
    h1 = md[name].register_forward_hook(partial(save_activations, actv, name, print_stats, print_layerinout))
    hooks.append(h1)
    hook_layer_dict.update({name: md[name]})

   return actv, hook_layer_dict, globals()["hooks"]

###################################################################
## For probing the inputs

from tqdm import tqdm

'''

def test_predictor_contributions(model, val_loader,  predictor_dim = -1, avg_input=None, score_fn = None, avg_target=None, grayl=slice(None)):
	"""
	This is an example of Google style.

	Args:
			model (torch.nn.module): neural network model
			val_loader: iterator over the dataset on which to benchmark the predictor contributions
					delivering input_data, target_pairs as required by the model
					default would be: (n_subjs, n_vertices, n_predictors or n_target_vars) 
					a different order can be indicted using the "predictor_dim" parameter
			avg_input: currently not utilized
			score_fn: should be "compute_all_scores" normally       
			predictor_dim (int): dimension indexing the predictor variables to be iterated 
					over to test for their contributions

	Returns:
			the tuple: (all_scores, all_scores_indv, mean_dev, abs_mean_dev)
			each of these arrays contains as many items as there are predictors in the model
			+ all_scores is a dict with the score names as keys and values are of shape (n_subjects, n_target_vars)
			+ if avg_target is None, all_scores_indv is an empty list
			+ items of mean_dev, abs_mean_dev are if shape (n_vertices, n_target_vars) 
	""" 

	n_inputs = next(iter(val_loader.dataset))[0].shape[predictor_dim]
	#print("n_inputs ", n_inputs)
	n_batches = len(val_loader)

	all_scores = []
	all_scores_indv = []
	mean_dev = []
	abs_mean_dev = []

	pbar = tqdm(total=(n_inputs+1) * n_batches)

	for i in range(n_inputs):
		
		outps   = []
		targets = []
		model.eval()

		for bn, (data, target) in enumerate(val_loader):
			
			# data e.g. of shape [4, 29696, 41] and predictor_dim=>2
			# with 41 indexing the predictors (e.g. fibre tracts)
			idxs = [slice(None)] * len(data.shape)
			idxs[predictor_dim] = i 
			# given that slice(None) corresponds to :
			# this would then equal to indexing with
			# data[:,:, i] = 0
			data[idxs] = 0

			output = model(data) # input shape: torch.Size([1, 50, 32492])
			outps.append(output.cpu().detach().numpy())
			targets.append(target.cpu().detach().numpy())

			pbar.update(1)
			pbar.set_postfix_str(f"Input {i}/{n_inputs}, Batch: {bn}/{n_batches}")
		
		predicted=np.concatenate(outps);
		targets = np.concatenate(targets);
		# compute_all_scores requires an input shape of (n_subjs, n_tasks, n_vertices)
		# with the n_tasks beeing the to predicted variables
		# so when we have a target of shape [n_subjs=4, n_verts=29696, n_tasks=10]
		# assuming the predicted variable dimension mirrors the predictor dimension,
		# and this beeing the last dimension, we have to swap the axes:
		if (predictor_dim == len(data.shape)-1) or (predictor_dim==-1):
			predicted = np.swapaxes(predicted,-2,-1)
			targets = np.swapaxes(targets,-2,-1)
            
		#print(predicted.shape, targets.shape)       
		#break;

		# compute scores for the overall prediction
		scoresd = score_fn(targets, predicted)
        #compute_all_scores(..., grayl=grayl, scores = {"r2_score": r2_score, "corr": comp_corr, "dice": comp_dice})
		all_scores.append(scoresd)

		if not(avg_target is None):
			# and the scores when regressing out a scaled version of the group mean by task
			# => relevant scores for assessing the prediction of individual differences
			iscoresd = score_fn(targets, predicted, avg_target = avg_target,  scaled_resid=1) 
			#compute_all_scores(..., grayl=grayl, scores = {"r2_score": r2_score, "corr": comp_corr, "dice": comp_dice})
			all_scores_indv.append(iscoresd)

		# save also mean deviation for the predictions from the target 
		mean_dev.append(np.absolute(targets-predicted).mean(axis=0))
		abs_mean_dev.append((targets-predicted).mean(axis=0))

	return all_scores, all_scores_indv, mean_dev, abs_mean_dev;
'''


def score_model(model, val_loader,  predictor_dim = -1, perturb_indices = None, score_fn = None, avg_target=None):
	
	outps   = []
	targets = []
	model.eval()

	for bn, (data, target) in enumerate(val_loader):
		
		if not(perturb_indices is None):
			# data e.g. of shape [4, 29696, 41] and predictor_dim=>2
			# with 41 indexing the predictors (e.g. fibre tracts)
			idxs = [slice(None)] * len(data.shape)
			idxs[predictor_dim] = perturb_indices # before: i
			# given that slice(None) corresponds to :
			# this would then equal to indexing with
			# data[:,:, i] = 0
			data[idxs] = 0

		output = model(data) # input shape: torch.Size([1, 50, 32492])
		outps.append(output.cpu().detach().numpy())
		targets.append(target.cpu().detach().numpy())

	predicted=np.concatenate(outps);
	targets = np.concatenate(targets);
	# compute_all_scores requires an input shape of (n_subjs, n_tasks, n_vertices)
	# with the n_tasks beeing the to predicted variables
	# so when we have a target of shape [n_subjs=4, n_verts=29696, n_tasks=10]
	# assuming the predicted variable dimension mirrors the predictor dimension,
	# and this beeing the last dimension, we have to swap the axes:
	if (predictor_dim == len(data.shape)-1) or (predictor_dim==-1):
		predicted = np.swapaxes(predicted,-2,-1)
		targets = np.swapaxes(targets,-2,-1)
	    
	#print(predicted.shape, targets.shape)       
	#break;

	# compute scores for the overall prediction
	scoresd = score_fn(targets, predicted)
	#e.g. compute_all_scores(..., grayl=grayl, scores = {"r2_score": r2_score, "corr": comp_corr, "dice": comp_dice})
	
	iscoresd = None
	if not(avg_target is None):
		# and the scores when regressing out a scaled version of the group mean by task
		# => relevant scores for assessing the prediction of individual differences
		iscoresd = score_fn(targets, predicted, avg_target = avg_target,  scaled_resid=1) 
		# e.g. compute_all_scores(..., grayl=grayl, scores = {"r2_score": r2_score, "corr": comp_corr, "dice": comp_dice})

	# save also mean deviation for the predictions from the target 
	mean_dev = np.absolute(targets-predicted).mean(axis=0)
	abs_mean_dev = (targets-predicted).mean(axis=0)

	return scoresd, iscoresd, mean_dev, abs_mean_dev;



def test_predictor_contributions(model, val_loader,  predictor_dim = -1, avg_input=None, score_fn = None, avg_target=None, grayl=slice(None)):
	"""
	This is an example of Google style.

	Args:
			model (torch.nn.module): neural network model
			val_loader: iterator over the dataset on which to benchmark the predictor contributions
					delivering input_data, target_pairs as required by the model
					default would be: (n_subjs, n_vertices, n_predictors or n_target_vars) 
					a different order can be indicted using the "predictor_dim" parameter
			avg_input: currently not utilized
			score_fn: should be "compute_all_scores" normally       
			predictor_dim (int): dimension indexing the predictor variables to be iterated 
					over to test for their contributions

	Returns:
			the tuple: (all_scores, all_scores_indv, mean_dev, abs_mean_dev)
			each of these arrays contains as many items as there are predictors in the model
			+ all_scores is a dict with the score names as keys and values are of shape (n_subjects, n_target_vars)
			+ if avg_target is None, all_scores_indv is an empty list
			+ items of mean_dev, abs_mean_dev are if shape (n_vertices, n_target_vars) 
	""" 

	n_inputs = next(iter(val_loader.dataset))[0].shape[predictor_dim]
	#print("n_inputs ", n_inputs)
	n_batches = len(val_loader)

	all_scores = []
	all_scores_indv = []
	mean_devs = []
	abs_mean_devs = []

	pbar = tqdm(total=n_inputs)

	for i in range(n_inputs):
		
		scoresd, iscoresd, mean_dev, abs_mean_dev = score_model(model, val_loader,  predictor_dim = predictor_dim, perturb_indices = i, score_fn = score_fn, avg_target=avg_target);
		
		all_scores.append(scoresd)
		all_scores_indv.append(iscoresd)
		mean_devs.append(mean_dev)
		abs_mean_devs.append(abs_mean_dev)
        
		pbar.update(1)
		pbar.set_postfix_str(f"Input {i}/{n_inputs}")


	return all_scores, all_scores_indv, mean_devs, abs_mean_devs;



###################################################################
## New contrast function(s)

def contr_batch_mse_loss(predicted, target, verbose=False, mse = nn.MSELoss(), sum_losses=False, weighting=1):
    """
    predicted and target, both of shape (n_subjs, n_vertices, n_pred)
    
    loss = MSE(pred, true) + p * torch.abs(MSE(pred, true') - MSE(true, true')))
    """
    
    mse_loss = mse(predicted, target)
    avg_real_cross_subj_loss = mse(target[1:], target[:-1])
    avg_pred_cross_subj_loss = mse(predicted[1:], target[:-1])
    
    contr_loss = torch.abs(avg_real_cross_subj_loss -avg_pred_cross_subj_loss)
    
    if verbose:
        print("target cross subj loss:", avg_real_cross_subj_loss)
        print("prediction corss subj loss:", avg_pred_cross_subj_loss)
        print("# mse:", mse_loss, "ctr_loss:", contr_loss)
    
    return (mse_loss+weighting*contr_loss) if sum_losses else (mse_loss, weighting*contr_loss);



def contr_batch_mse_loss_nonreduced(predicted, target, verbose=False, mse = nn.MSELoss(reduce=False), sum_losses=False):
    """
    predicted and target, both of shape (n_subjs, n_vertices, n_pred)
    
    loss = MSE(pred, true) + p * torch.abs(MSE(pred, true') - MSE(true, true')))
    """
    
    # nn.MSELoss(reduce=False) returns something in  shape: torch.Size([20, 29696, 10])
    # first mean across subject dimension, then across remaining ...
    mse_loss = mse(predicted, target).mean(0).mean() 
    
    # mean across subjec dimension
    avg_real_cross_subj_loss = mse(target[1:], target[:-1]).mean(0) 
    avg_pred_cross_subj_loss = mse(predicted[1:], target[:-1]).mean(0)
    # mean across the remaining dimensions
    contr_loss = torch.abs(avg_real_cross_subj_loss -avg_pred_cross_subj_loss).mean() 
    
    if verbose:
        print("target cross subj loss:", avg_real_cross_subj_loss)
        print("prediction corss subj loss:", avg_pred_cross_subj_loss)
        print("# mse:", mse_loss, "ctr_loss:", contr_loss)
    
    return (mse_loss+contr_loss) if sum_losses else (mse_loss, contr_loss);


"""
never used so far:
def contr_batch_mse_loss_nonreduced(predicted, target, verbose=False, mse = nn.MSELoss(reduce=False), sum_losses=False):

    
    # nn.MSELoss(reduce=False) returns something in  shape: torch.Size([20, 29696, 10])
    # first mean across subject dimension, then across remaining ...
    mse_loss = mse(predicted, target).mean(0).mean() 
    
    # mean across subjec dimension
    avg_real_cross_subj_loss = mse(target[1:], target[:-1]).mean(0) 
    avg_pred_cross_subj_loss = mse(predicted[1:], target[:-1]).mean(0)
    # mean across the remaining dimensions
    contr_loss = torch.abs(avg_real_cross_subj_loss -avg_pred_cross_subj_loss).mean() 
    
    if verbose:
        print("target cross subj loss:", avg_real_cross_subj_loss)
        print("prediction corss subj loss:", avg_pred_cross_subj_loss)
        print("# mse:", mse_loss, "ctr_loss:", contr_loss)
    
    return (mse_loss+contr_loss) if sum_losses else (mse_loss, contr_loss);

""";


def torch_differentiation_score(A,B):
    n_items= B.shape[0]
    n_tasks = B.shape[2]
    diag_idxs= np.diag_indices(n_items)
    
    diff_scores=torch.zeros(n_tasks)

    for nt in range(n_tasks):
        corr_mat = torch.corrcoef(torch.concatenate((A[:,:, nt], B[:,:, nt]), axis=0))
        #plt.matshow(corr_mat.detach().numpy())
        corr_mat = corr_mat[:n_items,n_items:]
        diagonal_corrs = corr_mat[diag_idxs]
        avg_cross_corrs = corr_mat.mean(axis=0); 
        diff_scores[nt] = torch.mean(diagonal_corrs- avg_cross_corrs);
        
    return diff_scores;
    
def batch_differentiating_mse_loss(predicted, target, verbose=False, mse = nn.MSELoss(), sum_losses=False):
    """
    predicted and target, both of shape (n_subjs, n_vertices, n_pred)
    
    loss = MSE(pred, true) + p * torch.abs(MSE(pred, true') - MSE(true, true')))
    """
    mse_loss = mse(predicted, target)
    
    target_differetiability  = torch_differentiation_score(target,target);
    current_differetiability = torch_differentiation_score(predicted,target)
    #print(target_differetiability.mean(), current_differetiability.mean())
    
    diff_loss = torch.abs(target_differetiability - current_differetiability).mean()
    
    #if verbose:
    #    print("target cross subj loss:", avg_real_cross_subj_loss)
    #    print("prediction corss subj loss:", avg_pred_cross_subj_loss)
    #    print("# mse:", mse_loss, "ctr_loss:", contr_loss)
    
    return (mse_loss+diff_loss) if sum_losses else (mse_loss, diff_loss);

