
from tqdm import tqdm
import numpy as np

def lin_layer_lrp(layer_k, act_j, R_k, E = 0.00000000001, prog=False):
  # Implements Montavon Eq. (10.2), 
  #   but doesnt follow the 4 decomposition (zk,sk,cj,Rj) steps directly 
  # sample shapes are indicted for 
  # Relevance at layer k:         
  #    R_k of shape (3, 1, 10): 3 datapoints, with 10 relevance values 
  # activtiy at previous layer j: 
  #    act_j of shape (3, 1, 892): 3 datapoints, with 892 scalar activation values 
  # yields relevance at previous layer j: 
  #    R_J  of shape (3, 1, 892): 3 datapoints, with 892 relevance values
  
  ## layer k transforms the activations/outpts from the prev layer j to outputs k 
  ## with relevance Rk; here we get the weight for the forward pass:
  w = layer_k.weight.detach().numpy().T; # shape (892, 10)
  b = layer_k.bias.detach().cpu().numpy()

  n_samples = R_k.shape[0]
  n_j = w.shape[0];
  R_J = np.zeros((n_samples, 1, n_j));
    
  rng = tqdm(range(n_samples), miniters=int(n_samples/30),maxinterval=200) if prog else range(n_samples)
  for i in rng:
    ## ~ aj * p(wjk)
    upper = act_j[i] * w.T  # shape (10, 892)

    # ~ zk = E + sum_j( aj * p(wjk) )
    #lower = E + np.expand_dims(upper.sum(axis=-1), axis=0) # (1, 10)
    lower = E + upper.sum(axis=-1) + b # (10)
    
    uT = np.swapaxes(upper,-1,-2) #(892, 10)
    ## [aj * p(wjk) ] / zk 
    inner = uT / lower # (892, 10)
    #print(inner)
    
    ## ... * Rk
    inner_x_Rk = np.swapaxes(inner * R_k[i],-1,-2) # (10, 892)
    # sum over K
    R_J[i,0,:] = inner_x_Rk.sum(axis=-2) # (892)
    
  return R_J

## Input Attention
"""
def lin_layer_lrp_inpAtt_OLD(layer_k, R_k, E = 0.00000000001):
  # implements Montavon Table 10.A, w2-rule
  # layer k is the first layer here
    
  n_samples = R_k.shape[0];
  w = layer_k.weight.detach().numpy().T; # (223, 892)
  upper = np.repeat(np.expand_dims((w.T**2), axis=0), n_samples, axis=0)  # shape (3, 892, 223)
  print(upper.shape)
  uT = np.swapaxes(upper,-1,-2) # (3, 223, 892)
  lower = E + np.expand_dims(upper.sum(axis=-1), axis=1) # (3, 1, 223)
  inner = uT / lower # (3, 892, 10)
  inner_x_Rk = np.swapaxes(inner * R_k,-1,-2) # (3, 10, 892)
    
  # sum over K
  R_J = np.expand_dims(inner_x_Rk.sum(axis=-2), axis=1) # (3, 1, 892)
  return R_J
"""

def lin_layer_lrp_inpAtt(layer_k, R_k, E = 0.00000000001, prog=False):
  # implements Montavon Table 10.A, w2-rule
  # layer k is the first layer here
    
  n_samples = R_k.shape[0];
  w = layer_k.weight.detach().numpy().T; # (223, 892)
  b = layer_k.bias.detach().cpu().numpy()
  
  n_j = w.shape[0];
  R_J = np.zeros((n_samples, 1, n_j));
    
  rng = tqdm(range(n_samples), miniters=int(n_samples/30),maxinterval=200) if prog else range(n_samples)
  for i in rng:
      upper = (w.T**2) # shape (892, 223) ?
      #lower = E + np.expand_dims(upper.sum(axis=-1), axis=1) # (3, 1, 223)
      lower = E + upper.sum(axis=-1) + b # (892)
    
      uT = np.swapaxes(upper,-1,-2) # (223, 892)

      inner = uT / lower # (3, 892, 10)
      inner_x_Rk = np.swapaxes(inner * R_k[i],-1,-2) # (3, 10, 892)
    
      # sum over K
      #R_J = np.expand_dims(inner_x_Rk.sum(axis=-2), axis=1) # (3, 1, 892)
      R_J[i,0,:] = inner_x_Rk.sum(axis=-2) # (892)
  
  return R_J


def get_2layerNN_relevances(model, inp, out, interm, tn, layer1_lrp_rule="att", prog=False, mem_stats=True, nan2zero=True):
  if mem_stats: get_machine_stats(verb=1, rnd=2, ret=0)
  R_out = np.zeros_like(out)
  R_out[:,:,tn] = out[:,:,tn]
  R_L1 = lin_layer_lrp(model[2], interm, R_out)  
  R_inp = None;
  if layer1_lrp_rule == "att":
    R_inp = lin_layer_lrp_inpAtt(model[0], R_L1, prog=prog)
  else:
    R_inp = lin_layer_lrp(model[0], inp, R_L1, prog=prog)
  if mem_stats: get_machine_stats(verb=1, rnd=2, ret=0)
  if nan2zero:
    R_inp = np.nan_to_num(R_inp, nan=0)
    R_L1 = np.nan_to_num(R_L1, nan=0)
  return (R_inp, R_L1)