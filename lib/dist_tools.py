import nibabel as nib
import numpy as np
import os

def get_struct_from_cifti2bma(bma, structure="left_cortex", return_idx=0):
  struc = bma.to_cifti_brain_structure_name(structure) # i.e. "CIFTI_STRUCTURE_CORTEX_LEFT"
  for x in bma.iter_structures():
    if x[0]==struc: 
      return (i, x) if return_idx else x   

# returns the first brain model axis returned
def get_cifti_axis_by_type(cifti2, axtype="bma" , return_idx = False):
  if axtype == "bma": axtype=nib.cifti2.cifti2_axes.BrainModelAxis;
  if axtype == "label": axtype = nib.cifti2.cifti2_axes.LabelAxis;
  for i in range(cifti2.header.number_of_mapped_indices):
    x = cifti2.header.get_axis(i).__class__ is axtype;
    if x: return (i, cifti2.header.get_axis(i)) if return_idx else cifti2.header.get_axis(i)

def load_cifti_parcellation(cifti2, structure="left_cortex", verbose=0):
  if isinstance(cifti2, (str, os.PathLike)): cifti2 = nib.load(cifti2);
  if not isinstance(cifti2, nib.cifti2.Cifti2Image):
    raise Exception("Parcellation provided is not a cifit2 file nor a filename pointing to one.")

  if verbose: print("cifti_shape:", cifti2.shape)
  bma_idx, bma = get_cifti_axis_by_type(cifti2, axtype="bma", return_idx=1)
  if verbose: print("BMA:", bma_idx, bma)
  c_slice = get_struct_from_cifti2bma(bma, structure=structure)[1]
  if verbose: print("slice for strcuture:", structure, c_slice)
  lba_idx, lba = get_cifti_axis_by_type(cifti2, axtype="label", return_idx=1)
  # get the dictionary that maps integer labels to names
  labels = lba.label.item() # potentially includes labels for both hemispheres
  label_dict = {key: labels[key][0] for key in labels.keys()}
  if verbose: print("label dict info:", len(label_dict) , "items contained, first 10:", {k : label_dict[k] for k in list(label_dict.keys())[:10]})
  # get the parcellation data, irrespective of axis order
  ndim = len(cifti2.shape)
  a = {lba_idx:0, bma_idx: c_slice}
  idxs = [a[x] for x in range(ndim)]
  #label_data = cifti2.get_fdata()[idxs]
  label_data = cifti2.get_fdata()[tuple(idxs)]
  if verbose: print("label_data info: shape of ", label_data.shape, "i.e.", label_data)
  if verbose: print("unqiue label_data:", len(np.unique(label_data)), "labels contained, i.e.:", np.unique(label_data)[:10])
  return label_data, label_dict;

def get_region_based_conn(scmfd, label_list, regions, verbose=0):
  scout = np.zeros((3, len(label_list), len(regions)))

  for n, roi in enumerate(regions):
      if verbose: print(n, end=", ", flush=True)
      source_node_indices = np.where(label_list==roi)[0].astype(np.int32)
      ## min & mean
      scout[0, :, n] = scmfd[source_node_indices,:].min(axis=0)
      scout[1, :, n] = scmfd[source_node_indices,:].mean(axis=0)
      ## surf_centroid
      xf= scmfd[source_node_indices,:]
      local_sdm = xf[:, source_node_indices]
      cum_dist = local_sdm.sum(axis=0);
      min_idx = np.argsort(cum_dist)[0]
      scout[2, :, n] = scmfd[:,source_node_indices[min_idx]]
  return scout;