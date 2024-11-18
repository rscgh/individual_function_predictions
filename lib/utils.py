

# for making dynamic replacement strings
f= lambda str: eval("f'" + f"{str}" + "'")




import sys, psutil, os
import numpy as np;


def get_cuda_device_stats(device, n=0, rs = lambda x: x, udiv = (1024**3), unit="gb"):
  #cmem_all = torch.cuda.memory_allocated(device = device)/ unit
  cmem_all = torch.cuda.torch.cuda.memory_reserved(device = device)/ udiv
  cmem_max = torch.cuda.get_device_properties(device = device).total_memory/udiv;
  cmem_cached=torch.cuda.memory_cached(device = device)/ udiv
  #cmem_perc= torch.cuda.memory_usage(device=device)
  dname= torch.cuda.get_device_name(device=device);
  torch.cuda.synchronize()
  data = {f"cuda{n}": dname, f"cuda{n}_mem_res": rs(cmem_all)+unit, f"cuda{n}_mem_cached": rs(cmem_cached)+unit, \
          f"cuda{n}": rs(cmem_all*100/cmem_max)+"%", f"cuda{n}_mem_tot" : rs(cmem_max)+unit} #, "%cuda": cmem_perc
  return data;

## from lib.utils import get_machine_stats
## get_machine_stats(verb=1, gpu=0, ret=0, sep="  ", proc=1, rnd=2, per_cpu=1)

def get_machine_stats(verb=False, gpu=False, ret=True, unit="gb", sep="\t", proc=False, rnd=False, per_cpu=False, per_gpu=False):
  unit2div = {"gb": (1024**3), "mb": (1024**2)};
  if not (unit in unit2div.keys()): unit = "gb";
  udiv = unit2div[unit];
  
  rs = lambda x : str(round(x, rnd)) if rnd != False else str(x);
  
  # CPU; average CPU utilization since last call (?)
  cpu_perc = psutil.cpu_percent()
  # Memory 
  vmem_perc = psutil.virtual_memory().percent;
  vmem_total = psutil.virtual_memory().total / udiv
  data = {"cpu": rs(cpu_perc)+"%", "vmem" : rs(vmem_perc) +"%", "vmem_tot" : rs(vmem_total)+unit , "n_cpus": psutil.cpu_count()}
  
  if proc:
    pid = os.getpid() if proc == True else proc;
    process = psutil.Process(pid)
    proc_mem = process.memory_info()[0]/ udiv; # (1024**3) ~ GB...I think
    proc_perc = proc_mem*100 / (vmem_total)
    data.update({"pid": pid, "proc_mem" : rs(proc_mem)+unit, "proc_mem/tot": rs(proc_perc)+"%" })
    cpu_ids= process.cpu_affinity()
    sum_cpu_perc = process.cpu_percent(); 
    data.update({"proc_n_cpus": len(cpu_ids), "proc_cum_cpu": rs(sum_cpu_perc)+"%", \
                 "proc_avg_cpu": rs(sum_cpu_perc/len(cpu_ids)) + "%"});
                 
    if per_cpu: 
        assigned_cpu_usage = np.array(psutil.cpu_percent(interval=None, percpu=True))[cpu_ids]
        data.update({"proc_assigned_cpus": cpu_ids, "assigned_cpu_util_%" : assigned_cpu_usage.round(rnd) })
        pass;
    
  # GPU memory usage whith pytorch
  if gpu!=False:
   if not("torch" in sys.modules):  
        print("please import torch first")
   
   if not torch.cuda.is_available():
     data.update({"cuda": "not_avail"});
    
   elif per_gpu: 
    for x in range(torch.cuda.device_count()):
        device = torch.device("cuda:"+ str(x));
        data.update(get_cuda_device_stats(device, x, rs, udiv, unit));
   else:
    device = torch.device("cuda" if gpu==True else gpu) 
    data.update(get_cuda_device_stats(device, "", rs, udiv, unit));
    
  if verb:
    print(sep.join([f"{k}: {data[k]}" for k in data.keys()]))
  if ret: 
    return data;


import os, time
def get_last_modified_time(file_name, time_format='%d.%m.%Y %H:%M:%S'):
  modTimesinceEpoc = os.path.getmtime(file_name)   
  return time.strftime(time_format, time.localtime(modTimesinceEpoc))


# function to load the 4 resting state time series
import nibabel as nib 
import scipy.stats

def load_hcp_ts_data(file_path, subj, session, v=False, bma_slice=slice(0, 29696, None), zscore=False):
  file_path = file_path.format(**{"session":session, "subj" : subj})
  if v: print(file_path)
  nimg = nib.load(file_path)
  fsdata = nimg.get_fdata()[:, bma_slice]
  # somehow expand to 32k????
  return scipy.stats.zscore(fsdata, axis=0) if zscore else fsdata
