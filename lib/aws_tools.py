

import subprocess as sp
import os
aws_bucket = "s3://hcp-openaccess/"
s3copy      = 'python -m awscli s3 cp {aws_bucket}{file} {local_dir}{file}'
aws_exists_query = "python -m awscli s3api head-object --bucket hcp-openaccess --key {file}"
f= lambda str: eval("f'" + f"{str}" + "'")

def aws_exists(file, bucket="hcp-openaccess", suppress_exeption_print=True):
  global aws_exists_query
  globals()["file"]=file;
  globals()["bucket"]=bucket;
  return (run_cmd(f(aws_exists_query),suppress_exeption_print=suppress_exeption_print) != -1);


def exists_locally_or_on_aws(file, bucket="hcp-openaccess", local_dir = "/scratch/users/robert.scholz2/" ):
  if os.path.exists(f"{local_dir}{file}"):
    return True
  return aws_exists(file, bucket)

def run_cmd(cmd, suppress_exeption_print=False):
  try:
    res = sp.run(cmd, shell=True, check=True,  capture_output=True);
    return res.stdout.decode('utf-8')
  except Exception as e:
    if not suppress_exeption_print: print("Exception raised: ", e);
    return -1;

def download_aws_file(file, bucket="s3://hcp-openaccess/", local_dir = "/scratch/users/robert.scholz2/",
                     force_local_overwrite=False):
  if os.path.exists(f"{local_dir}{file}") and (not force_local_overwrite): 
    print(bc.blue, "File exists locally already. Not overwriting.", bc.end)
    return f"{local_dir}{file}"
    
  remote_extists = aws_exists(file)
  if not remote_extists: 
    print(bc.red,"Not downloading file, as it doesnt exist on aws:", file, bc.end);
    return False;
  else:
    print("Downloading file:", file);
    globals()["local_dir"]=local_dir
    #!python -m awscli s3 cp {aws_bucket}{file} {local_dir}{file}
    cmd = f"python -m awscli s3 cp {aws_bucket}{file} {local_dir}{file}"
    res = sp.run(cmd, shell=True, check=True,  capture_output=True);
    #return res.stdout.decode('utf-8')sp.
    return f"{local_dir}{file}"

class bc: 
  cyan='\033[96m'; green='\033[92m'; end='\033[0m'; red='\033[91m'; under='\033[4m';
  pink='\033[95m'; blue='\033[94m'; yellow='\033[93m'; bold='\033[1m'