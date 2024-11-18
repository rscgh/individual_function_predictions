

import numpy as np

from sklearn.linear_model import LinearRegression


def residualize(data, confound, return_reg = False, **kwargs):
  # data and confounds are both one dim arrays
  # data' = Int + ( C* confound ) + res
  # with int and c beeing scalars; and res having same dim as data.
  # kwargs, e.g. fit_intercept = 0
  # returns the residuals
  a = data.reshape(1,-1)
  b = confound.reshape(1,-1)
  reg = LinearRegression(**kwargs).fit(b.T, a.T)
  # reg.coef_[0]
  pred = reg.predict(b.T).T
  residuals = a[0]-pred[0]
  if return_reg: return residuals, reg
  return residuals


res_example_usage = """
a = np.array([1,3,3    ,2,2,2,1,5,6,6,4,2,2])
b = 2*a
a = np.array([2,3,3    ,1,2,3,1,5,8,4,4,0,2])*0.9
res = residualize(a, b, fit_intercept=0)

from matplotlib import pyplot as pyt
plt.plot(a);
plt.plot(b);
plt.plot(res);

""" 

def batch_resid(batch_data, confounds, **kwargs):
  b_resid = np.zeros_like(batch_data)
  for sn in range(batch_data.shape[0]):
    for tn in range(batch_data.shape[1]):
       b_resid[sn, tn, :] =  residualize(batch_data[sn,tn,:], confounds[tn], **kwargs)
    
  return b_resid;

#batch_resid(targets[:,:, grayl], avg_target, fit_intercept=False).shape # (152, 10, 29696)


################### Efficient way to do selective correlation of two arays ###############################

# https://cancerdatascience.org/blog/posts/pearson-correlation/
import numpy as np
def np_pearson_cor(x, y):
    # x and y of shape (n_features, n_vars)
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

# x = np.random.uniform(size=(500,500))
#y = np.random.uniform(size=(500,100))
#np_pearson_cor(x,y).shape # (500, 100)

################### MEASURES FOR MODEL FIT ###############################

from scipy.spatial.distance import dice

def comp_dice(y_true, y_pred, perc=95 ):
  # p = np.percentile(array2d, 95, axis=-1); (array2d.T > p).T
  y_true_perc = np.percentile(y_true, perc) #, axis=-1)
  y_pred_perc = np.percentile(y_pred, perc)
  # dice takes 1D boolean arrays as input and delivers back a dissimiliarity in range(0,1)
  return 1 - dice(y_true> y_true_perc, y_pred> y_pred_perc)

def comp_dice_Nd(Y_true, Y_pred, perc=95 ):
  shp = Y_true.shape[:-1]
  # flatten all but last dimension
  yt = Y_true.reshape(-1,shp[-1]) 
  yp = Y_pred.reshape(-1,shp[-1]) 
  return np.array([comp_dice(yt[i], yp[i]) for i in range(len(yt))])

def comp_corr(y_true, y_pred):
  """returns the correlation coeff between two 1D arrays via: np.corrcoef(y_true, y_pred)[0,1]"""
  return np.corrcoef(y_true , y_pred)[0,1]

#def r2_score()
                    
                    
## old:

# row wise correlation of two arrays 
def compute_corr_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


from sklearn.metrics import r2_score
from scipy.spatial.distance import cosine
comp_cosine_similiarity = lambda x,y: 1-cosine(x,y);


################### Further MEASURES FOR MODEL FIT ###############################
#https://github.com/taki0112/Vector_Similarity/blob/master/python/TS_SS/vector_similarity_vectorized.py

import math
import numpy as np
import torch

class TS_SS:
    
    def Cosine(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.dot(vec1, vec2.T)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def VectorSize(self, vec: np.ndarray):
        return np.linalg.norm(vec)

    def Euclidean(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.linalg.norm(vec1-vec2)

    def Theta(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.arccos(self.Cosine(vec1, vec2)) + np.radians(10)

    def Triangle(self, vec1: np.ndarray, vec2: np.ndarray):
        theta = np.radians(self.Theta(vec1, vec2))
        return (self.VectorSize(vec1) * self.VectorSize(vec2) * np.sin(theta))/2

    def Magnitude_Difference(self, vec1: np.ndarray, vec2: np.ndarray):
        return abs(self.VectorSize(vec1) - self.VectorSize(vec2))

    def Sector(self, vec1: np.ndarray, vec2: np.ndarray):
        ED = self.Euclidean(vec1, vec2)
        MD = self.Magnitude_Difference(vec1, vec2)
        theta = self.Theta(vec1, vec2)
        return math.pi * (ED + MD)**2 * theta/360


    def __call__(self, vec1: np.ndarray, vec2: np.ndarray):
        return self.Triangle(vec1, vec2) * self.Sector(vec1, vec2)
    
comp_tsss_similiarity = TS_SS()
    

################### Score Computation ###############################

def compute_scores(outp, targ, score_fn = r2_score, avg_target = None, scaled_resid=False, grayl=slice(None)):
  """
  takes outputs and targets of shape (n_subjs, n_tasks, n_vertices)
  and returns the scores of shape (n_subjs, n_tasks) computed with the
  given score_fn (default: sklearn.metrics.r2_score)
  
  Args:
        outp (np.array): outputs/predictions of shape (n_subjs, n_tasks, n_vertices)
        targ (np.array): the corresponding targets of the same shape as the outputs
        score_fn: function to compute the scores
        
        avg_target (np.array): 
           group mean (n_tasks, n_vertices) to be removed from both outp & targ 
           before scoring scoring to compute scores on the individual residuals
           
  Returns:
        np.array : Scores of shape (n_subjs, n_tasks).
  """
  if "torch.Tensor" in str(outp.__class__):
    outp= outp.cpu().detach().numpy()
  if "torch.Tensor" in str(targ.__class__):
    targ= targ.cpu().detach().numpy()   

  outp = outp[:,:,grayl]
  targ = targ[:,:,grayl]
    
  if not(avg_target is None):
    targ = batch_resid(targ, avg_target, fit_intercept=False) if scaled_resid else targ - avg_target         # (19, 10, 32492)
    outp = batch_resid(outp, avg_target, fit_intercept=False) if scaled_resid else outp - avg_target
    
  n_subjs = len(targ)
  n_tasks = targ.shape[1]
  scores = np.zeros((n_subjs, n_tasks));
  for sn in range(n_subjs):
    scores[sn, :]= np.array([score_fn(targ[sn,tn], outp[sn,tn]) for tn in range(n_tasks)])
    
  return scores;


from sklearn.utils import Bunch
def compute_all_scores(outp, targ, scores = {"r2_score": r2_score}, **kwargs):
  score_dict = {}
  for key, score_fn in scores.items():
    scores = compute_scores(outp, targ, score_fn = score_fn, **kwargs); 
    score_dict.update({key : scores})
                
  return Bunch(**score_dict)
                      


## Individual differentiability ...

def compute_batch_differentiability_score(A, B, verbose=False, return_corrmats=False, reduce="mean"):
    #xc = compute_corr_coeff(A,B)
    #print(A.__class__)
    if "torch.Tensor" in str(A.__class__):
       A= A.cpu().detach().numpy()
    if "torch.Tensor" in str(B.__class__):
       B= B.cpu().detach().numpy()   
    
    n_items= B.shape[0]
    n_tasks = B.shape[1]
    #print(B.shape, n_items, n_tasks)
    
    corrmats=[]
    diff_scores=np.zeros(n_tasks)
    for nt in range(n_tasks):
        corr_mat = np.corrcoef(A[:,nt,:],B[:,nt,:])
        corr_mat = corr_mat[:n_items,n_items:]
        corrmats.append(corr_mat)
        diagonal_corrs = np.diag(corr_mat);
        corr_tmp=corr_mat.copy()
        np.fill_diagonal(corr_tmp, np.nan)
        avg_cross_corrs = np.nanmean(corr_tmp,axis=0); 
        diff_scores[nt] = np.mean(diagonal_corrs- avg_cross_corrs);
        
        if verbose ==2:
            print("Task number:",tn)
            print("Diagonal corrs:", diagonal_corrs)
            print("Avg cross corrs:", avg_cross_corrs)
            print("diff_score:", diff_scores[tn])
    
    
    if verbose==1:
        print("diff_scores mean:", diff_scores.mean(), "| detailed:", diff_scores)
    
    diff_scores = diff_scores.mean() if reduce=="mean" else diff_scores;
    return diff_scores if not return_corrmats else (diff_scores, corrmats);
 



                    
################### MEASURES FOR MODEL FIT ###############################
                    
"""                    
def validate(output, target, avg_target = None, grayl=slice(None)):
  outp = output#.cpu().detach().numpy()   # (19, 10, 32492)
  targ = target#.cpu().detach().numpy()   # (19, 10, 32492)

  outp = outp[:,:,grayl]
  targ = targ[:,:,grayl]
  print(outp.shape, targ.shape)

  #avg_target = targ.mean(axis=0)         # (10, 32492)
  #avg_target = retest_upper_baseline["mean_task_maps"] # (10, 29696)
  if not(avg_target is None):
    targ = targ - avg_target         # (19, 10, 32492)
    outp = outp - avg_target
  
  print(targ.shape, outp.shape)
  reshaped_output = np.swapaxes(outp, 0, 1)
  reshaped_target = np.swapaxes(targ, 0, 1)
  print(reshaped_output.shape, reshaped_target.shape)

  corrs = np.diag(compute_corr_coeff(reshaped_output.reshape(reshaped_output.shape[0], -1), reshaped_target.reshape(reshaped_target.shape[0], -1)))
  return corrs
"""  


################### DUAL REGRESSION ###############################


#https://github.com/CoBrALab/RABIES/blob/019f06e61adb0e45ace53cc1a0b486ce16c8fffb/rabies/analysis_pkg/analysis_functions.py
#https://github.com/CoBrALab/RABIES/blob/788e22d4e0da41a5cc432a2f3a66c84362fae2ad/rabies/analysis_pkg/analysis_math.py

# functions that computes the Least Squares Estimates
def closed_form(X, Y, intercept=False):  
    if intercept:
        X = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)
    return np.linalg.inv(X.dot(X.transpose())).dot(X).dot(Y.transpose())

## Function
from sklearn.utils import Bunch

def compute_dual_regression(all_IC_vectors, timeseries):
    """
    needs regressors in shape: n_regr x n_verts (Q x V), and timeseries of shape time x n_verts (T x V)
    yields ...
      Bunch.A: time_course for each of the group components from Sg of shape (T x Q)
           .Si: (variance normalized) estimates of subject-level ICs (spatial maps) of shape ??? (Q x V)
           .VS: per component variances?
    """

    # mentions: https://mandymejia.com/2018/03/29/the-role-of-centering-in-dual-regression/
    Sg = all_IC_vectors    # shape (Q x V)
    Y = timeseries        # shape (T x V)

    ########
    ## In Regression 1, the group ICs are regressed against the subject’s fMRI data to estimate the subject-level time courses associated with each IC
    # Y_i = Ai Si + Ei
    # Y_i.T = Sg.T Ai.T + Ei.T
    # Ai.T = (Sg Sg.T)^-1 Sg Yi.T
    A = closed_form(Sg, Y, intercept=False).T # W 
    # A is the time_course for each of the group components from Sg
    # A isof shape (T x Q)

    # the temporal domain is variance-normalized so that the weights are contained in the spatial maps
    ## idk what this step really does; is this the "centering" in the temporal domain? looks a bit late for that
    A /= np.sqrt((A ** 2).mean(axis=0)) 

    ########
    ## In Regression 2, those estimated time courses are regressed against the subject’s fMRI data to obtain estimates of subject-level ICs (spatial maps).
    # Yi = Ai.dot(Si) + E, and hence via OLS?:
    # Si = np.inv(Ai.T.dot(Ai)).dot(Ai.T).dot(Y)
    Si = closed_form(A.T, Y.T, intercept=False)

    VS = np.sqrt((Si ** 2).mean(axis=0)) # the component variance/scaling is taken from the spatial maps
    Si /= VS # the spatial maps are variance normalized; the variance is stored in S

    # we thus output a model of the timeseries of the form X = W.dot((S*C).T)
    DR = {'Si': Si, 'A':A, 'VS':VS}
    return Bunch(Si=Si, A=A, VS=VS)



## rank_based_inverse_normal_transformation

#https://github.com/edm1/rank-based-INT/blob/master/rank_based_inverse_normal_transformation.py
import scipy.stats as ss

def rank_based_inverse_normalization(data, c=3.0/8, stochastic=True):
  #data = x= np.array([1,4,6,2,1,23])
  #np.random.seed(123)
  n=len(data)
  assert len(data.shape)==1
  if stochastic == True:
    perm = np.random.permutation(np.arange(n))
    rank = ss.rankdata(data[perm], method="ordinal")
    undo_perm = np.argsort(perm)
    transformed = ss.norm.ppf((rank - c)/(n-2*c+1))[undo_perm]
  else:
    rank = ss.rankdata(data, method="average")
    transformed = ss.norm.ppf((rank - c)/(n-2*c+1))
  return transformed;