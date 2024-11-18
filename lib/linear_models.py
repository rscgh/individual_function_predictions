import numpy as np
import sklearn.linear_model

from sklearn.model_selection import train_test_split, ShuffleSplit
from himalaya.ridge import RidgeCV

from lib.stats import compute_all_scores, compute_scores
from lib.stats import comp_dice, comp_corr, r2_score, comp_tsss_similiarity, comp_cosine_similiarity
from lib.stats import compute_batch_differentiability_score

data_dir = "/scratch/users/robert.scholz2/acc_dists/"
#task_data_file = "all_10_tasks_254_full_unrelated.raw.npy"
# subjs=np.loadtxt("data/subjs_hcp254_full_unrelated.txt").astype(int).astype(str);

def predict_y_from_x(xdata, ydata, verbose=0, lmodel="ridge", alphas = "auto"): #, pert=None):
  # alpha = 0 ~ similair to linear regression; higher alpha ~ more regularization
  x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, shuffle=False)
  if verbose: print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
  x = x_train.reshape((-1, x_train.shape[-1]))
  #if not(pert is None): x=  x[:, pert] = 0
  y = y_train.reshape((-1, y_train.shape[-1]))
  if verbose: print(x.shape, y.shape)
  
  if lmodel == "ridge": 
    if isinstance(alphas, str) and alphas == "auto":
      alphas = approximate_best_alphas(x, y, n_samples = 29696)
      if verbose: print("approx. alphas:", alphas);
    skr = sklearn.linear_model.Ridge(alpha =  alphas) # abest_alphas
  elif lmodel == "linear":
    skr = sklearn.linear_model.LinearRegression()
  else: raise Exception(f"Desired linear model {lmodel} does not exist/is not implemented");
  
  skr.fit(x, y)
  y_train_pred =skr.predict(x).reshape(y_train.shape)
  xt = x_test.reshape((-1, x_test.shape[-1]))
  #if not(pert is None): xt[:, pert] = 0
  y_test_pred =skr.predict(xt).reshape(y_test.shape)
  return y_train, y_train_pred, y_test, y_test_pred, skr;
    
def predict_from_modality_file(subjs, predictor_data_file, task_data_file, data_dir= data_dir, resid_task_maps=False):
  isubjs, xdata, ydata = load_xy_data(subjs, data_dir, predictor_data_file, task_data_file, resid_task_maps=resid_task_maps)
  #n_predictors = xdata.shape[-1] 
  return (isubjs, ) + predict_y_from_y(isubjs, xdata, ydata);

## just roguhly estimate the alphas on a subset of data

def approximate_best_alphas(x,y, n_samples = 29696, \
      test_alphas = np.logspace(1, 20, 20), cv = ShuffleSplit(n_splits=1, test_size=0.5, random_state=36)):
  kernel_ridge_cv = RidgeCV(alphas=test_alphas, cv=cv)
  indices = (np.random.rand(n_samples)* len(x)).astype(int)
  kernel_ridge_cv.fit(x[indices], y[indices])
  return kernel_ridge_cv.best_alphas_
  

def score(y_train_pred, y_train, desired_scores = {"corr": comp_corr, "r2_score": r2_score}):
  vpredicted = np.swapaxes(y_train_pred, 1,2)
  vtargets = np.swapaxes(y_train, 1,2)  
  scoresd = compute_all_scores(vtargets, vpredicted, scores = desired_scores)
  diff_scores_test = compute_batch_differentiability_score(vtargets, vpredicted, reduce=False)
  return scoresd, diff_scores_test