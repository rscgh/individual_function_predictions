from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np


def run_cross_validated_model(model, X_pre, Y_pre, n_splits = 5, verbose=False, very_verbose=False, silent = False):

    if very_verbose: verbose = True;

    # X_pre of shape (n_subjs, n_vertices, n_parcels), i.e. (100, 29696, 224)
    # Y_pre of shape (n_subjs, n_vertices, n_tasks), i.e. (100, 29696, 7)
    n_subjs = X_pre.shape[0]
    n_tasks = Y_pre.shape[2]
    n_parcels = X_pre.shape[2]

    # split the 100 subjects in 5 different ways for training vs test sets
    kf = KFold(n_splits=n_splits)
    splits = kf.split(np.arange(n_subjs))#kf.get_n_splits(np.arange(n_subjs))

    # result arrays
    # array to store the results shape (2-for train, test, n_splits, 1 for overall + n_tasks)
    varexp = np.zeros((2, n_splits, n_tasks+1)) 
    # and one set for coefficients for each split
    # coefficients have the shape (n_tasks, n_parcels) i.e. (7, 224) 
    coefs = np.zeros((n_splits, n_tasks, n_parcels));

    # iterate over those test sets
    for split_n, (train_index, test_index) in enumerate(splits):

        if not very_verbose and not silent: print(f"{split_n+1}/{n_splits}", end = ", ")
        if very_verbose: print(f"SPLIT {split_n+1}/{n_splits}\nTRAIN:", train_index, "TEST:", test_index)

        # subselect subjects for the current training split, then flatten the arrays again
        #y_true_train = np.moveaxis(Y_pre[train_index], 1, 2).reshape((-1,n_tasks))    
        y_true_train = Y_pre[train_index].reshape((-1,n_tasks))    
        # y_true_train of shape (n_subjs*n_vertices, n_tasks)
        X_train = X_pre[train_index].reshape((y_true_train.shape[0], n_parcels))   
        # X_train of shape (n_subjs*n_vertices, n_parcels)

        # same for test split
        #y_true_test = np.moveaxis(Y_pre[test_index], 1, 2).reshape((-1,n_tasks))    # shape (n_subjs*n_vertices)
        y_true_test = Y_pre[test_index].reshape((-1,n_tasks))    # shape (n_subjs*n_vertices)
        X_test = X_pre[test_index].reshape((y_true_test.shape[0], n_parcels))   

        if very_verbose: print(y_true_train.shape, y_true_test.shape, X_train.shape, X_test.shape)

        model.fit(X_train, y_true_train)     
        coefs[split_n] = model.coef_; # (n_tasks, n_parcels) i.e. (7, 224) 

        #print(f"took {np.round(time.time() - start, 2)}s")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        expv = r2_score(y_true_train, y_pred_train)
        if verbose: print(f"Split {split_n+1} train r2 score: {expv}")
        varexp[0,split_n, 0] = expv;

        expv = r2_score(y_true_test, y_pred_test)
        if verbose: print(f"Split {split_n+1} test  r2 score: {expv}")
        varexp[1,split_n, 0] = expv;

        if very_verbose: print("Per Task, train r2 score:")
        for tn in range(n_tasks):
           expv = r2_score(y_true_train[:,tn], y_pred_train[:,tn])
           varexp[0,split_n, tn+1] = expv;
           if very_verbose: print(task_names[tn], np.round(expv,3), end=" ")

        if very_verbose: print("\nPer Task, test  r2 score:")
        for tn in range(n_tasks):
           expv = r2_score(y_true_test[:,tn], y_pred_test[:,tn])
           varexp[1,split_n, tn+1] = expv;
           if very_verbose: print(task_names[tn], np.round(expv,3), end=" ")

    # summary
    if verbose: 
      print("Variance explained across all {n_splits} TEST splits: mean=", varexp[1,:,0].mean(), ", per-split: ", varexp[1,:,0].mean() )
      print("Mean TEST variance per task: ", varexp[1,:,:].mean(axis=0))

    return varexp, coefs; 