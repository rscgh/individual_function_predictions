# Contents

### Task analysis scripts & notebooks

| File                                               | Description                                                                                                                                  | Figures |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `download_hcp_data.ipynb`                          | download resting brain surfaces, state runs, task maps and (freesurfer-derived) structural maps                                              |         |
| `hcp_task_retest_baseline.ipynb`                   | Notebook to compute the test-retest baselines (accuracy, discriminability, vertex-wise-scores ...)                                           |         |
| **Common functionality**                           | The scripts located in the `lib/` folder                                                                                                     |         |
| `lib/data_loading.py`                              | Some helper functions to load data, e.g. `load_xy_data`, `gather_modalities`                                                                                                   |         |
| `lib/linear_models.py`                             | Helper functions for model fitting and prediction, e.g. `predict_y_from_x, score`                                                            |         |
| `lib/plotting.py`                                  | Some custom plotting functions (as surfplot cannot be used on our current cluster as no x-server is available), e.g. `plot_bars`, `plot_29k` |         |
| `lib/stats.py`                                     | Functions to compute some statstics & scores, e.g. `compute_all_scores`, `comp_dice`, or `comp_corr`                                         |         |
| **Preparation**                                    | i.e. preprocessing nessesary for each of the predictors                                                                                      |         |
| **`prepare_rs_gradients.ipynb`**                   | Computation of resting-state functional connectivity components                                                                              |         |
| `prepare_rs_ica.ipynb`                             | Similiar process for the ICA components                                                                                                      |         |
| `prepare_distances.ipynb`                          | Computation of distances (vertex-to-parcels/landmarks) and PCA of the full vertex-to-vertex distance matrix                                  |         |
| `prepare_structural_eigenmodes.ipynb`              | Computation of the structural eigenmodes of the individual left-hemisphere cortical surfaces                                                 |         |
| `prepare_blueprints.ipynb`                         | This notebook loads individual blueprints (prev computed by FSL XTract) and concatenates them into a single file.                            |         |
| **`prepare_task_maps_(pred target).ipynb`**        | Concatenates task contrasts for each subject into a single file                                                                              |         |
| **Model fitting and analysis**                     |                                                                                                                                              |         |
| **`linear_models.ipynb`**                          | The full model fitting pipeline for all modalities, along with plotting of the main results                                                  |         |
| **`linear_models_pca_component_annotation.ipynb`** | Annotation of the main RS-PCA components underlying the best performing model through correlation with known maps of brain organization      |         |
| `linear_models_feature_contributions_(pca).ipynb`  | Alternative ways of assessing the importance of each RS-PCA component as feature in the linear model.                                        |         |
| tbd: figure plotting script                        |                                                                                                                                              |         |
| **Validation**                                     |                                                                                                                                              |         |
| `replicate_on_right_hemisphere.ipynb`              | Replication of the full model fitting pipeline based on data from the right hemisphere in the same subject (includes preparation & model fitting)      |         |
| `cneuromod_validation.ipynb`                       | Test of the generalizeability of the previously fitted linear model (based on HCP subjects) to a new dataset (CNeuroMod) + Replication of the full model fitting pipeline based on CNeuroMod data of 3 subjects  (includes download, task-contrast computation, preparation & model fitting) |         |
| **Other**                                          | (not part of the manuscript)                                                                                                                 |         |
| `linear_models_yresid.ipynb`                       |                                                                                                                                              |         |
| `ann_results_summary.ipynb`                        | summarizes the NN results                                                                                                                    |         |
| `BrainSurfCNN.ipynb`                               |                                                                                                                                              |         |


<br>

### Dependencies 

These scripts make wide usage of the following python libraries: `nibabel`, `nilearn`, `sklearn`, `surfplot`, `brainspace`, and `hcp_utils`. 

Some scripts contain commands to connectome workbench as well.


### Resources and results

+ `results/retest_hcp45.task_maps.npy` - task maps for the 46 retest subjects (dict, keys: task_names, values: list of tuples (test_contrast, retest_contrast), each contrast of shape (29696))

+ `results/retest_hcp45.test.mean_task_maps.npy` - array of shape (task, vertcies)

+ `results/scores/retest_hcp45.retest_scores.npy` - as dict

# Preprint / Cite ...

The Courtois project on neural modelling was made possible by a generous donation from the Courtois foundation, administered by the Fondation Institut Gériatrie Montréal at CIUSSS du Centre-Sud-de-l’île-de-Montréal and the University of Montreal. The Courtois NeuroMod team is based at “Centre de Recherche de l’Institut Universitaire de Gériatrie de Montréal”, with several other institutions involved. See the CNeuroMod documentation for an up-to-date list of contributors (https://docs.cneuromod.ca).
