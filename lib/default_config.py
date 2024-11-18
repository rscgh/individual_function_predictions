
# contains all the hcp subjects for which all the required contrasts exist?
full_subj_path  = "/home/users/robert.scholz2/bigdata/hcp_full_subjects.txt"


## Task Fmri files
tmap_type = "tstat1";
smooth_lv = "2"; 
tmsmall = "_MSMAll" if smooth_lv in ['2','4'] else ""; 

task_fmri_sam   = "{subj}/MNINonLinear/Results/tfMRI_{task}/tfMRI_{task}_hp200_s{smooth_lv}_level2{tmsmall}.feat/GrayordinatesStats/cope{cope_num}.feat/{tmap_type}.dtseries.nii";

rest_sessions = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL" ];

rest_file_stub = "HCP_1200/{subj}/MNINonLinear/Results/{session}/{session}_Atlas_MSMAll_hp2000_clean.dtseries.nii";

task_info = """WM 19 BODY-AVG
WM 20 FACE-AVG
WM 21 PLACE-AVG
WM 22 TOOL-AVG
GAMBLING 6 REWARD-PUNISH
MOTOR 21 AVG-CUE
LANGUAGE 4 STORY-MATH
SOCIAL 6 TOM-RANDOM
RELATIONAL 4 REL-MATCH
EMOTION 3 FACES-SHAPES"""


contrast_info = [x.split() for x in task_info.split("\n")]
contrast_info

task_names = task_info.split("\n");

structural_file_templates = [
  "{subj}/MNINonLinear/fsaverage_LR32k/{subj}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii",
  "{subj}/MNINonLinear/fsaverage_LR32k/{subj}.corrThickness_MSMAll.32k_fs_LR.dscalar.nii",
  "{subj}/MNINonLinear/fsaverage_LR32k/{subj}.curvature_MSMAll.32k_fs_LR.dscalar.nii",
  "{subj}/MNINonLinear/fsaverage_LR32k/{subj}.sulc_MSMAll.32k_fs_LR.dscalar.nii",
  "{subj}/MNINonLinear/fsaverage_LR32k/{subj}.thickness_MSMAll.32k_fs_LR.dscalar.nii"]