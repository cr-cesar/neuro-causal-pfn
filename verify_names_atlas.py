import nibabel as nib, numpy as np
p = nib.load("data/atlases/functional_parcellation_2mm.nii.gz").get_fdata()
print("parcellation:", p.shape, "values:", np.unique(p))
r = nib.load("data/atlases/2mm_parcellations/receptor/1_hearing_Noradrenaline_Glutamate.nii.gz").get_fdata()
print("one receptor subnetwork:", r.shape, "n_values:", len(np.unique(r)), "first:", np.unique(r)[:10])