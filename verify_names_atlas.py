import nibabel as nib, numpy as np
p = nib.load("data/atlases/functional_parcellation_2mm.nii.gz").get_fdata()
print("parcelacion:", p.shape, "valores:", np.unique(p))
r = nib.load("data/atlases/2mm_parcellations/receptor/1_hearing_Noradrenaline_Glutamate.nii.gz").get_fdata()
print("una subred receptor:", r.shape, "n_valores:", len(np.unique(r)), "primeros:", np.unique(r)[:10])