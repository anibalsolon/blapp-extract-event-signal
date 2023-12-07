import sys
import numpy as np
import nibabel as nib
import pandas as pd

from nilearn.glm.first_level import compute_regressor

fmri = nib.load(sys.argv[1])
fmri_data = fmri.get_fdata()
events_data = pd.read_csv(sys.argv[2], sep="\t")
events = list(map(lambda e: e.strip(), sys.argv[3].split(',')))
frame_times = np.linspace(0, fmri_data.shape[-1], fmri_data.shape[-1])

event_onsets = events_data[events_data['trial_type'].isin(events)]

signal, _ = compute_regressor(
    (event_onsets.onset, event_onsets.duration, event_onsets.weight),
    'spm',
    frame_times,
    con_id="condition",
    oversampling=1,
)

fmri_selected_data = fmri_data[..., signal.flatten() > 0]
nib.Nifti1Image(fmri_selected_data, fmri.affine).to_filename('output/bold.nii.gz')

event_onsets.to_csv('output/events.tsv', sep='\t', index=False, header=True)