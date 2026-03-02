# CODEX_cHL restricted to the 18-marker panel used by KRONOS (arXiv:2506.03373).
# Enables a like-for-like LP accuracy comparison with KRONOS's reported 0.736.
#
# NOTE: this config uses the same HDF5 as the standard CODEX_cHL run.
# FoxP3 and CD56 are included (present in the 49-marker AllMarkers HDF5)
# even though they were excluded from the standard 41-marker panel due to
# poor CODEX staining SNR. If runs look unreasonably bad, try removing them
# from the markers file and setting n_markers = 16.
#
# Dataset discrepancy vs KRONOS:
#   KRONOS cHL: 134,552 cells, 16 phenotypes
#   Our cHL:    ~115k cells, 17 phenotypes
# The difference likely comes from different segmentation/QC. The comparison
# is approximate, not exact.

used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MCA/configs/_markers_/CODEX_cHL_KRONOS18.txt'
h5_filepath  = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt'
val_indicies   = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/val.txt'
test_indicies  = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/test.txt'

ignore_annotation = ['Seg Artifact']

patch_size = 32
n_markers  = 18
cutter_size = 24

preprocess = None

dataset_kwargs = dict(
    h5_filepath=h5_filepath,
    used_markers=used_markers,
    patch_size=patch_size,
    used_indicies=train_indicies,
    ignore_annotation=ignore_annotation,
    preprocess=preprocess,
)

dataset = dict(
    type='MCIDataset'
)
