used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/used_markers.txt'
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/CODEX_DLBCL2_with_norm_stats.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/train_LN0251.txt'
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/val_LN0265.txt'
test_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/test_LN0265.txt'

ignore_annotation = ['na']

patch_size = 24
n_markers = 40
cutter_size = 12

preprocess=None

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