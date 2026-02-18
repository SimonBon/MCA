used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/used_markers.txt'
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/CODEX_DLBCL2.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/train.txt'
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/val.txt'
test_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_DLBCL2/test.txt'

ignore_annotation = ['na']

patch_size = 24
n_markers = 40
cutter_size = 12 

dataset = dict(
    type='MCIDataset',
    used_markers=used_markers,
    patch_size=patch_size,
    h5_filepath=h5_filepath,
    ignore_annotation = ignore_annotation
)