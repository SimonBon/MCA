used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/used_markers.txt'
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/CODEX_cHL.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/train.txt'
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/val.txt'
test_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/CODEX_cHL/test.txt'

ignore_annotation = ['Seg Artifact']

patch_size = 32
n_markers = 41
cutter_size = 24

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