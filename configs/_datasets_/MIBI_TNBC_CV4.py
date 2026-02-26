used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/MIBI_TNBC/used_markers.txt'
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/MIBI_TNBC/MIBI_TNBC.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/MIBI_TNBC/cv_splits/split_4/train.txt'
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/MIBI_TNBC/cv_splits/split_4/val.txt'

ignore_annotation = ['Unidentified']

patch_size = 32
n_markers = 37
cutter_size = 20

preprocess = None

dataset_kwargs = dict(
    h5_filepath=h5_filepath,
    used_markers=used_markers,
    patch_size=patch_size,
    ignore_annotation=ignore_annotation,
)

dataset = dict(
    type='MCIDataset'
)
