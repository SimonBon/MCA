used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_FineCT/used_markers.txt'
h5_filepath = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_FineCT/IMC_NB_FineCT.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_FineCT/train.txt'
val_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_FineCT/val.txt'
test_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_FineCT/test.txt'

# 'Other' is excluded — not a biologically defined cell type
ignore_annotation = ['Other']

patch_size = 24
n_markers = 31
cutter_size = 12

dataset = dict(
    type='MCIDataset',
    used_markers=used_markers,
    patch_size=patch_size,
    h5_filepath=h5_filepath,
    ignore_annotation=ignore_annotation
)
