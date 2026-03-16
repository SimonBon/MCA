used_markers = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/used_markers.txt'
h5_filepath  = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/IMC_NB_TumorSub.h5'

train_indicies = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/train.txt'
val_indicies   = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/val.txt'
test_indicies  = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/MCI_data/h5_files/IMC_NB_TumorSub/test.txt'

# NK_DC has only 111 cells — too few for LP; 'Other' not biologically defined
ignore_annotation = ['Other']

patch_size  = 24
n_markers   = 31
cutter_size = 12

dataset = dict(
    type='MCIDataset',
    used_markers=used_markers,
    patch_size=patch_size,
    h5_filepath=h5_filepath,
    ignore_annotation=ignore_annotation,
)
