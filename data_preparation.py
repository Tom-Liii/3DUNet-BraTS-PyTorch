# mask location: /home/tom/Documents/y4_t1/fyp/annotation_registered
# data location: /home/tom/Documents/y4_t1/fyp/brats_processed
# dig into mask location, find all the masks, and then find the corresponding data in data location
# then save the data and mask into a new location
# data location: /home/tom/Documents/y4_t1/fyp/brats_2023
# generate a csv file for the data and mask with columns: name, split(train, val, test)
import os
mask_location = '/home/tom/Documents/y4_t1/fyp/annotation_registered'
data_location = '/home/tom/Documents/y4_t1/fyp/brats_processed'
new_data_location = '/home/tom/Documents/y4_t1/fyp/brats2023/brats2023'
# find all the masks
masks = []
for root, dirs, files in os.walk(mask_location):
    for file in files:
        if file.endswith('.nii.gz'):
            masks.append(os.path.join(root, file))
print('masks: ', len(masks), masks)
csv_data = []
for mask in masks:
    # go to data location and find the corresponding data
    id = mask.split('/')[-1].split('.')[0]
    csv_data.append([id, 'train'])
    data = mask.split('.')[0].replace('annotation_registered', 'brats_processed')
    # copy the data and mask to new location if not exist
    if not os.path.exists(new_data_location):
        os.system(f'mkdir {new_data_location}')
    if not os.path.exists(f'{new_data_location}/{id}'):
        os.system(f'mkdir {new_data_location}/{id}')
    # only copy the data that ends with `_to_SRI.nii.gz`
    os.system(f'cp {data}/*_to_SRI.nii.gz {new_data_location}/{id}/')
    os.system(f'cp {mask} {new_data_location}/{id}/{id}_seg.nii.gz')
    # data files end with `_to_SRI.nii.gz` change to `_flair.nii.gz`, `_t1.nii.gz`, `_t1ce.nii.gz`, `_t2.nii.gz` accordingly
    os.system(f'mv {new_data_location}/{id}/T1CE_to_SRI.nii.gz {new_data_location}/{id}/{id}_t1ce.nii.gz')
    os.system(f'mv {new_data_location}/{id}/T1_to_SRI.nii.gz {new_data_location}/{id}/{id}_t1.nii.gz')
    os.system(f'mv {new_data_location}/{id}/T2_to_SRI.nii.gz {new_data_location}/{id}/{id}_t2.nii.gz')
    os.system(f'mv {new_data_location}/{id}/FL_to_SRI.nii.gz {new_data_location}/{id}/{id}_flair.nii.gz')

# generate csv file, columns: name, split(train, val, test)
import csv
with open('/home/tom/Documents/y4_t1/fyp/3DUNet-BraTS-PyTorch/data/split/brats2023_split_fold0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'split'])
    writer.writerows(csv_data)