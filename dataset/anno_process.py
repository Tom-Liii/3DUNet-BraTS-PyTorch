# This is used to process the annotation file, resample into the same size as the image
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import os

def resample_3d_array(array, target_size):
    current_size = array.shape
    if current_size == target_size:
        return array

    zoom_factors = (np.array(target_size) / np.array(current_size)).tolist()
    resampled_array = zoom(array, zoom_factors, order=0)  # Using order=1 for linear interpolation

    return resampled_array

# This is used to process the annotation file, by translating (moving) the annotation
import numpy as np

def translate_3d_array(array, offset_x, offset_y, offset_z):
    translated_array = np.roll(array, (offset_x, offset_y, offset_z), axis=(0, 1, 2))
    return translated_array


if __name__ == '__main__':
    # # Define the paths to the source and target NIfTI files
    # source_path = "/media/mrjiang/DATADISK/Dataset/GBM_post/annotation/2/2_060719_mask.nii.gz"
    # target_path = "/media/mrjiang/DATADISK/Dataset/GBM_post/brats_processed/2/2_060719/FL_to_SRI.nii.gz"

    # target_size = [155,240,240]
    # # Load the source and target NIfTI files
    # source_image = sitk.ReadImage(source_path)
    # source_arr = sitk.GetArrayFromImage(source_image)

    # tar_arr = resample_3d_array(source_arr, target_size)
    # print(tar_arr.shape)

    # resampled_image = sitk.GetImageFromArray(tar_arr)
    # resampled_image.SetSpacing(source_image.GetSpacing())
    # resampled_image.SetOrigin(source_image.GetOrigin())
    # resampled_image.SetDirection(source_image.GetDirection())


    # output_path = "/media/mrjiang/DATADISK/Dataset/GBM_post/resampled_5.nii.gz"
    # sitk.WriteImage(resampled_image, output_path)


    id = 10
    date = '300118'
    source_path = f"/home/tom/Documents/y4_t1/fyp/brats2023/annotation_resampled/{id}/{id}_{date}_mask.nii.gz"
    # find all the masks under source path
    masks = [source_path]
    # for root, dirs, files in os.walk(source_path):
    #     for file in files:
    #         if file.endswith('.nii.gz'):
    #             masks.append(os.path.join(root, file))
    translated_masks = [x.replace('annotation_resampled', 'annotation_translated') for x in masks]
    # output_path = "/home/tom/Documents/y4_t1/fyp/brats2023/annotation_translated/2"

    print(f'Total masks: {len(masks)}')
    count = 0
    for mask in masks:
        # Load the source and target NIfTI files
        source_image = sitk.ReadImage(mask)
        source_arr = sitk.GetArrayFromImage(source_image)
        
        tar_arr = translate_3d_array(source_arr, 5, 30, 0) # x, y, z

        tar_image = sitk.GetImageFromArray(tar_arr)
        tar_image.SetSpacing(source_image.GetSpacing())
        tar_image.SetOrigin(source_image.GetOrigin())
        tar_image.SetDirection(source_image.GetDirection())

        sitk.WriteImage(tar_image, translated_masks[count])
        count += 1