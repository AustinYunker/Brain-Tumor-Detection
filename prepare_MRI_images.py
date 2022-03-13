import numpy as np
from PIL import Image
import os

def load_MRI_scans(path_to_no, path_to_tumor):
    """
    This function loads the MRI images into two lists based on if they contain a tumor. 
    
    params:
        path_to_no - path to the directory containing images without tumors
        path_to_tumors - path to the directory containing images with tumors
        
    returns:
        tumors - list of images
    """
    
    no_tumor_lst = list()
    for MRI in os.listdir(path_to_no):
        full_path = os.path.join(path_to_no, MRI)
        no_tumor_lst.append(Image.open(full_path))
    
    tumor_lst = list()
    for MRI in os.listdir(path_to_tumor):
        full_path = os.path.join(path_to_tumor, MRI)
        tumor_lst.append(Image.open(full_path))
        
    tumors = no_tumor_lst + tumor_lst
    return tumors

def resize_MRI(lst, width, height):
    """
    This function resized the MRI images.
    
    params:
        lst - list containing the images
        width - specified width
        height - specified height
        
    returns:
        lst
    """
    lst = [MRI.resize((width, height)) for MRI in lst]
    
    return lst

def MRI_color_scheme(lst, color_scheme):
    """
    This function converts the MRI images to the specified color
    
    params:
        lst - list containing MRI images
        color_scheme - string containing the specified color
        
    returns:
        lst
    """
    lst = [MRI.convert(color_scheme) for MRI in lst]
    return lst

def pixel_normalization(lst):
    """
    This function normalizes the pixel values in the MRI images to be between 0 and 1
    
    params:
        lst - list containing MRI images
        
    returns:
        mri_array - array containing arrays of images
    """
    mri_array = np.array([np.array(MRI) for MRI in lst])
    mri_array = mri_array.astype('float32')/255
    return mri_array

def create_target(num_no_tumors, num_tumors):
    """
    This function creates the target array of 0s and 1s based on how many images are found to have tumors or no tumors
    
    params:
        num_no_tumors - number of non tumor images
        num_tumors - number of tumor images
        
    returns:
        target - array of 0s and 1s with length num_no_tumors + num_tumors
    """
    target = np.array([0]*num_no_tumors + [1]*num_tumors)
    return target

def clean_MRI_scans(path_to_no_tumors, path_to_tumors, width, height, color, num_no_tumors, num_tumors):
    """
    This function is a wrapper over various functions to clean the MRI images.
    
    params:
        path_to_no_tumors - directory path to non tumor images
        path_to_tumors - directly path to tumor images
        width - specified image width
        height - specified image height
        color - specified color scheme
        num_no_tumors - number of non tumor images
        num_tumors - number of tumor images
        
    returns:
        mri - array of image arrays
        target - array of 0s and 1s 
    
    """
    
    mri = load_MRI_scans(path_to_no_tumors, path_to_tumors)
    
    mri = resize_MRI(mri, width = width, height = height)
    
    mri = MRI_color_scheme(mri, color_scheme=color)
    mri = pixel_normalization(mri)
    
    target = create_target(num_no_tumors=num_no_tumors, num_tumors=num_tumors)
    
    return mri, target

