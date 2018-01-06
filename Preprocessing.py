# Denne fil tager en fil/image og cutter 100x100px fra centrum. 
import numpy as np
from astropy.utils.data import download_file
from astropy import wcs
from astropy.io import fits
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed



# function that crops the galaxy pictures to 64x64 px
def crop_center(img, size):
    cropx = size#64
    cropy = size#64
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

#def pre_proc_output():

# plt.imshow(np.sqrt(test-test.min()), cmap='gray', vmax=0.1, vmin=0.1)
# plt.colorbar()
# plt.show()


# function that tests if the images of the galaxies aree too small to be run, and saves these en a array 
def size_of_img(img): 
    counter = 0
    if np.all(img.shape > (50, 50)):
        counter += 1
    return counter

if(__name__ == "__main__"):
    image_file = fits.open('1398488404370417664.fits')
    image_data = image_file[0].data
    test = crop_center(image_data)
    test_size = size_of_img(test)
    print(test_size)
    # print(size_of_img(test))
