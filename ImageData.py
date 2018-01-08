import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from astropy.utils.data import download_file
from astropy import wcs
from astropy.io import fits
import pandas as pd 
#from matplotlib.colors import LogNorm
#import csv
#from urllib.parse import quote
import Preprocessing
from IPython import embed
import os


def download_and_save(url):
        # print (url)
        # embed()
        image_file = download_file(url, cache=True)
        os.makedirs("data/"+os.path.dirname(path), exist_ok=True)
        hdu_list = fits.open(image_file)
        hdu_list.writeto("data/" + path, overwrite=True)

def preprocess_image(image_data):
        if np.all(image_data.shape >= (100, 100)):
                img = Preprocessing.crop_center(image_data, 100)
                return img
        else:
                return None

        # plt.imshow(np.sqrt(test-test.min()), cmap='gray')
        # plt.colorbar()
        #plt.show()
        # save_result = np.append(save_result, test)
        #save_result.append(test)
        #np.save(save_result)
        #embed()
        
def open_and_preprocess(path):
        with fits.open(path, memmap=False) as hdu_list:
                image_data = hdu_list[0].data
                img = preprocess_image(image_data)
                
        return img

if __name__ == "__main__": 
        data =pd.read_csv("image_info.csv", dtype={"specobjid":str})
        bands = "r" #"ugriz"
        images = []
        ssfrs = []
        specobjids = []
        N = len(data)
        
        counter = 0 #<---------------#
        #sizes = []
        for i in range(N):
                g=data.iloc[i]
                band_img = []
                
                # ---------- REMOVE START -----------                
                #path = '301/{g.run:06d}/{g.camcol:d}/{g.field:04d}/{band}/{g.specobjid}.fits'.format(g=g, band="r")
                #info = fits.info("data/" + path, output=False)
                
                #if info[0][5] > (64, 64): #<----------------#
                #        counter += 1
                #sizes.append(info[0][5][0])
                #print("{}/{}".format(i+1, N), end="\r")        
                #continue
                # ---------- REMOVE END ----------- 
                        
                for band in bands:
                        path = '301/{g.run:06d}/{g.camcol:d}/{g.field:04d}/{band}/{g.specobjid}.fits'.format(g=g, band=band)
                        url = 'https://sid.erda.dk/share_redirect/gicXMUqD0l/' + path
                        #download_and_save(url)

                        
                        #with fits.open("data/" + path) as hdu_list:
                         #       image_data = hdu_list[0].data
                          #      img = preprocess_image(image_data)
                                
                        #hdu_list = fits.open("data/" + path, memmap=False)
                        #image_data = hdu_list[0].data
                        #img = preprocess_image(image_data)
                        #del hdu_list[0].data
                        #hdu_list.close()
                        
                        img = open_and_preprocess("data/" + path)  
                             
                        if img is None:
                                break
                        else:
                                band_img.append(img)
                         

                if len(band_img) != len(bands):
                        continue
                else:
                        images.append(band_img) 
                        ssfrs.append(g.specsfr_median)
                        specobjids.append(g.specobjid)
                                        
                print("{}/{}".format(i+1, N), end="\r")

        #print("Counter = ", counter) #<----------------------#
        #sizes = np.array(sizes)
        #np.save("data/sizes.npy", sizes)
        #embed()
        #stop
        #print(size)
        images = np.array(images)
        #embed()
        images = images[:,None,...] #<------ Udkommentér når netværket skal køres på alle bånd
       # embed()        
        np.save("data/images.npy", images) 
        ssfrs = np.array(ssfrs)
        np.save("data/ssfrs.npy", ssfrs) 
        with open("data/specobjids.txt", "w") as specs:
                for sid in specobjids:
                        specs.write(sid+"\n")
                        #        np.savetxt("data/specobjids.txt", specobjids)


