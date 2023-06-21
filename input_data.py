import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

indian_pines = sio.loadmat('indian_pines/Indian_pines.mat')
indian_pines_corrected = sio.loadmat('indian_pines/Indian_pines_corrected.mat')
indian_pines_gt = sio.loadmat('indian_pines/Indian_pines_gt.mat')

indian_pines = indian_pines['indian_pines']
indian_pines_corrected = indian_pines_corrected['indian_pines_corrected']
indian_pines_gt = indian_pines_gt['indian_pines_gt']

plt.imshow(indian_pines_gt)
plt.colorbar()
plt.show()

gt_class = {0: 'unknown',
            1: 'alfalfa',
            2:'Corn—notill',
            3:'Corn mintill',
            4:'Corn',
            5:'Grass—pasture',
            6:'Grass-trees',
            7:'Grass—pasture-mowed',
            8:'Hay—windrowed',
            9:'Oats',
            10:'Soybean not ill',
            11:'Soybean mintill',
            12:'Soybean clean',
            13:'Wheat',
            14:'Woods',
            15:'Buildings—Grass—Trees—Drives',
            16:'Stone—Steel—Towers'}

gt_class_avg_spectra = {}


for i in np.unique(indian_pines_gt):
    gt_class_avg_spectra[i] = np.average(indian_pines_corrected[indian_pines_gt==i], axis=0)

for k,v in gt_class_avg_spectra.items():
    plt.plot(v, label=k)
plt.legend()
plt.show()


