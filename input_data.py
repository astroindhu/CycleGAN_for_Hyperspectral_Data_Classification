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


