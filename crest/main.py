import numpy as np
import PyPhenom as ppi
from datetime import datetime
from tqdm import tqdm
import time 
#%%

phenom = ppi.Phenom( 'xxxxYourPhenomAPIxxxx')
#phenom.SemAutoContrastBrightness() 

#%%
phenom.SetSemBrightness(0.477)
phenom.SetSemContrast(2.0)

#%%
'''
index=np.linspace(-3,-5,360)


time.sleep(2)
for scale in tqdm(range(len(index))):
    phenom.SetHFW(10**index[scale])
    phenom.SemAutoFocus()
    
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(1600, 1600)
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 8
    acqScanParams.hdr = False
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)
    ppi.Save(acqWithDatabar, 'zoom//zoom_%d.tiff'%(scale))
    time.sleep(2)
'''
#%%
phenom.SetHFW(10e-6)  

row = 30
col = 30
X_abs= 0.0
#phenom.GetStageModeAndPosition().position.x
Y_abs= 0.0
#phenom.GetStageModeAndPosition().position.y

horizontal_field_width = phenom.GetHFW()

X = np.linspace(X_abs, X_abs + horizontal_field_width * (col - 1)*0.6, col)
Y = np.linspace(Y_abs, Y_abs - horizontal_field_width * (row - 1)*0.6, row)


#%%
#phenom.SetHFW(2e-5)
for i in range(row):
    for j in tqdm(range(col)):
        phenom.MoveTo(X[j], Y[i])
        phenom.SemAutoFocus()
        #phenom.SemAutoContrastBrightness() 

        acqScanParams = ppi.ScanParams()
        acqScanParams.size = ppi.Size(1600, 1600)
        acqScanParams.detector = ppi.DetectorMode.All
        acqScanParams.nFrames = 16
        acqScanParams.hdr = False
        acqScanParams.scale = 1.0
        acq = phenom.SemAcquireImage(acqScanParams)
        acq.metadata.displayWidth = 0.5
        acq.metadata.dataBarLabel = "Label"
        acqWithDatabar = ppi.AddDatabar(acq)
        ppi.Save(acqWithDatabar, 'pd1//row%d_col%d.tiff'%(i,j))
        time.sleep(2)
#%%

phenom.SetHFW(30e-6)  

row = 30
col = 30
X_abs= -0.0
#phenom.GetStageModeAndPosition().position.x
Y_abs= 0.0
#phenom.GetStageModeAndPosition().position.y

horizontal_field_width = phenom.GetHFW()

X = np.linspace(X_abs, X_abs + horizontal_field_width * (col - 1)*0.9, col)
Y = np.linspace(Y_abs, Y_abs - horizontal_field_width * (row - 1)*0.9, row)
#phenom.SetHFW(2e-5)
for i in range(row):
    for j in tqdm(range(col)):
        phenom.MoveTo(X[j], Y[i])
        phenom.SemAutoFocus()

        acqScanParams = ppi.ScanParams()
        acqScanParams.size = ppi.Size(1600, 1600)
        acqScanParams.detector = ppi.DetectorMode.All
        acqScanParams.nFrames = 16
        acqScanParams.hdr = False
        acqScanParams.scale = 1.0
        acq = phenom.SemAcquireImage(acqScanParams)
        acq.metadata.displayWidth = 0.5
        acq.metadata.dataBarLabel = "Label"
        acqWithDatabar = ppi.AddDatabar(acq)
        ppi.Save(acqWithDatabar, 'pd2//row%d_col%d.tiff'%(i,j))
        time.sleep(2)
        
        
        
index=np.linspace(-3,-6,400)

from tqdm import tqdm
import time 
phenom.MoveTo(0,0)
for scale in tqdm(range(len(index))):
    phenom.SetHFW(10**index[scale])
    phenom.SemAutoFocus()
    
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(1600, 1600)
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16
    acqScanParams.hdr = False
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)
    ppi.Save(acqWithDatabar, 'pd3//zoom_%d.tiff'%(scale))
    time.sleep(2)
