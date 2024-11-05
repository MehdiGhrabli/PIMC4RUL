import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def get_GP(data_path,noise_std =0.2):
    Cross_section_analysis = pd.read_excel(data_path)
    Cross_section_analysis_cleaned = Cross_section_analysis.iloc()[:176,:]
    Cross_section_analysis_cleaned['Contact area reduction %']/=100 #Scale from 0 to 1
    Cross_section_analysis_cleaned['Number of cycles']/=np.max(Cross_section_analysis_cleaned['Number of cycles']) #Scale from 0 to 1
    Cross_section_analysis_cleaned['delta Vce / Vce % ']/=100 #From percentage to raw value
    Cross_section_analysis_cleaned['Contact area reduction %']-=0.5
    Cross_section_analysis_cleaned['Contact area reduction %']*=2 #From -1 to 1
    Cross_section_analysis_cleaned['delta Vce / Vce % ']*=200 #Doubling percentage
    X_train= np.array(Cross_section_analysis_cleaned['delta Vce / Vce % '])
    y = np.array(Cross_section_analysis_cleaned['Contact area reduction %'])
    X_train = np.concatenate([np.array(1000*[0]),X_train])
    y = np.concatenate([np.array(1000*[-1]),y])
    Sort_indexx = X_train.argsort()
    X_train = np.sort(X_train)
    y = y[Sort_indexx]
    X_train= X_train.reshape(-1,1)
    y = y.reshape(-1,1)
    kernel = 1 * RBF(length_scale=0.01, length_scale_bounds=(1e-4, 1e2))
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=20
    )
    gaussian_process.fit(X_train, y)
    return gaussian_process 
