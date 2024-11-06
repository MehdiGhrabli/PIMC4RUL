import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# This python file contains the implementation of the gaussian process regression, linking the degradation to the health indicator. 
# It contains one function, get_GP, which performs the regression, returning the regression function 
# The input values are : 
#     data_path : the path of the dataset containing Lc and Vce values, in our case it is the "Cross section analysis.xlsx" file
#     noise_std : The noise standard deviation used in the regression, equal to 0.2 by default. This parameter controls the smoothness of the predictions.
# The ouput value is a function that takes as input the Vce value and returns the corresponding crack length 
def get_GP(data_path,noise_std =0.2):
    Cross_section_analysis = pd.read_excel(data_path)
    #Data preprocessing
    Cross_section_analysis_cleaned = Cross_section_analysis.iloc()[:176,:] #Removing invalid rows
    # Scaling the contact area
    Cross_section_analysis_cleaned['Contact area reduction %']/=100 #Scaling from 0 to 1
    Cross_section_analysis_cleaned['Contact area reduction %']-=0.5 #Scaling from -0.5 to 0.5
    Cross_section_analysis_cleaned['Contact area reduction %']*=2 #Scaling rom -1 to 1
    # Scaling the number of cycles
    Cross_section_analysis_cleaned['Number of cycles']/=np.max(Cross_section_analysis_cleaned['Number of cycles']) #Scale from 0 to 1
    # Scaling the collector emitter voltage     
    Cross_section_analysis_cleaned['delta Vce / Vce % ']*=2
    
    # Initializing the model 
    kernel = 1 * RBF(length_scale=0.01, length_scale_bounds=(1e-4, 1e2))
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=20
    )
    
    # formatting the data
    X_train= np.array(Cross_section_analysis_cleaned['delta Vce / Vce % '])
    y = np.array(Cross_section_analysis_cleaned['Contact area reduction %'])
    X_train = np.concatenate([np.array(1000*[0]),X_train])
    y = np.concatenate([np.array(1000*[-1]),y])
    Sort_indexx = X_train.argsort()
    X_train = np.sort(X_train)
    y = y[Sort_indexx]
    X_train= X_train.reshape(-1,1)
    y = y.reshape(-1,1)
    
    # Fitting the model 
    gaussian_process.fit(X_train, y)
    
    return gaussian_process 
