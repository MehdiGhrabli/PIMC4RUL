from Data_Treatment import get_period, get_sampled_varied
from GP_Code import get_GP 
import pandas as pd 
import numpy as np
import random
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from tqdm import tqdm

# This python is used to generate mechanical properties from different states (crack lengths) of runs to failure. 
# It uses numerical simulation results, presented in the "Simulated_data.csv" file in the "data" folder, to train a regression model.
# Support vector regression (SVR) was chosen as the regression algorithm

Init_len =0.3615 #Initial contact length
eps = Init_len*1e-8 #A small contact length equivalent to liftoff, used instead of 0 to avoid numerical errors
T_min=55 #Minimal temperature in the experiments

df =pd.read_csv("data/Simulated_data.csv") #Simulated data as a pandas dataframe

#Feautre engineering
df.insert(loc=1, column='Log_Contact_Radius_Length', value= np.log(df["Contact_Radius_Length"]))
df.insert(loc=2, column='Exp_Contact_Radius_Length', value= np.exp(df["Contact_Radius_Length"]))
df.insert(loc=4, column='Log_Contact_Radius_Width', value= np.log(df["Contact_Radius_Width"]))
df.insert(loc=5, column='Exp_Contact_Radius_Width', value= np.exp(df["Contact_Radius_Width"]))
df.insert(loc=7, column='Log_Average_Chip_1', value= np.log(df["Average_Chip_1"]))
df.insert(loc=8, column='Exp_Average_Chip_1', value= np.exp(df["Average_Chip_1"]))

#Normalization 
Mu = dict()
Sig = dict()
for col in df.columns:
  Mu[col]=df[col].mean()
  Sig[col]=df[col].std()
  df[col] = (df[col] - df[col].mean()) / df[col].std()

# Fitting the support vector regression 
# The input of each support vector regression is denoted as X, and consists of the contact radius lenght and the average chip temperature, alongside their exponential and logarithmic values, to introduce relevant features
# The output is denoted as y, and is 1 dimensional. Depending on the case, y can be either the average stress, strain, or displacement in either the contact or the metallization for either the heating or the cooling phase. This means that SVR is used to predicted 12 different variables
X=np.array(df[["Contact_Radius_Length","Log_Contact_Radius_Length","Exp_Contact_Radius_Length","Average_Chip_1","Log_Average_Chip_1","Exp_Average_Chip_1"]])
# SVR to predict average contact stress in heating phase  
y=np.array(df["Stress_Contact_1"])
svr_stress_contact_1 = SVR(kernel='rbf', C=100, gamma=0.1);svr_stress_contact_1.fit(X, y)

# SVR to predict average contact stress in cooling phase  
y=np.array(df["Stress_Contact_2"])
svr_stress_contact_2 = SVR(kernel='rbf', C=100, gamma=0.1);svr_stress_contact_2.fit(X, y)

# SVR to predict average metallization stress in heating phase 
y=np.array(df["Stress_Metal_1"])
svr_stress_Metal_1 = SVR(kernel='rbf', C=100, gamma=0.1);svr_stress_Metal_1.fit(X, y)

# SVR to predict average metallization stress in cooling phase 
y=np.array(df["Stress_Metal_2"])
svr_stress_Metal_2 = SVR(kernel='rbf', C=100, gamma=0.1);svr_stress_Metal_2.fit(X, y)

# SVR to predict average contact strain in heating phase 
y=np.array(df["Total_Strain_Contact_1"])
svr_strain_contact_1 = SVR(kernel='rbf', C=100, gamma=0.1);svr_strain_contact_1.fit(X, y)

# SVR to predict average contact strain in cooling phase 
y=np.array(df["Total_Strain_Contact_2"])
svr_strain_contact_2 = SVR(kernel='rbf', C=100, gamma=0.1);svr_strain_contact_2.fit(X, y)

# SVR to predict average metallization strain in heating phase 
y=np.array(df["Total_Strain_Metal_1"])
svr_strain_Metal_1 = SVR(kernel='rbf', C=100, gamma=0.1);svr_strain_Metal_1.fit(X, y)

# SVR to predict average metallization strain in cooling phase 
y=np.array(df["Total_Strain_Metal_2"])
svr_strain_Metal_2 = SVR(kernel='rbf', C=100, gamma=0.1);svr_strain_Metal_2.fit(X, y)

# SVR to predict average contact displacement in heating phase 
y=np.array(df["Total_Displacement_Contact_1"])
svr_deformation_contact_1 = SVR(kernel='rbf', C=100, gamma=0.1);svr_deformation_contact_1.fit(X, y)

# SVR to predict average contact displacement in cooling phase
y=np.array(df["Total_Displacement_Contact_2"])
svr_deformation_contact_2 = SVR(kernel='rbf', C=100, gamma=0.1);svr_deformation_contact_2.fit(X, y)

# SVR to predict average metallization displacement in heating phase
y=np.array(df["Total_Displacement_Metal_1"])
svr_deformation_Metal_1 = SVR(kernel='rbf', C=100, gamma=0.1);svr_deformation_Metal_1.fit(X, y)

# SVR to predict average metallization displacement in cooling phase
y=np.array(df["Total_Displacement_Metal_2"])
svr_deformation_Metal_2 = SVR(kernel='rbf', C=100, gamma=0.1);svr_deformation_Metal_2.fit(X, y)

# Importing runs to failure data, this portion can be modified by the user to import their own data. 
## Delta T = 70°C
Rth33L = pd.read_excel("data/Runs_to_failure_70/Rth Module 33L.xlsx");Rth34L = pd.read_excel("data/Runs_to_failure_70/Rth Module 34L.xlsx");Rth35L = pd.read_excel("data/Runs_to_failure_70/Rth Module 35L.xlsx");
Rth33H = pd.read_excel("data/Runs_to_failure_70/Rth Module 33H.xlsx");Rth34H = pd.read_excel("data/Runs_to_failure_70/Rth Module 34H.xlsx");Rth35H = pd.read_excel("data/Runs_to_failure_70/Rth Module 35H.xlsx");
## Delta T = 90°C
Rth27L = pd.read_excel("data/Runs_to_failure_90/Rth Module 27L.xlsx");Rth28L = pd.read_excel("data/Runs_to_failure_90/Rth Module 28L.xlsx");Rth29L = pd.read_excel("data/Runs_to_failure_90/Rth Module 29L.xlsx");
Rth27H = pd.read_excel("data/Runs_to_failure_90/Rth Module 27H.xlsx");Rth28H = pd.read_excel("data/Runs_to_failure_90/Rth Module 28H.xlsx");Rth29H = pd.read_excel("data/Runs_to_failure_90/Rth Module 29H.xlsx");
## Delta T = 110°C
Rth21L = pd.read_excel("data/Runs_to_failure_110/Rth Module 21L.xlsx");Rth22L = pd.read_excel("data/Runs_to_failure_110/Rth Module 22L.xlsx");Rth23L = pd.read_excel("data/Runs_to_failure_110/Rth Module 23L.xlsx");
Rth21H = pd.read_excel("data/Runs_to_failure_110/Rth Module 21H.xlsx");Rth22H = pd.read_excel("data/Runs_to_failure_110/Rth Module 22H.xlsx");Rth23H = pd.read_excel("data/Runs_to_failure_110/Rth Module 23H.xlsx");

# The list of data used for training, this should be adapted depending on the use case. Each element of the list is a tuple containing the run to failure data and the maximum temperature T_max
## For standard predictions, one run to failure should be removed from L_data, to then be used as the test run 
L_data = [(Rth33L,125),(Rth34L,125),(Rth35L,125),(Rth33H,125),(Rth34H,125),(Rth35H,125),(Rth27L,145),(Rth28L,145),(Rth29L,145),(Rth27H,145),(Rth28H,145),(Rth29H,145),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]
## For the extrapolation task, use L_data_extrap
L_data_extrap = [(Rth27L,145),(Rth28L,145),(Rth29L,145),(Rth27H,145),(Rth28H,145),(Rth29H,145),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]
## For the interpolation task, use L_data_interp
L_data_interp = [(Rth33L,125),(Rth34L,125),(Rth35L,125),(Rth33H,125),(Rth34H,125),(Rth35H,125),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]

#Calculating the step size from the training data 
per = 0
for i in range(len(L_data_extrap)):
    per+=get_period(L_data_extrap[i][0])
per/=(len(L_data_extrap))
per = int(np.floor(per))

# Create synthetic data with a uniform step size 
Synthetic_uniform_data = []
Base_data_size = int(0.05*per*len(L_data_extrap))  #Select the number of the synthetic runs to failure. This can largely be bigger than the size of the experimental runs to failure as multiple scenarios can be obtained, depending on the starting point

for i in tqdm(range(Base_data_size)):
    index_og_data = random.randint(0,len(L_data_extrap)-1)
    Synthetic_uniform_data.append((get_sampled_varied(L_data_extrap[index_og_data][0],per),L_data_extrap[index_og_data][1]))


gen_size = 2*Base_data_size # Select the size of the generated synthetic data, this is different from Base_data_size as the gaussian process regression can produce multiple results for the same synthetic run
Unique_data_size = len(Synthetic_uniform_data)
GP_Vce_to_Lc = get_GP("Cross section analysis.xlsx",0.2) #Gaussian process regression function

Regular_data_path = "Generated_data_for_extrapolation" #Output path
if os.path.exists(Regular_data_path) and os.path.isdir(Regular_data_path):
    shutil.rmtree(Regular_data_path)
os.mkdir(Regular_data_path) 

Normalized_data_path = "Generated_data_normalized_for_extrapolation" #Output path for normalized values
if os.path.exists(Normalized_data_path) and os.path.isdir(Normalized_data_path):
    shutil.rmtree(Normalized_data_path)
os.mkdir(Normalized_data_path) 

#Loop to generate runs to failure supplemented by the estimations of the corresponding mechanical properties
for i in tqdm(range(gen_size)):
    picked_index=random.randint(0, Unique_data_size-1)#Choose a random element from the synthetic data list, which will be the synthetic run for this instance of the loop
    current_data = Synthetic_uniform_data[picked_index][0]
    current_vce = current_data[1] #Vce values of the current run
    Candidate_Lcs=[] #Initializing the list of the candidate LC values, corresponding to current_vce
  
    Temp = (Synthetic_uniform_data[picked_index][1]- Mu["Average_Chip_1"])/Sig["Average_Chip_1"]#Normalized temperature of the run 
    Log_temp = (np.log(Synthetic_uniform_data[picked_index][1]) -Mu["Log_Average_Chip_1"])/Sig["Log_Average_Chip_1"]#Normalized value of the logarithm of the temperature of the run
    Exp_temp = (np.exp(Synthetic_uniform_data[picked_index][1])-Mu["Exp_Average_Chip_1"])/Sig["Exp_Average_Chip_1"]#Normalized value of the exponential of temperature of the run
    Names = ["Cycle","Contact_Radius_Length","Stress_Contact_1","Stress_Contact_2","Stress_Metal_1","Total_Strain_Contact_1","Total_Strain_Contact_2","Total_Strain_Metal_1","Total_Displacement_Contact_1","Total_Displacement_Contact_2","Total_Displacement_Metal_1","Tmin","DeltaT"] #Names of the used features
    unnormalized_df = pd.DataFrame(columns=Names)# Dataframe containing the results (contact length, temperature, features, mechanical features)
    normalized_df = pd.DataFrame(columns=Names)# Dataframe containing the normalized results
    for j in range(len(current_vce)):
        mu, sig = GP_Vce_to_Lc.predict(np.array(current_vce[j]*2).reshape(-1,1),return_std=True) # Mean and variance corresponding to the predicted lc value
        Sample = min(max(((1-(np.random.normal(loc=mu,scale=sig)/2 + 0.5))*Init_len)[0],eps),Init_len) # Sampling from the mean and the variance and transforming the crack length to contact length
        Norm_len =(Sample-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"] # Normalizing the contact length 
        Log_length = (np.log(Sample)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"]# Normalizing the log contact length
        Exp_length = (np.exp(Sample)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"]# Normalizing the exp contact length
        #Support vector regression to predict mechanical properties
        SC1= svr_stress_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SC2=svr_stress_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SM1=svr_stress_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnC1= svr_strain_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnC2= svr_strain_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnM1= svr_strain_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DC1= svr_deformation_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DC2= svr_deformation_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DM1=svr_deformation_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
      
        normalized_df = pd.concat([ normalized_df, pd.DataFrame([[current_data[0][j],Norm_len,SC1,SC2,SM1,SnC1,SnC2,SnM1,DC1,DC2,DM1,Temp,T_min]], columns=normalized_df.columns) ], ignore_index=True) #Dataframe containing normalized features and mechanical properties
        # Denormalizing mechanical properties
        Reg_SC1 = SC1*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"]
        Reg_SC2 = SC2*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"]
        Reg_SM1 = SM1*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"]
        Reg_SnC1 = SnC1*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"]
        Reg_SnC2 = SnC2*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"]
        Reg_SnM1 = SnM1*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"]
        Reg_DC1 = DC1*Sig["Total_Deformation_Contact_1"] +Mu["Total_Deformation_Contact_1"]
        Reg_DC2 = DC2*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"]
        Reg_DM1 = DM1*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"]
        Reg_props = [Reg_SC1,Reg_SC2,Reg_SM1,Reg_SnC1,Reg_SnC2,Reg_SnM1,Reg_DC1,Reg_DC2,Reg_DM1]
        # Dataframe containing denormalized features and mechanical properties
        unnormalized_df = pd.concat([ unnormalized_df, pd.DataFrame([[current_data[0][j],Sample,Reg_SC1,Reg_SC2,Reg_SM1,Reg_SnC1,Reg_SnC2,Reg_SnM1,Reg_DC1,Reg_DC2,Reg_DM1,Temp,T_min]], columns=unnormalized_df.columns) ], ignore_index=True)
    # Writing the resulting data frame in a csv file 
    unnormalized_df.to_csv(Regular_data_path+'/Unnormalized_' + str(i)+ '.csv', index=False)




