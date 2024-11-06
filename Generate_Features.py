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
# import warnings
# warnings.filterwarnings("ignore")
Init_len =0.3615
eps = Init_len*1e-8
T_min=55
df =pd.read_csv("Simulated_data.csv")
df.insert(loc=1, column='Log_Contact_Radius_Length', value= np.log(df["Contact_Radius_Length"]))
df.insert(loc=2, column='Exp_Contact_Radius_Length', value= np.exp(df["Contact_Radius_Length"]))
df.insert(loc=4, column='Log_Contact_Radius_Width', value= np.log(df["Contact_Radius_Width"]))
df.insert(loc=5, column='Exp_Contact_Radius_Width', value= np.exp(df["Contact_Radius_Width"]))
df.insert(loc=7, column='Log_Average_Chip_1', value= np.log(df["Average_Chip_1"]))
df.insert(loc=8, column='Exp_Average_Chip_1', value= np.exp(df["Average_Chip_1"]))

################################################################## SVR ###############################################################
Mu = dict()
Sig = dict()
for col in df.columns:
  Mu[col]=df[col].mean()
  Sig[col]=df[col].std()
  df[col] = (df[col] - df[col].mean()) / df[col].std()


X=np.array(df[["Contact_Radius_Length","Log_Contact_Radius_Length","Exp_Contact_Radius_Length","Average_Chip_1","Log_Average_Chip_1","Exp_Average_Chip_1"]])

y=np.array(df["Stress_Contact_1"])

svr_stress_contact_1 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_stress_contact_1.fit(X, y)
y=np.array(df["Stress_Contact_2"])
svr_stress_contact_2 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_stress_contact_2.fit(X, y)

y=np.array(df["Stress_Metal_1"])

svr_stress_Metal_1 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_stress_Metal_1.fit(X, y)

y=np.array(df["Stress_Metal_2"])

svr_stress_Metal_2 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_stress_Metal_2.fit(X, y)


y=np.array(df["Total_Strain_Contact_1"])

svr_strain_contact_1 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_strain_contact_1.fit(X, y)

y=np.array(df["Total_Strain_Contact_2"])

svr_strain_contact_2 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_strain_contact_2.fit(X, y)

y=np.array(df["Total_Strain_Metal_1"])

svr_strain_Metal_1 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_strain_Metal_1.fit(X, y)

y=np.array(df["Total_Strain_Metal_2"])

svr_strain_Metal_2 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_strain_Metal_2.fit(X, y)


y=np.array(df["Total_Deformation_Contact_1"])

svr_deformation_contact_1 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_deformation_contact_1.fit(X, y)
y=np.array(df["Total_Deformation_Contact_2"])

svr_deformation_contact_2 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_deformation_contact_2.fit(X, y)

y=np.array(df["Total_Deformation_Metal_1"])

svr_deformation_Metal_1 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_deformation_Metal_1.fit(X, y)
y=np.array(df["Total_Deformation_Metal_2"])

svr_deformation_Metal_2 = SVR(kernel='rbf', C=100, gamma=0.1)
svr_deformation_Metal_2.fit(X, y)
##############################################################################################################################################################

## Importing Data for T_ref = 55°C 
Rth33L = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 33L.xlsx");Rth34L = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 34L.xlsx");Rth35L = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 35L.xlsx");
Rth33H = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 33H.xlsx");Rth34H = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 34H.xlsx");Rth35H = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 35H.xlsx");

Rth27L = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 27L.xlsx");Rth28L = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 28L.xlsx");Rth29L = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 29L.xlsx");
Rth27H = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 27H.xlsx");Rth28H = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 28H.xlsx");Rth29H = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 29H.xlsx");

Rth21L = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 21L.xlsx");Rth22L = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 22L.xlsx");Rth23L = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 23L.xlsx");
Rth21H = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 21H.xlsx");Rth22H = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 22H.xlsx");Rth23H = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 23H.xlsx");

L_data = [(Rth33L,125),(Rth34L,125),(Rth35L,125),(Rth33H,125),(Rth34H,125),(Rth35H,125),(Rth27L,145),(Rth28L,145),(Rth29L,145),(Rth27H,145),(Rth28H,145),(Rth29H,145),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]
L_data_unseen = [(Rth27L,145),(Rth28L,145),(Rth29L,145),(Rth27H,145),(Rth28H,145),(Rth29H,145),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]
L_data_interp = [(Rth33L,125),(Rth34L,125),(Rth35L,125),(Rth33H,125),(Rth34H,125),(Rth35H,125),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]
per = 0

# for i in range(len(L_data)):
#     per+=get_period(L_data[i][0])
# per/=(len(L_data))

for i in range(len(L_data_unseen)):
    per+=get_period(L_data_unseen[i][0])
per/=(len(L_data_unseen))
per = int(np.floor(per))
print(per)
Periodic_data = []
# Base_data_size = int(0.05*per*len(L_data))
Base_data_size = int(0.05*per*len(L_data_unseen))
print("Base_data_size : " , Base_data_size)
# for i in tqdm(range(Base_data_size)):
#     index_og_data = random.randint(0,len(L_data)-1)
#     Periodic_data.append((get_sampled_varied(L_data[index_og_data][0],per),L_data[index_og_data][1]))

for i in tqdm(range(Base_data_size)):
    index_og_data = random.randint(0,len(L_data_unseen)-1)
    Periodic_data.append((get_sampled_varied(L_data_unseen[index_og_data][0],per),L_data_unseen[index_og_data][1]))


gen_size = 2*Base_data_size
Unique_data_size = len(Periodic_data)
GP_Vce_to_Lc = get_GP("Cross section analysis.xlsx")
Regular_data_path = "Generated_data_for_extrapolation"
if os.path.exists(Regular_data_path) and os.path.isdir(Regular_data_path):
    shutil.rmtree(Regular_data_path)

os.mkdir(Regular_data_path) 

Normalized_data_path = "Generated_data_normalized_for_extrapolation"
if os.path.exists(Normalized_data_path) and os.path.isdir(Normalized_data_path):
    shutil.rmtree(Normalized_data_path)

os.mkdir(Normalized_data_path) 

## Can be optimized
for i in tqdm(range(gen_size)):
    picked_index=random.randint(0, Unique_data_size-1)
    current_data = Periodic_data[picked_index][0]
    current_vce = current_data[1]
    Candidate_Lcs=[]
    Temp = (Periodic_data[picked_index][1]- Mu["Average_Chip_1"])/Sig["Average_Chip_1"]
    Log_temp = (np.log(Periodic_data[picked_index][1]) -Mu["Log_Average_Chip_1"])/Sig["Log_Average_Chip_1"]
    Exp_temp = (np.exp(Periodic_data[picked_index][1])-Mu["Exp_Average_Chip_1"])/Sig["Exp_Average_Chip_1"]
    Names = ["Cycle","Contact_Radius_Length","Stress_Contact_1","Stress_Contact_2","Stress_Metal_1","Total_Strain_Contact_1","Total_Strain_Contact_2","Total_Strain_Metal_1","Total_Deformation_Contact_1","Total_Deformation_Contact_2","Total_Deformation_Metal_1","Tmin","DeltaT"]
    unnormalized_df = pd.DataFrame(columns=Names)
    normalized_df = pd.DataFrame(columns=Names)
    for j in range(len(current_vce)):
        mu, sig = GP_Vce_to_Lc.predict(np.array(current_vce[j]*2).reshape(-1,1),return_std=True)
        mu =mu
        sig = sig
        Sample = min(max(((1-(np.random.normal(loc=mu,scale=sig)/2 + 0.5))*Init_len)[0],eps),Init_len)
        # print("Sample is ", Sample)
        Norm_len =(Sample-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"]
        Log_length = (np.log(Sample)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"]
        Exp_length = (np.exp(Sample)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"]
        # print("Norm len is ", Norm_len)
        # print("Log_len is : ", Log_length)
        # print("Exp len is : ", Exp_length)
        # print("All is ", np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))
        SC1= svr_stress_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SC2=svr_stress_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SM1=svr_stress_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # SM2= svr_stress_Metal_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnC1= svr_strain_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnC2= svr_strain_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnM1= svr_strain_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # SnM2= svr_strain_Metal_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DC1= svr_deformation_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DC2= svr_deformation_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DM1=svr_deformation_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # DM2= svr_deformation_Metal_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]

        normalized_df = pd.concat([ normalized_df, pd.DataFrame([[current_data[0][j],Norm_len,SC1,SC2,SM1,SnC1,SnC2,SnM1,DC1,DC2,DM1,Temp,T_min]], columns=normalized_df.columns) ], ignore_index=True)
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

        unnormalized_df = pd.concat([ unnormalized_df, pd.DataFrame([[current_data[0][j],Sample,Reg_SC1,Reg_SC2,Reg_SM1,Reg_SnC1,Reg_SnC2,Reg_SnM1,Reg_DC1,Reg_DC2,Reg_DM1,Temp,T_min]], columns=unnormalized_df.columns) ], ignore_index=True)
        #Normalized_data_frame
        #Unnormalized dataframe
        # Candidate_Lcs.append(Sample)
    unnormalized_df.to_csv(Regular_data_path+'/Unnormalized_' + str(i)+ '.csv', index=False)
    # normalized_df.to_csv(Normalized_data_path+'/Normalized_'+ str(i)+'.csv', index=False)



