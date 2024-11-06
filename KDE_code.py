from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.stats
from Data_Treatment import get_period, get_sampled, get_sampled_varied
from GP_Code import get_GP 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import random
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
# import warnings
# warnings.filterwarnings("ignore")
random.seed(3)
############################################################# Useful functions ##################################################
def conditional_sample(kde, training_data, columns, values, samples = 100):
    if len(values) - len(columns) != 0:
        raise ValueError("length of columns and values should be equal")
    if training_data.shape[1] - len(columns) != 1:
        raise ValueError(f'Expected {training_data.shape[1] - 1} columns/values but {len(columns)} have be given')  
    cols_values_dict = dict(zip(columns, values))
    
    #create array to sample from
    steps = 10000  
    X_test = np.zeros((steps,training_data.shape[1]))
    for i in range(training_data.shape[1]):
        col_data = training_data.iloc[:, i]
        X_test[:, i] = np.linspace(col_data.min(),col_data.max(),steps)
    for col in columns: 
        idx_col_training_data = list(training_data.columns).index(col)
        idx_in_values_list = list(cols_values_dict.keys()).index(col)
        X_test[:, idx_col_training_data] = values[idx_in_values_list] 
        
    #compute probability of each sample
    prob_dist = np.exp(kde.score_samples(X_test))/np.exp(kde.score_samples(X_test)).sum()
    
    #sample using np.random.choice
    return X_test[np.random.choice(range(len(prob_dist)), p=prob_dist, size = (samples))]

########################################################################################

GP_Vce_to_Lc = get_GP("Cross section analysis.xlsx")
Vce_anchors = np.linspace(0,12,1000).reshape(-1,1)
Lc_continuous = GP_Vce_to_Lc.predict(Vce_anchors,return_std=False)

Init_len =0.3615
eps = Init_len*0.0000001
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

################################################ Importing data ###########################################################################

## Importing Data for T_ref = 55°C 

Rth33L = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 33L.xlsx");Rth34L = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 34L.xlsx");Rth35L = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 35L.xlsx");
Rth33H = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 33H.xlsx");Rth34H = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 34H.xlsx");Rth35H = pd.read_excel("Cycling tests/VceN/Campaign_1/Rth Module 35H.xlsx");

Rth27L = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 27L.xlsx");Rth28L = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 28L.xlsx");Rth29L = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 29L.xlsx");
Rth27H = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 27H.xlsx");Rth28H = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 28H.xlsx");Rth29H = pd.read_excel("Cycling tests/VceN/Campaign_8/Rth Module 29H.xlsx");

Rth21L = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 21L.xlsx");Rth22L = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 22L.xlsx");Rth23L = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 23L.xlsx");
Rth21H = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 21H.xlsx");Rth22H = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 22H.xlsx");Rth23H = pd.read_excel("Cycling tests/VceN/Campaign_9/Rth Module 23H.xlsx");


L_data = [(Rth33L,125),(Rth34L,125),(Rth35L,125),(Rth33H,125),(Rth34H,125),(Rth35H,125),(Rth27L,145),(Rth28L,145),(Rth29L,145),(Rth27H,145),(Rth28H,145),(Rth29H,145),(Rth21L,165),(Rth22L,165),(Rth23L,165),(Rth21H,165),(Rth22H,165),(Rth23H,165)]
per = 0
for i in range(len(L_data)):
    per+=get_period(L_data[i][0])
per/=(len(L_data))
per = int(np.floor(per))
print(per)
test_data = L_data[0]
# test_data = L_data[-1]
#####################################################################################################################################

######################################################### Fitting KDE ############################################################
History_size = 2
Prediction_size = 1
print("Begin")
data_size=40
path_of_data = "C:\\Users\Mehdi-GHRABLI\\Desktop\\EsrefSims\\Generated_data_regular_new_improved"
Features = ["Contact_Radius_Length","Stress_Contact_1","Total_Strain_Contact_1","Total_Deformation_Contact_1"]
feature_length = len(Features)-1
onlyfiles = [f for f in listdir(path_of_data)[:data_size] if isfile(join(path_of_data, f))]

Container = []
for i in range(feature_length*History_size+Prediction_size):
    Container.append([])
### Create points 
for small_path in onlyfiles:
    full_path_data = path_of_data + "\\" + small_path
    current_df = pd.read_csv(full_path_data)
    current_df = current_df[Features]
    for i in range(History_size,len(current_df)-Prediction_size+1):
        delta_lc = current_df["Contact_Radius_Length"].iloc[i]-current_df["Contact_Radius_Length"].iloc[i-1]
        Container[0].append(delta_lc)
        for k in range(History_size):
            for l in range(feature_length): 
                Container[(l+1)+k*feature_length].append(current_df[Features[l+1]].iloc[i-(k+1)])
Train_data = np.zeros((len(Container[0]),len(Container)))
norm_lis=[]
max_len_delta = np.max(Container[0])
min_len_delta = np.min(Container[0])

max_SC1 = max(np.max(Container[1]),np.max(Container[4]))
min_SC1 = min(np.min(Container[1]),np.min(Container[4]))

max_SnC1 = max(np.max(Container[2]),np.max(Container[5]))
min_SnC1 = min(np.min(Container[2]),np.min(Container[5]))

max_DC1 = max(np.max(Container[3]),np.max(Container[6]))
min_DC1 = min(np.min(Container[3]),np.min(Container[6]))

Train_data[:,0] = (np.array(Container[0])-min_len_delta)/(max_len_delta-min_len_delta)
Train_data[:,1] = (np.array(Container[1])-min_SC1)/(max_SC1-min_SC1)
Train_data[:,2] = (np.array(Container[2])-min_SnC1)/(max_SnC1-min_SnC1)
Train_data[:,3] = (np.array(Container[3])-min_DC1)/(max_DC1-min_DC1)
Train_data[:,4] = (np.array(Container[4])-min_SC1)/(max_SC1-min_SC1)
Train_data[:,5] = (np.array(Container[5])-min_SnC1)/(max_SnC1-min_SnC1)
Train_data[:,6] = (np.array(Container[6])-min_DC1)/(max_DC1-min_DC1)

kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(Train_data)
###################################################################################################################################

#################################################### Starting Loop ################################################################
profile_temp = test_data[1]
Temp = (profile_temp- Mu["Average_Chip_1"])/Sig["Average_Chip_1"]
Log_temp = (np.log(profile_temp) -Mu["Log_Average_Chip_1"])/Sig["Log_Average_Chip_1"]
Exp_temp = (np.exp(profile_temp)-Mu["Exp_Average_Chip_1"])/Sig["Exp_Average_Chip_1"]
begin_point = 10
num_iterations =  int((np.array(test_data[0]["N° de cycle"])[-1]-np.array(test_data[0]["N° de cycle"])[begin_point-1])/per)
N_cycle,Vce_cycle=get_sampled(test_data[0].iloc[:begin_point],per)

mu1, sig1 = GP_Vce_to_Lc.predict(np.array(Vce_cycle[-1]*2).reshape(-1,1),return_std=True) ; mu2, sig2 = GP_Vce_to_Lc.predict(np.array(Vce_cycle[-2]*2).reshape(-1,1),return_std=True)


Sample1 = min(max(((1-(np.random.normal(loc=mu1,scale=sig1)/2 + 0.5))*Init_len)[0],eps),Init_len); Sample2 = min(max(((1-(np.random.normal(loc=mu2,scale=sig2)/2 + 0.5))*Init_len)[0],eps),Init_len); 

print("Sample 1 is ", Sample1)
print("Sample 2 is ", Sample2)
Norm_len1 =(Sample1-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"];Norm_len2 =(Sample2-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"];
Log_length1 = (np.log(Sample1)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"];Log_length2 = (np.log(Sample2)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"]
Exp_length1 = (np.exp(Sample1)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"];Exp_length2 = (np.exp(Sample2)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"]

SC1_1= svr_stress_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SC1_2= svr_stress_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
SnC1_1= svr_strain_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SnC1_2= svr_strain_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
DC1_1= svr_deformation_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];DC1_2= svr_deformation_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];

Reg_SC1_1_out = SC1_1*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];Reg_SC1_2_out = SC1_2*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];
# Reg_SC2_1 = SC2_1*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"];Reg_SC2_2 = SC2_2*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"];
# Reg_SM1_1 = SM1_1*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"];Reg_SM1_2 = SM1_2*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"];
Reg_SnC1_1_out = SC1_1*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];Reg_SnC1_2_out = SnC1_2*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];
# Reg_SnC2_1 = SnC2_1*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"];Reg_SnC2_2 = SnC2_2*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"];
# Reg_SnM1_1 = SnM1_1*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"];Reg_SnM1_2 = SnM1_2*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"];
Reg_DC1_1_out = DC1_1*Sig["Total_Deformation_Contact_1"] +Mu["Total_Deformation_Contact_1"];Reg_DC1_2_out = DC1_2*Sig["Total_Deformation_Contact_1"] +Mu["Total_Deformation_Contact_1"];
# Reg_DC2_1 = DC2_1*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"];Reg_DC2_2 = DC2_2*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"];
# Reg_DM1_1 = DM1_1*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"];Reg_DM1_2 = DM1_2*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"];


# Reg_props = [Reg_SC1,Reg_SC2,Reg_SM1,Reg_SnC1,Reg_SnC2,Reg_SnM1,Reg_DC1,Reg_DC2,Reg_DM1]
Needed_props = [(Reg_SC1_1_out-min_SC1)/(max_SC1-min_SC1)  ,(Reg_SnC1_1_out-min_SnC1)/(max_SnC1-min_SnC1) ,(Reg_DC1_1_out-min_DC1)/(max_DC1-min_DC1),(Reg_SC1_2_out-min_SC1)/(max_SC1-min_SC1)  ,(Reg_SnC1_2_out-min_SnC1)/(max_SnC1-min_SnC1) ,(Reg_DC1_2_out-min_DC1)/(max_DC1-min_DC1) ]
print("Mechanical values : ")
print(Needed_props)
train_data_df = pd.DataFrame(Train_data)
train_data_df.columns=["Contact_Radius_Length","Stress_Contact_1_close","Total_Strain_Contact_1_close","Total_Deformation_Contact_1_close","Stress_Contact_1_far","Total_Strain_Contact_1_far","Total_Deformation_Contact_1_far"]
Cols = ["Stress_Contact_1_close","Total_Strain_Contact_1_close","Total_Deformation_Contact_1_close","Stress_Contact_1_far","Total_Strain_Contact_1_far","Total_Deformation_Contact_1_far"]
New_lengths_list = []
old_length = Sample2
new_length = Sample1
print("Old length is : ", old_length)
print("new length is : ", new_length )
Num_monte_iterations=50
# num_iterations=int(np.floor(np.array(test_data[0]["N° de cycle"])[-1]/per))
num_iterations = 5
# num_iterations = int(np.ceil((np.array((test_data[0]["N° de cycle"]))[-1]-begin_point*per)/per))

scenarios_lc = np.zeros((Num_monte_iterations,num_iterations))
scenarios_Vce = np.zeros((Num_monte_iterations,num_iterations))
for m in tqdm(range(Num_monte_iterations)):
    # for i in tqdm(range(num_iterations)):
    New_lengths_list = []
    old_length = Sample2
    new_length = Sample1
    Needed_props = [(Reg_SC1_1_out-min_SC1)/(max_SC1-min_SC1)  ,(Reg_SnC1_1_out-min_SnC1)/(max_SnC1-min_SnC1) ,(Reg_DC1_1_out-min_DC1)/(max_DC1-min_DC1),(Reg_SC1_2_out-min_SC1)/(max_SC1-min_SC1)  ,(Reg_SnC1_2_out-min_SnC1)/(max_SnC1-min_SnC1) ,(Reg_DC1_2_out-min_DC1)/(max_DC1-min_DC1) ]

    for i in tqdm(range(num_iterations)):   

        #Generate Candidate with KDe 
        #May need to change train_data to dataframe

        Candidate_delta_lc = conditional_sample(kde, train_data_df, Cols, Needed_props, samples = 100) #Returns normalized values
        Candidate_delta_lc=np.mean(Candidate_delta_lc[:,0])

        Candidate_delta_lc = Candidate_delta_lc*(max_len_delta-min_len_delta)+min_len_delta # True value
        old_length = new_length
        new_length = max(old_length + Candidate_delta_lc,eps)
        New_lengths_list.append(new_length)
        Norm_len1 =(new_length-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"];Norm_len2 =(old_length-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"];
        Log_length1 = (np.log(new_length)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"];Log_length2 = (np.log(old_length)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"]
        Exp_length1 = (np.exp(new_length)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"];Exp_length2 = (np.exp(old_length)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"]

        SC1_1= svr_stress_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SC1_2= svr_stress_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # SC2_1=svr_stress_contact_2.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SC2_2=svr_stress_contact_2.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # SM1_1=svr_stress_Metal_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SM1_2=svr_stress_Metal_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        SnC1_1= svr_strain_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SnC1_2= svr_strain_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # SnC2_1= svr_strain_contact_2.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SnC2_2= svr_strain_contact_2.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        # SnM1_1= svr_strain_Metal_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SnM1_2= svr_strain_Metal_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
        DC1_1= svr_deformation_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];DC1_2= svr_deformation_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];
        # DC2_1= svr_deformation_contact_2.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];DC2_2= svr_deformation_contact_2.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];
        # DM1_1=svr_deformation_Metal_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];DM1_2=svr_deformation_Metal_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];

        Reg_SC1_1 = SC1_1*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];Reg_SC1_2 = SC1_2*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];
        # Reg_SC2_1 = SC2_1*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"];Reg_SC2_2 = SC2_2*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"];
        # Reg_SM1_1 = SM1_1*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"];Reg_SM1_2 = SM1_2*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"];
        Reg_SnC1_1 = SnC1_1*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];Reg_SnC1_2 = SnC1_2*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];
        # Reg_SnC2_1 = SnC2_1*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"];Reg_SnC2_2 = SnC2_2*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"];
        # Reg_SnM1_1 = SnM1_1*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"];Reg_SnM1_2 = SnM1_2*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"];
        Reg_DC1_1 = DC1_1*Sig["Total_Deformation_Contact_1"] +Mu["Total_Deformation_Contact_1"];Reg_DC1_2 = DC1_2*Sig["Total_Deformation_Contact_1"] +Mu["Total_Deformation_Contact_1"];
        # Reg_DC2_1 = DC2_1*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"];Reg_DC2_2 = DC2_2*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"];
        # Reg_DM1_1 = DM1_1*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"];Reg_DM1_2 = DM1_2*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"];
        Needed_props = [(Reg_SC1_1-min_SC1)/(max_SC1-min_SC1)  ,(Reg_SnC1_1-min_SnC1)/(max_SnC1-min_SnC1) ,(Reg_DC1_1-min_DC1)/(max_DC1-min_DC1),(Reg_SC1_2-min_SC1)/(max_SC1-min_SC1)  ,(Reg_SnC1_2-min_SnC1)/(max_SnC1-min_SnC1) ,(Reg_DC1_2-min_DC1)/(max_DC1-min_DC1) ]
    New_lengths_arr = np.array(New_lengths_list)
    print("New lengths array : ")
    print(New_lengths_arr)
    scenarios_lc[m,:]=New_lengths_arr
    New_cracks = Init_len-New_lengths_arr
    print("New cracks array : ")
    print(New_cracks)
    New_lengths_arr_normed = 2*(New_cracks/Init_len -0.5)
    New_Vces_normed = np.interp(New_lengths_arr_normed.flatten(),Lc_continuous.flatten(),Vce_anchors.flatten())
    New_Vces = New_Vces_normed/2
    print("New Vces : ")
    print(New_Vces)
    scenarios_Vce[m,:]=New_Vces
    # print("New Vces : ",New_Vces )
    # print("New lengths are : ")
    # print(New_lengths_list)

    # Infer Mechanical Parameters using SVR 
added_cycles = np.arange(1,num_iterations+1)*per+test_data[0]["N° de cycle"][begin_point-1]
All_cycles_in_pred = np.concatenate([test_data[0]["N° de cycle"][:begin_point-1],added_cycles])
mean_vce = np.mean(scenarios_Vce,axis=0)
std_vce = np.std(scenarios_Vce,axis=0)
confidence = 0.90
h = std_vce * scipy.stats.t.ppf((1 + confidence) / 2., Num_monte_iterations-1)
lower_bound = mean_vce-h
lower_bound =np.concatenate([np.array(test_data[0]["%Vce corrigé"][:begin_point-1]),lower_bound])
upper_bound = mean_vce+h
upper_bound =np.concatenate([np.array(test_data[0]["%Vce corrigé"][:begin_point-1]),upper_bound])

all_vces_in_pred= np.concatenate([np.array(test_data[0]["%Vce corrigé"][:begin_point-1]),mean_vce])
plt.plot(np.array(test_data[0]["N° de cycle"]),np.array(test_data[0]["%Vce corrigé"]),color="blue")
plt.plot(All_cycles_in_pred,all_vces_in_pred,color='orange')
lab = str(100*confidence) + " confidence interval"
plt.fill_between(
    All_cycles_in_pred,
    lower_bound,
    upper_bound,
    color="tab:orange",
    alpha=0.5,
    # label=lab,
)
plt.legend(["True $Vce$ evolution","Predicted $Vce$ evolution",lab],loc="upper left")
plt.xlabel("Cycles")
plt.ylabel("$\Delta$ Vce / Vce %" )
plt.title("$\Delta T : 70°C , T_{ref} : 55°C , t_{on} : 3 , t_{off} : 6$ ")
plt.grid()

plt.show()

# ############################## Metrics
# N_tot_interp = np.interp(np.arange(np.array(test_data[0]["N° de cycle"])[0],np.array(test_data[0]["N° de cycle"])[-1]+1),np.array(test_data[0]["N° de cycle"]),np.array(test_data[0]["N° de cycle"]))# interpolation des cycles 
# Vce_tot_interp = np.interp(np.arange(np.array(test_data[0]["N° de cycle"])[0],np.array(test_data[0]["N° de cycle"])[-1]+1),np.array(test_data[0]["N° de cycle"]),np.array(test_data[0]["%Vce corrigé"]))# interpolation des valeurs

# mean_values_interp = np.interp(np.arange(resN[0],resN[-1]+1),resN,mean_values) # interpolation des prediction
# Lower_bound_interp = np.interp(np.arange(resN[0],resN[-1]+1),resN,Lower_bound) # interpolation des valeurs min
# Upper_bound_interp = np.interp(np.arange(resN[0],resN[-1]+1),resN,Upper_bound) # interpolation des valeurs max
# resN_interp = np.interp(np.arange(resN[0],resN[-1]+1),resN,resN) # interpolation des cycles de predictions 

# Stop_pred_N_1 = NcTot_arr[ind1] #Où arreter les predictions pour la phase 1
# Start_N = Nc[-1] # doubtful

# Pred_Start_ind_b = np.where(resN_interp==(Nc[-1]+1))[0][0] #Indice debut de prediction
# Pred_End_ind_b = np.where(resN_interp==End_N)[0][0]#Indice fin de prediction

# Pred_Start_ind_GT = np.where(N_tot_interp==(Nc[-1]+1))[0][0] # Indice debut ground truth 
# Pred_End_ind_GT = np.where(N_tot_interp==End_N)[0][0] # Indice fin ground truth 

# ################################################### Calcul RBT ##########################################################
# Low_b_portion =Lower_bound_interp[Pred_Start_ind_b:Pred_End_ind_b+1]
# Up_b_portion =Upper_bound_interp[Pred_Start_ind_b:Pred_End_ind_b+1]
# GT_portion = Vce_tot_interp[Pred_Start_ind_GT:Pred_End_ind_GT+1]
# Bool_array = (GT_portion<Up_b_portion)*(GT_portion>Low_b_portion) # On regarde sur chaque cycle si la prediction est à l'interieur de l'intervalle
# RBT = np.mean(Bool_array)  # RBT : le pourcentage de temps ou la courbe test est a l'interieur de l'intervalle, c'est la moyenne

# ################################################## Calcul RMSBT ##########################################################
# if Bool_array[-1] == False : #Si la derniere iteration est en dehors de l'intervalle, RMSBT = 0 par definition
#     RMSBT = 0
# else :
#     if np.where(Bool_array==False)[0].size == 0 : # Si aucune valeur n'est en dehors, RMSBT = 1 par definition
#         RMSBT = 1
#     else :
#         last_occ = np.where(Bool_array==False)[0][-1] # On cherche le dernier instant ou la courbe est en dehors 
#         RMSBT = 1-(last_occ+1)/len(Bool_array)  # RMSBT : le premier instant (normalisé par le nombre de cycle de prediction) ou la courbe est TOUJOURs a l'interieur, c'est l'instant qui suit l'instant défini juste en dessous  

# ################################################# Affichage des courbes ###################################################
# plot_end_ind_bound = np.where(resN_interp==End_N)[0][0] 

# plt.plot(resN_interp[:plot_end_ind_bound+1],mean_values_interp[:plot_end_ind_bound+1],color='orange')
# plt.fill_between(resN_interp[:plot_end_ind_bound+1], Lower_bound_interp[:plot_end_ind_bound+1], Upper_bound_interp[:plot_end_ind_bound+1], color='orange', alpha=.5)
# plt.plot(N_tot_interp,Vce_tot_interp,color='blue')


# plt.xlabel("Cycles")
# plt.ylabel("$V_{ce}$")

# plt.title("Phase " + str(phase) + " predictions with a confidence interval of " +str(1-IC_alpha) +" on " + str(Prediction_steps*step_size) + " cycles" )

# strRBT = f'{RBT:.3f}'
# strRMSBT = f'{RMSBT:.3f}'
# # plt.text(1000, 1.56, 'RBT = ' + str(RBT) + " RMSBT = " + str(RMSBT), fontsize = 22)
# plt.text(500000, 1.5, 'RBT = ' + strRBT+ "\nRMSBT = " + strRMSBT, fontsize = 8)


## Load test data 

## Normalize data 

## Infer Lc from Vce 

## Infer Mechanical quanities from loading profile 

## Use KDE to infer New LC 