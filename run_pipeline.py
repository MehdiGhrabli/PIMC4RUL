import click
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
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from tqdm import tqdm
from KDE_helper import conditional_sample
import random
random.seed(3)

import warnings
warnings.filterwarnings("ignore")

@click.command() 

@click.option("-sdp","--simulated-data-path",default = "data\Simulated_data.csv", type =str)
@click.option("-icl","--initial-contact-length",default=0.3615,type=float)
@click.option("-e","--eps",default=0.3615*1e-8,type=float)
@click.option("-tm","--t-min",default=55,type=int)
@click.option("-sk","--svr-kernel",default="rbf",type=str)
@click.option("-sc","--svr-c",default=100,type=float)
@click.option("-sg","--svr-gamma",default=0.1,type=float)
@click.option("-dtp","--destructive-data-path",default="data\Cross section analysis.xlsx",type=str)
@click.option("-trdp","--train-data-path",default="train_data_files_standard_70.txt" ,type=str)
@click.option("-tedp","--test-data-path",default= "data\Runs_to_failure_70\Rth Module 33L.xlsx" ,type=str)
@click.option("-hs","--history-size",default=2 ,type=int)
@click.option("-ps","--prediction-size",default=1 ,type=int)
@click.option("-ds","--data-size",default=40 ,type=int)
@click.option("-pogd","--path-of-generated-data",default="Generated_data" ,type=str)
@click.option("-ufp","--used-features-path",default="Features_file.txt" ,type=str)
@click.option("-bwp","--bwp",default=0.01 ,type=float)
@click.option("-gs","--gp-std",default=0.2 ,type=float)
@click.option("-kk","--kde-kernel",default="gaussian" ,type=str)
@click.option("-sci","--start-cycle-index",default=10 ,type=int)
@click.option("-c","--confidence",default=0.90 ,type=float)
@click.option("-nomi","--number-of-montecarlo-iterations",default=100 ,type=int)
@click.option("-rdp","--regular-data-path",default="Generated_data" ,type=str)

def full_pipeline(
    simulated_data_path,
    initial_contact_length,
    eps,
    destructive_data_path,
    gp_std,
    svr_kernel,
    svr_c,
    svr_gamma,
    train_data_path,
    test_data_path,
    history_size,
    prediction_size,
    data_size,
    path_of_generated_data,
    used_features_path,
    bwp,
    kde_kernel,
    start_cycle_index,
    confidence,
    regular_data_path,
    number_of_montecarlo_iterations,
    t_min
):
    
    GP_Vce_to_Lc = get_GP(destructive_data_path,noise_std=gp_std) # Fitting the gaussian process on data from destructive tests
    Vce_anchors = np.linspace(0,12,1000).reshape(-1,1) # Anchor values of Vce to infer Lc from
    Lc_continuous = GP_Vce_to_Lc.predict(Vce_anchors,return_std=False) # Predicting Lc values from Vce


    Df =pd.read_csv(simulated_data_path) # Reading simulated data
    # Generating features using log and exp
    Df.insert(loc=1, column='Log_Contact_Radius_Length', value= np.log(Df["Contact_Radius_Length"]))
    Df.insert(loc=2, column='Exp_Contact_Radius_Length', value= np.exp(Df["Contact_Radius_Length"]))
    Df.insert(loc=4, column='Log_Contact_Radius_Width', value= np.log(Df["Contact_Radius_Width"]))
    Df.insert(loc=5, column='Exp_Contact_Radius_Width', value= np.exp(Df["Contact_Radius_Width"]))
    Df.insert(loc=7, column='Log_Average_Chip_1', value= np.log(Df["Average_Chip_1"]))
    Df.insert(loc=8, column='Exp_Average_Chip_1', value= np.exp(Df["Average_Chip_1"]))

    # Noramlizing data
    Mu = dict()
    Sig = dict()
    for Col in Df.columns:
        Mu[Col]=Df[Col].mean()
        Sig[Col]=Df[Col].std()
        Df[Col] = (Df[Col] - Df[Col].mean()) / Df[Col].std()

    # Applying support vector regression
    X=np.array(Df[["Contact_Radius_Length","Log_Contact_Radius_Length","Exp_Contact_Radius_Length","Average_Chip_1","Log_Average_Chip_1","Exp_Average_Chip_1"]])

    Y=np.array(Df["Stress_Contact_1"]);svr_stress_contact_1 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_stress_contact_1.fit(X, Y);
    Y=np.array(Df["Stress_Contact_2"]);svr_stress_contact_2 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_stress_contact_2.fit(X, Y)
    Y=np.array(Df["Stress_Metal_1"]);svr_stress_Metal_1 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_stress_Metal_1.fit(X, Y);
    Y=np.array(Df["Stress_Metal_2"]);svr_stress_Metal_2 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_stress_Metal_2.fit(X, Y)
    Y=np.array(Df["Total_Strain_Contact_1"]);svr_strain_contact_1 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_strain_contact_1.fit(X, Y)
    Y=np.array(Df["Total_Strain_Contact_2"]);svr_strain_contact_2 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_strain_contact_2.fit(X, Y)
    Y=np.array(Df["Total_Strain_Metal_1"]);svr_strain_Metal_1 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_strain_Metal_1.fit(X, Y)
    Y=np.array(Df["Total_Strain_Metal_2"]);svr_strain_Metal_2 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_strain_Metal_2.fit(X, Y)
    Y=np.array(Df["Total_Displacement_Contact_1"]);svr_displacement_contact_1 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_displacement_contact_1.fit(X, Y)
    Y=np.array(Df["Total_Displacement_Contact_2"]);svr_displacement_contact_2 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_displacement_contact_2.fit(X, Y)
    Y=np.array(Df["Total_Displacement_Metal_1"]);svr_displacement_Metal_1 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_displacement_Metal_1.fit(X, Y)
    Y=np.array(Df["Total_Displacement_Metal_2"]);svr_displacement_Metal_2 = SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma);svr_displacement_Metal_2.fit(X, Y)

    # Reading training runs to failure data
    L_data = []
    Train_file = open(train_data_path, "r")
    Train_paths = Train_file.readlines()

    for Run_path in Train_paths : 
        Current_run = pd.read_excel(Run_path.replace("\n",""))
        Run_delta_temp = Current_run["DeltaT"].iloc[0]
        L_data.append((Current_run,Run_delta_temp))

    # Computing the average step size
    Per = 0
    for i in range(len(L_data)):
        Per+=get_period(L_data[i][0])
    Per/=(len(L_data))
    Per = int(np.floor(Per))
    print(Per)

    ############################################################ Generate features #################################################
    # Create synthetic data with a uniform step size 
    Synthetic_uniform_data = []
    Base_data_size = int(0.05*Per*len(L_data))  #Select the number of the synthetic runs to failure. This can largely be bigger than the size of the experimental runs to failure as multiple scenarios can be obtained, depending on the starting point

    for i in tqdm(range(Base_data_size)):
        Index_og_data = random.randint(0,len(L_data)-1)
        Synthetic_uniform_data.append((get_sampled_varied(L_data[Index_og_data][0],Per),L_data[Index_og_data][1]))


    Gen_size = 2*Base_data_size # Select the size of the generated synthetic data, this is different from Base_data_size as the gaussian process regression can produce multiple results for the same synthetic run
    Unique_data_size = len(Synthetic_uniform_data)
    # GP_Vce_to_Lc = get_GP("Cross section analysis.xlsx",0.2) #Gaussian process regression function

    
    if os.path.exists(regular_data_path) and os.path.isdir(regular_data_path):
        shutil.rmtree(regular_data_path)
    os.mkdir(regular_data_path) 

    #Loop to generate runs to failure supplemented by the estimations of the corresponding mechanical properties
    for i in tqdm(range(Gen_size)):
        Picked_index=random.randint(0, Unique_data_size-1)#Choose a random element from the synthetic data list, which will be the synthetic run for this instance of the loop
        Current_data = Synthetic_uniform_data[Picked_index][0]
        Current_vce = Current_data[1] #Vce values of the current run
    
        Temp = (Synthetic_uniform_data[Picked_index][1]- Mu["Average_Chip_1"])/Sig["Average_Chip_1"]#Normalized temperature of the run 
        Log_temp = (np.log(Synthetic_uniform_data[Picked_index][1]) -Mu["Log_Average_Chip_1"])/Sig["Log_Average_Chip_1"]#Normalized value of the logarithm of the temperature of the run
        Exp_temp = (np.exp(Synthetic_uniform_data[Picked_index][1])-Mu["Exp_Average_Chip_1"])/Sig["Exp_Average_Chip_1"]#Normalized value of the exponential of temperature of the run
        Names = ["Cycle","Contact_Radius_Length","Stress_Contact_1","Stress_Contact_2","Stress_Metal_1","Total_Strain_Contact_1","Total_Strain_Contact_2","Total_Strain_Metal_1","Total_Displacement_Contact_1","Total_Displacement_Contact_2","Total_Displacement_Metal_1","Tmin","DeltaT"] #Names of the used features
        Denormalized_df = pd.DataFrame(columns=Names)# Dataframe containing the results (contact length, temperature, features, mechanical features)
        for j in range(len(Current_vce)):
            mu, sig = GP_Vce_to_Lc.predict(np.array(Current_vce[j]*2).reshape(-1,1),return_std=True) # Mean and variance corresponding to the predicted lc value
            Sample = min(max(((1-(np.random.normal(loc=mu,scale=sig)/2 + 0.5))*initial_contact_length)[0],eps),initial_contact_length) # Sampling from the mean and the variance and transforming the crack length to contact length
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
            DC1= svr_displacement_contact_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            DC2= svr_displacement_contact_2.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            DM1=svr_displacement_Metal_1.predict(np.array([Norm_len,Log_length,Exp_length,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            # Denormalizing mechanical properties
            Reg_SC1 = SC1*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"]
            Reg_SC2 = SC2*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"]
            Reg_SM1 = SM1*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"]
            Reg_SnC1 = SnC1*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"]
            Reg_SnC2 = SnC2*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"]
            Reg_SnM1 = SnM1*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"]
            Reg_DC1 = DC1*Sig["Total_Displacement_Contact_1"] +Mu["Total_Displacement_Contact_1"]
            Reg_DC2 = DC2*Sig["Total_Displacement_Contact_2"] +Mu["Total_Displacement_Contact_2"]
            Reg_DM1 = DM1*Sig["Total_Displacement_Metal_1"] +Mu["Total_Displacement_Metal_1"]
            # Dataframe containing denormalized features and mechanical properties
            Denormalized_df = pd.concat([ Denormalized_df, pd.DataFrame([[Current_data[0][j],Sample,Reg_SC1,Reg_SC2,Reg_SM1,Reg_SnC1,Reg_SnC2,Reg_SnM1,Reg_DC1,Reg_DC2,Reg_DM1,Temp,t_min]], columns=Denormalized_df.columns) ], ignore_index=True)
        # Writing the resulting data frame in a csv file 
        Denormalized_df.to_csv(regular_data_path+'/Unnormalized_' + str(i)+ '.csv', index=False)



    ################################################################################################################################

    # Reading test data
    Test_data = pd.read_excel(test_data_path)

    # Selecting the mechanical features to use
    Features = []
    Features_file = open(used_features_path, "r")
    Feature_names = Features_file.readlines()
    for Feature_name in Feature_names : 
        Features.append(Feature_name.replace("\n",""))
    Feature_length = len(Features)-1

    # Reading training data consisting of crack lengths and corresponding mechanical properties
    Generated_data_files = [f for f in listdir(path_of_generated_data)[:data_size] if isfile(join(path_of_generated_data, f))]
    Container = []
    for i in range(Feature_length*history_size+prediction_size):
        Container.append([])
    for Small_path in Generated_data_files:
        Full_path_data = path_of_generated_data + "\\" + Small_path
        Current_df = pd.read_csv(Full_path_data)
        Current_df = Current_df[Features]
        for i in range(history_size,len(Current_df)-prediction_size+1):
            Delta_lc = Current_df["Contact_Radius_Length"].iloc[i]-Current_df["Contact_Radius_Length"].iloc[i-1]
            Container[0].append(Delta_lc)
            for k in range(history_size):
                for l in range(Feature_length): 
                    Container[(l+1)+k*Feature_length].append(Current_df[Features[l+1]].iloc[i-(k+1)])

    Train_data = np.zeros((len(Container[0]),len(Container)))

    #Normalizing the training data
    Max_len_delta = np.max(Container[0]);Min_len_delta = np.min(Container[0])
    Max_SC1 = max(np.max(Container[1]),np.max(Container[4]));Min_SC1 = min(np.min(Container[1]),np.min(Container[4]))
    Max_SnC1 = max(np.max(Container[2]),np.max(Container[5]));Min_SnC1 = min(np.min(Container[2]),np.min(Container[5]))
    Max_DC1 = max(np.max(Container[3]),np.max(Container[6]));Min_DC1 = min(np.min(Container[3]),np.min(Container[6]))

    Train_data[:,0] = (np.array(Container[0])-Min_len_delta)/(Max_len_delta-Min_len_delta)
    Train_data[:,1] = (np.array(Container[1])-Min_SC1)/(Max_SC1-Min_SC1)
    Train_data[:,2] = (np.array(Container[2])-Min_SnC1)/(Max_SnC1-Min_SnC1)
    Train_data[:,3] = (np.array(Container[3])-Min_DC1)/(Max_DC1-Min_DC1)
    Train_data[:,4] = (np.array(Container[4])-Min_SC1)/(Max_SC1-Min_SC1)
    Train_data[:,5] = (np.array(Container[5])-Min_SnC1)/(Max_SnC1-Min_SnC1)
    Train_data[:,6] = (np.array(Container[6])-Min_DC1)/(Max_DC1-Min_DC1)

    Kde = KernelDensity(kernel=kde_kernel, bandwidth=bwp).fit(Train_data) # Applying kernel density estimation

    Temperature_variation = Test_data["DeltaT"].iloc[0] # Temperature variation of the test run

    # Normalized temperature features
    Normalized_temperature_variation = (Temperature_variation- Mu["Average_Chip_1"])/Sig["Average_Chip_1"] 
    Normalized_Log_temperature = (np.log(Temperature_variation) -Mu["Log_Average_Chip_1"])/Sig["Log_Average_Chip_1"]
    Normalized_Exp_temperature = (np.exp(Temperature_variation)-Mu["Exp_Average_Chip_1"])/Sig["Exp_Average_Chip_1"]

    # Data treatment of the test run to obtain a uniform step size
    N_cycle,Vce_cycle=get_sampled(Test_data.iloc[:start_cycle_index],Per)

    # Sampling LC values corresponding to observed Vce values
    Mu1, Sig1 = GP_Vce_to_Lc.predict(np.array(Vce_cycle[-1]*2).reshape(-1,1),return_std=True) ; Mu2, Sig2 = GP_Vce_to_Lc.predict(np.array(Vce_cycle[-2]*2).reshape(-1,1),return_std=True)
    Sample1 = min(max(((1-(np.random.normal(loc=Mu1,scale=Sig1)/2 + 0.5))*initial_contact_length)[0],eps),initial_contact_length); Sample2 = min(max(((1-(np.random.normal(loc=Mu2,scale=Sig2)/2 + 0.5))*initial_contact_length)[0],eps),initial_contact_length); 
    
    print("Sample 1 is ", Sample1)
    print("Sample 2 is ", Sample2)

    # Creating normalized crack length features
    Norm_len1 =(Sample1-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"];Norm_len2 =(Sample2-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"]
    Log_length1 = (np.log(Sample1)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"];Log_length2 = (np.log(Sample2)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"]
    Exp_length1 = (np.exp(Sample1)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"];Exp_length2 = (np.exp(Sample2)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"]

    # Predicting mechanical properties from crack lengths
    SC1_1= svr_stress_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0];SC1_2= svr_stress_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0]
    SnC1_1= svr_strain_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0];SnC1_2= svr_strain_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0]
    DC1_1= svr_displacement_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0];DC1_2= svr_displacement_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0]
    # Denormalizing predicted mechanical properties
    Reg_SC1_1_out = SC1_1*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];Reg_SC1_2_out = SC1_2*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];
    Reg_SnC1_1_out = SC1_1*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];Reg_SnC1_2_out = SnC1_2*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];
    Reg_DC1_1_out = DC1_1*Sig["Total_Displacement_Contact_1"] +Mu["Total_Displacement_Contact_1"];Reg_DC1_2_out = DC1_2*Sig["Total_Displacement_Contact_1"] +Mu["Total_Displacement_Contact_1"];


    Needed_props = [(Reg_SC1_1_out-Min_SC1)/(Max_SC1-Min_SC1) ,(Reg_SnC1_1_out-Min_SnC1)/(Max_SnC1-Min_SnC1) ,(Reg_DC1_1_out-Min_DC1)/(Max_DC1-Min_DC1),(Reg_SC1_2_out-Min_SC1)/(Max_SC1-Min_SC1)  ,(Reg_SnC1_2_out-Min_SnC1)/(Max_SnC1-Min_SnC1) ,(Reg_DC1_2_out-Min_DC1)/(Max_DC1-Min_DC1) ]
    print("Mechanical values : ")
    print(Needed_props)
    # Transorming the training data into a pandas dataframe
    Train_data_df = pd.DataFrame(Train_data)
    Train_data_df.columns=["Contact_Radius_Length","Stress_Contact_1_close","Total_Strain_Contact_1_close","Total_Displacement_Contact_1_close","Stress_Contact_1_far","Total_Strain_Contact_1_far","Total_Displacement_Contact_1_far"]
    Cols = ["Stress_Contact_1_close","Total_Strain_Contact_1_close","Total_Displacement_Contact_1_close","Stress_Contact_1_far","Total_Strain_Contact_1_far","Total_Displacement_Contact_1_far"]
    
    # To be commented
    New_lengths_list = []
    Old_length = Sample2
    New_length = Sample1
    print("Old length is : ", Old_length)
    print("new length is : ", New_length )

    Number_of_iterations = int(np.ceil((np.array((Test_data["Number of cycles"]))[-1]-start_cycle_index*Per)/Per)) # Number of predictions steps in the loop

    Scenarios_lc = np.zeros((number_of_montecarlo_iterations,Number_of_iterations)) # Numpy array containing the predicted Lc values per montecarlo run
    Scenarios_Vce = np.zeros((number_of_montecarlo_iterations,Number_of_iterations)) # Numpy array containing the predicted Vce values per montecarlo run

    # Running the Monte Carlo Loop
    for m in tqdm(range(number_of_montecarlo_iterations)):

        New_lengths_list = [] # List containing predicted the crack lengths
        Old_length = Sample2
        New_length = Sample1
        Needed_props = [(Reg_SC1_1_out-Min_SC1)/(Max_SC1-Min_SC1)  ,(Reg_SnC1_1_out-Min_SnC1)/(Max_SnC1-Min_SnC1) ,(Reg_DC1_1_out-Min_DC1)/(Max_DC1-Min_DC1),(Reg_SC1_2_out-Min_SC1)/(Max_SC1-Min_SC1)  ,(Reg_SnC1_2_out-Min_SnC1)/(Max_SnC1-Min_SnC1) ,(Reg_DC1_2_out-Min_DC1)/(Max_DC1-Min_DC1) ] # Normalized mechanical properties used for regression
        
        # Running the prediction loop for this Monte Carlo instance
        for i in tqdm(range(Number_of_iterations)):   

            # Sampling the candidate crack length variation conditionnally on the observed mechanical properties using kernel density estimation
            Candidate_delta_lc = conditional_sample(Kde, Train_data_df, Cols, Needed_props, samples = 100) 
            Candidate_delta_lc=np.mean(Candidate_delta_lc[:,0]) 
            Candidate_delta_lc = Candidate_delta_lc*(Max_len_delta-Min_len_delta)+Min_len_delta # Denormalizing sampled value 
            
            # Updating the new crack length
            Old_length = New_length
            New_length = max(Old_length + Candidate_delta_lc,eps)
            New_lengths_list.append(New_length)

            # Calculating and normalizing length features
            Norm_len1 =(New_length-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"];Norm_len2 =(Old_length-Mu["Contact_Radius_Length"])/Sig["Contact_Radius_Length"]
            Log_length1 = (np.log(New_length)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"];Log_length2 = (np.log(Old_length)-Mu["Log_Contact_Radius_Length"])/Sig["Log_Contact_Radius_Length"]
            Exp_length1 = (np.exp(New_length)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"];Exp_length2 = (np.exp(Old_length)-Mu["Exp_Contact_Radius_Length"])/Sig["Exp_Contact_Radius_Length"]

            # Calculating mechanical properties 
            SC1_1= svr_stress_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0];SC1_2= svr_stress_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0]
            # SC2_1=svr_stress_contact_2.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SC2_2=svr_stress_contact_2.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            # SM1_1=svr_stress_Metal_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SM1_2=svr_stress_Metal_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            SnC1_1= svr_strain_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0];SnC1_2= svr_strain_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0]
            # SnC2_1= svr_strain_contact_2.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SnC2_2= svr_strain_contact_2.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            # SnM1_1= svr_strain_Metal_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];SnM1_2= svr_strain_Metal_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0]
            DC1_1= svr_displacement_contact_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0];DC1_2= svr_displacement_contact_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Normalized_temperature_variation,Normalized_Log_temperature,Normalized_Exp_temperature]).reshape(1,-1))[0]
            # DC2_1= svr_deformation_contact_2.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];DC2_2= svr_deformation_contact_2.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];
            # DM1_1=svr_deformation_Metal_1.predict(np.array([Norm_len1,Log_length1,Exp_length1,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];DM1_2=svr_deformation_Metal_1.predict(np.array([Norm_len2,Log_length2,Exp_length2,Temp,Log_temp,Exp_temp]).reshape(1,-1))[0];

            Reg_SC1_1 = SC1_1*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];Reg_SC1_2 = SC1_2*Sig["Stress_Contact_1"] +Mu["Stress_Contact_1"];
            # Reg_SC2_1 = SC2_1*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"];Reg_SC2_2 = SC2_2*Sig["Stress_Contact_2"] +Mu["Stress_Contact_2"];
            # Reg_SM1_1 = SM1_1*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"];Reg_SM1_2 = SM1_2*Sig["Stress_Metal_1"] +Mu["Stress_Metal_1"];
            Reg_SnC1_1 = SnC1_1*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];Reg_SnC1_2 = SnC1_2*Sig["Total_Strain_Contact_1"] +Mu["Total_Strain_Contact_1"];
            # Reg_SnC2_1 = SnC2_1*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"];Reg_SnC2_2 = SnC2_2*Sig["Total_Strain_Contact_2"] +Mu["Total_Strain_Contact_2"];
            # Reg_SnM1_1 = SnM1_1*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"];Reg_SnM1_2 = SnM1_2*Sig["Total_Strain_Metal_1"] +Mu["Total_Strain_Metal_1"];
            Reg_DC1_1 = DC1_1*Sig["Total_Displacement_Contact_1"] +Mu["Total_Displacement_Contact_1"];Reg_DC1_2 = DC1_2*Sig["Total_Displacement_Contact_1"] +Mu["Total_Displacement_Contact_1"];
            # Reg_DC2_1 = DC2_1*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"];Reg_DC2_2 = DC2_2*Sig["Total_Deformation_Contact_2"] +Mu["Total_Deformation_Contact_2"];
            # Reg_DM1_1 = DM1_1*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"];Reg_DM1_2 = DM1_2*Sig["Total_Deformation_Metal_1"] +Mu["Total_Deformation_Metal_1"];
            
            Needed_props = [(Reg_SC1_1-Min_SC1)/(Max_SC1-Min_SC1)  ,(Reg_SnC1_1-Min_SnC1)/(Max_SnC1-Min_SnC1) ,(Reg_DC1_1-Min_DC1)/(Max_DC1-Min_DC1),(Reg_SC1_2-Min_SC1)/(Max_SC1-Min_SC1)  ,(Reg_SnC1_2-Min_SnC1)/(Max_SnC1-Min_SnC1) ,(Reg_DC1_2-Min_DC1)/(Max_DC1-Min_DC1) ]
        
        New_lengths_arr = np.array(New_lengths_list) # Full run to failure values of the contact length 
        print("New lengths array : ")
        print(New_lengths_arr)

        Scenarios_lc[m,:]=New_lengths_arr
        New_cracks = initial_contact_length-New_lengths_arr # Estimated crack lengths
        print("New cracks array : ")
        print(New_cracks)
        New_lengths_arr_normed = 2*(New_cracks/initial_contact_length -0.5) #Normalized crack lengths
        New_Vces_normed = np.interp(New_lengths_arr_normed.flatten(),Lc_continuous.flatten(),Vce_anchors.flatten()) #Normalized predicted Vce values
        New_Vces = New_Vces_normed/2 # Denormalized Vce values 
        print("New Vces : ")
        print(New_Vces)
        Scenarios_Vce[m,:]=New_Vces # Predicted Vce values

    added_cycles = np.arange(1,Number_of_iterations+1)*Per+Test_data["Number of cycles"][start_cycle_index-1] # Loading cycles for the entire run
    All_cycles_in_pred = np.concatenate([Test_data["Number of cycles"][:start_cycle_index],added_cycles]) # Loading cycles for prediction
    
    Mean_vce = np.mean(Scenarios_Vce,axis=0) # Mean of the Monte Carlo runs
    Std_vce = np.std(Scenarios_Vce,axis=0) # standard deviation of the Monte Carlo runs
    
    # Calculating the confidence interval
    H = Std_vce * scipy.stats.t.ppf((1 + confidence) / 2., number_of_montecarlo_iterations-1)
    Lower_bound = Mean_vce-H
    Lower_bound =np.concatenate([np.array(Test_data["delta Vce / Vce %"][:start_cycle_index]),Lower_bound])
    Upper_bound = Mean_vce+H
    Upper_bound =np.concatenate([np.array(Test_data["delta Vce / Vce %"][:start_cycle_index]),Upper_bound])

    Lower_bound_calc = Mean_vce-H
    Upper_bound_calc = Mean_vce+H

    All_vces_in_pred= np.concatenate([np.array(Test_data["delta Vce / Vce %"][:start_cycle_index]),Mean_vce]) # Concatenating observed and predicted Vces
    
    ######################################### Plotting results #############################################################
    plt.plot(np.array(Test_data["Number of cycles"]),np.array(Test_data["delta Vce / Vce %"]),color="blue")
    plt.plot(All_cycles_in_pred,All_vces_in_pred,color='orange')
    Lab = str(int(100*confidence)) + "% confidence interval"
    plt.fill_between(
        All_cycles_in_pred,
        Lower_bound,
        Upper_bound,
        color="tab:orange",
        alpha=0.5,
        label=Lab,
    )
    plt.legend(["True $Vce$ evolution","Predicted $Vce$ evolution",Lab],loc="upper left")
    plt.xlabel("Cycles")
    plt.ylabel(r'Relative $V_{ce}$ variation (%)' )
    plt.title("$\Delta T : $" + str(int(Temperature_variation)) + "°C $T_{min} : 55°C , t_{on} : 3 , t_{off} : 6$ ")
    plt.grid()

    plt.show() 
    # Evaluating the predictions using the metrics RBT and RMSBT
    N_tot_interp = np.interp(np.arange(np.array(Test_data["Number of cycles"])[0],np.array(Test_data["Number of cycles"])[-1]+1),np.array(Test_data["Number of cycles"]),np.array(Test_data["Number of cycles"]))# Interpolating cycles
    Vce_tot_interp = np.interp(np.arange(np.array(Test_data["Number of cycles"])[0],np.array(Test_data["Number of cycles"])[-1]+1),np.array(Test_data["Number of cycles"]),np.array(Test_data["delta Vce / Vce %"]))# Interpolating Vce values
    Pure_cycles = np.arange(1,Number_of_iterations+1)*Per+Test_data["Number of cycles"][start_cycle_index-1]
    Mean_values_interp = np.interp(np.arange(Pure_cycles[0],Pure_cycles[-1]+1),Pure_cycles,Mean_vce) # Interpolating predictions
    Lower_bound_interp = np.interp(np.arange(Pure_cycles[0],Pure_cycles[-1]+1),Pure_cycles,Lower_bound_calc) # Interpolating lower bound
    Upper_bound_interp = np.interp(np.arange(Pure_cycles[0],Pure_cycles[-1]+1),Pure_cycles,Upper_bound_calc) # Interpolating upper bound
    ResN_interp = np.interp(np.arange(Pure_cycles[0],Pure_cycles[-1]+1),Pure_cycles,Pure_cycles) # Interpolating prediction cycles

    End_N = np.array(Test_data["Number of cycles"])[-1]

    Pred_Start_ind_b = np.where(ResN_interp==(Pure_cycles[0]))[0][0] # Start prediction index
    Pred_End_ind_b = np.where(ResN_interp==End_N)[0][0] # End of predictions  index

    Pred_Start_ind_GT = np.where(N_tot_interp==(Pure_cycles[0]))[0][0] # Start index of test data
    Pred_End_ind_GT = np.where(N_tot_interp==End_N)[0][0] # End index of test data

    ################################################### Calculating RBT ##########################################################
    Low_b_portion =Lower_bound_interp[Pred_Start_ind_b:Pred_End_ind_b+1]
    Up_b_portion =Upper_bound_interp[Pred_Start_ind_b:Pred_End_ind_b+1]
    GT_portion = Vce_tot_interp[Pred_Start_ind_GT:Pred_End_ind_GT+1]
    Bool_array = (GT_portion<Up_b_portion)*(GT_portion>Low_b_portion) 
    RBT = np.mean(Bool_array) 

    ################################################## Calculating RMSBT ##########################################################
    if Bool_array[-1] == False : 
        RMSBT = 0
    else :
        if np.where(Bool_array==False)[0].size == 0 : 
            RMSBT = 1
        else :
            Last_occ = np.where(Bool_array==False)[0][-1] 
            RMSBT = 1-(Last_occ+1)/len(Bool_array)  

    ################################################# Plotting results ###################################################

    print("RMSBT is ", RMSBT, " RBT is ", RBT)

    plt.plot(np.array(Test_data["Number of cycles"]),np.array(Test_data["delta Vce / Vce %"]),color="blue")
    plt.plot(ResN_interp[Pred_Start_ind_b:Pred_End_ind_b+1],Mean_values_interp[Pred_Start_ind_b:Pred_End_ind_b+1],color='orange')
    Lab = str(int(100*confidence)) + "% confidence interval"
    plt.fill_between(
        ResN_interp[Pred_Start_ind_b:Pred_End_ind_b+1],
        Low_b_portion,
        Up_b_portion,
        color="tab:orange",
        alpha=0.5,
        label=Lab,
    )
    plt.legend(["True $Vce$ evolution","Predicted $Vce$ evolution",Lab],loc="upper left")
    plt.xlabel("Cycles")
    plt.ylabel(r'Relative $V_{ce}$ variation (%)' )
    plt.title("$\Delta T : $" + str(int(Temperature_variation)) + "°C $T_{min} : 55°C , t_{on} : 3 , t_{off} : 6$ ")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    full_pipeline()
