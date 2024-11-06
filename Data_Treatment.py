import numpy as np
import random
# This python file contains the code to preprocess data 
# It contains three function, get_period, to get the average window between observations, get_sampled, to transform data to have a uniform step size starting from the first cycle, and get_sampled_varied, which is similar to get_sampled, but it starts at a random cycle from the first cycle to the cycle p, which is the step size


def get_period(Data_set):
  # Input :
  #   Data_set : a pandas dataframe containing the run to failure data
  # Output :
  #   The average step size
  try :
    N = np.array(Data_set["N° de cycle"])
  except:
    N = np.array(Data_set["N° de cycle\n"])
  diff = N[1:]-N[:-1]
  return np.mean(diff)

def get_sampled(Data_set,step_size):
  # Input :
  #   Data_set : a pandas dataframe containing the run to failure data
  #   step_size : average step size
  # Output :
  #   length_of_interp : a list containing the cycle indices (uniformly spaced by step_size)
  #   Vce_interp[length_of_interp] : Vce values, adjusted to the cycles in length_of_interp
  try :
    N = np.array(Data_set["N° de cycle"])
  except :
    N = np.array(Data_set["N° de cycle\n"])
  try :
    Vce = np.array(Data_set["%Vce corrigé"])
  except :
    Vce = np.array(Data_set["%Vce corrigé\n"])
  Vce_interp = np.interp(np.arange(0,N[-1]+1+step_size),N,Vce)
  length_of_sequence = int(N[-1]//step_size) +1
  length_of_interp = list(step_size*np.arange(length_of_sequence))
  return length_of_interp,Vce_interp[length_of_interp]


def get_sampled_varied(Data_set,step_size):
  # Input :
  #   Data_set : a pandas dataframe containing the run to failure data
  #   step_size : average step size
  # Output :
  #   length_of_interp : a list containing the cycle indices (uniformly spaced by step_size)
  #   Vce_interp[length_of_interp] : Vce values, adjusted to the cycles in length_of_interp
  try :
    N = np.array(Data_set["N° de cycle"])
  except :
    N = np.array(Data_set["N° de cycle\n"])
  try :
    Vce = np.array(Data_set["%Vce corrigé"])
  except :
    Vce = np.array(Data_set["%Vce corrigé\n"])
  begin_point = random.randint(0, step_size-1)
  Vce_interp = np.interp(np.arange(0,N[-1]+1+step_size),N,Vce)
  length_of_sequence = int(N[-1]//step_size) +1
  length_of_interp = list(step_size*np.arange(length_of_sequence)+begin_point)
  return length_of_interp,Vce_interp[length_of_interp]

