import numpy as np
import random
def get_period(Data_set):
  try :
    N = np.array(Data_set["N° de cycle"])
  except:
    N = np.array(Data_set["N° de cycle\n"])
  diff = N[1:]-N[:-1]
  return np.mean(diff)

def get_sampled(Data_set,periodicity):
  try :
    N = np.array(Data_set["N° de cycle"])
  except :
    N = np.array(Data_set["N° de cycle\n"])
  try :
    Vce = np.array(Data_set["%Vce corrigé"])
  except :
    Vce = np.array(Data_set["%Vce corrigé\n"])
  Vce_interp = np.interp(np.arange(0,N[-1]+1+periodicity),N,Vce)
  length_of_sequence = int(N[-1]//periodicity) +1
  length_of_interp = list(periodicity*np.arange(length_of_sequence))
  return length_of_interp,Vce_interp[length_of_interp]


def get_sampled_varied(Data_set,periodicity):
  try :
    N = np.array(Data_set["N° de cycle"])
  except :
    N = np.array(Data_set["N° de cycle\n"])
  try :
    Vce = np.array(Data_set["%Vce corrigé"])
  except :
    Vce = np.array(Data_set["%Vce corrigé\n"])
  begin_point = random.randint(0, periodicity-1)
  Vce_interp = np.interp(np.arange(0,N[-1]+1+periodicity),N,Vce)
  length_of_sequence = int(N[-1]//periodicity) +1
  length_of_interp = list(periodicity*np.arange(length_of_sequence)+begin_point)
  return length_of_interp,Vce_interp[length_of_interp]

