import numpy as np
# This python file contains a helper function to conditionally sample from a probability distribution. 
# This function is needed because the kernel density estimation provides a joint probability distribution, as opposed to a conditional distribution. This code was taken from https://stackoverflow.com/questions/60223573/conditional-sampling-from-multivariate-kernel-density-estimate-in-python
def conditional_sample(kde, training_data, columns, values, samples = 100):
    # Input 
    #   kde : The kernel density estimation function, obtained from the training data
    #   training_data : The data used to train the kde
    #   columns : The names of the known variables (columns)
    #   values : Values of the known variables
    #   samples : The number of samples taken from the conditional distribution
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
