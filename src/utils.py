import numpy as np


# ------- Exercise 11 utils functions
def align(X1, X2):
   """
   Given two dataframes, with indices being the time, aligns them by interpolating missing values.


   Args:
      X1 (DataFrame): Data to align
      X2 (DataFrame): Labels to align

   Returns:
      X1 (DataFrame): Data aligned
      X2 (DataFrame): Labels aligned

   """

   data = X1.join(X2, how = 'outer', sort = True)
   #Interpolate missing values (will interpolate labels but also X1)
   data = data.interpolate('index', limit_area ='inside')

   #Get only the data interpolated (remove NaN)
   data = data.loc[data.iloc[:,0].dropna().index.intersection(data.iloc[:,-1].dropna().index)]

   #Remove original labels index (X1 was interpolated at these rows, we don't want it)
   data = data[~data.index.isin(X2.index.intersection(data.index))]

   #X back but cut with labels
   X1 = data[X1.columns]

   #Labels and columns of subject and condition
   X2 = data[X2.columns]

   #Resolve some align issues
   X2 = X2.round(decimals=0)
   X2[(X2['Right_Hand_channel1'] == 0) & (X2['Right_Hand_channel2'] == 1) &  
      (X2['Right_Hand_channel3'] == 2) & (X2['Right_Hand_channel4'] == 2) &  
      (X2['Right_Hand_channel5'] == 2) & (X2['Right_Hand_channel6'] == 2)] = [0,0,2,2,2,2]

   return X1,X2

def find_nth_repetition(Labels,n):
    """
    This function is defined to find the nth repetition of a given action
    :param action: string containing the action to be found
    :param n: integer containing the number of repetition to be found
    :return: integer containing the index of the nth repetition of the given action
    """
    
    unique_actions = np.unique(Labels)
    action_indices = {action: [] for action in unique_actions}
    last_action = None

    for i, action in enumerate(Labels):
        if action != last_action:
            action_indices[action].append(i)
            last_action = action
    
    nth_repeat_indices = {}
    for action, indices in action_indices.items():
        if len(indices) >= n:
            nth_repeat_indices[action] = indices[n-1]
        else:
            nth_repeat_indices[action] = None 
    
    return nth_repeat_indices


def cut_datasets(EMG, Labels,val_cut, test_cut):

    """
    This function is defined to cut the data in three sets
    :param EMG: pandas DataFrame containing the data
    :param Targets: pandas DataFrame containing the targets
    :param val_cut: information on how/where to cut the dataset to obtain the validation set
    :param test_cut: information on how/where to cut the dataset to obtain the test set
    :return: 6 pandas DataFrames (or numpy arrays) containing EMG and targets of each sets
    """

    EMG_train = EMG[:val_cut]
    EMG_val = EMG[val_cut:test_cut]
    EMG_test = EMG[test_cut:]
    Labels_train = Labels[:val_cut]
    Labels_val = Labels[val_cut:test_cut]
    Labels_test = Labels[test_cut:]


    return EMG_train, EMG_val, EMG_test, Labels_train, Labels_val, Labels_test

def extract_time_windows(EMG,Labels, fs,win_len,step):
# This function is used to cut the time windows from the raw EMG 
# It return an array containing the EMG of each time window.
# It also returns the labels corresponding to the time of the end of the window
    """
    This function is defined to perform an overlapping sliding window 
    :param EMG: numpy array containing the data
    :param Labels: numpy array containing the labels
    :param fs: the sampling frequency of the signal
    :param win_len: The size of the windows (in seconds)
    :param step: The step size between windows (in seconds)
    :return: A numpy arrays containing the windows
    :return: A numpy array containing the labels aligned for each window
    :note: The length of both outputs should be the same
    """
    
    n,m = EMG.shape
    win_len = int(win_len*fs)
    start_points = np.arange(0,n-win_len,int(step*fs))
    end_points = start_points + win_len

    EMG_windows = np.zeros((len(start_points),win_len,m))
    Labels_window = [] 
    for i in range(len(start_points)):
        EMG_windows[i,:,:] = EMG[start_points[i]:end_points[i],:]
        Labels_window.append(Labels[start_points[i]])
    

    
    
    return EMG_windows, np.array(Labels_window)


def calc_fft_power(EMG_windows, fs):
    N = EMG_windows.shape[1]  # Number of points in each window
    freqs = np.fft.rfftfreq(N, 1/fs)  # Frequency bins

    # Fast Fourier Transform (FFT)
    fft_vals = np.fft.rfft(EMG_windows, axis=1)
    fft_power = np.abs(fft_vals) ** 2  # Power spectrum
    return freqs[1:], fft_power[:,1:,:]


def extract_features(EMG_windows, fs):
    """
    This function extracts features from raw EMG data. 
    
    Args:
        EMG_windows (numpy array): EMG data in sliding of shape (n_windows, time, n_channels)
        fs (int): sampling frequency of EMG data
    
    Returns:
        X (numpy array): features of shape (n_windows, n_features)
    """
    # Mean absolute value (MAV), axis=1 means mean along the time axis for each window
    mav = np.mean(np.abs(EMG_windows), axis=1)

    # Maximum absolute Value (MaxAV)
    maxav = np.max(np.abs(EMG_windows), axis=1)

    # Standard Deviation (STD)
    std = np.std(EMG_windows, axis=1)

    # Root mean square (RMS)
    rms = np.sqrt(np.mean(EMG_windows ** 2, axis=1))

    # Wavelength (WL)
    wl = np.sum(np.abs(np.diff(EMG_windows, axis=1)), axis=1) 

    # Zero crossing (ZC) (hint: you can use np.diff and np.sign to evaluate the zero crossing, then sum the occurance)
    zc = np.sum(np.diff(np.sign(EMG_windows), axis=1) != 0, axis=1)

    # Slope sign changes (SSC)
    diff = np.diff(EMG_windows, axis=1)
    ssc = np.sum(np.diff(np.sign(diff), axis=1) != 0, axis=1)

    # Get frequency and spectrogram power 
    freqs, fft_power = calc_fft_power(EMG_windows, fs=fs)

    # Mean power 
    mean_power = np.mean(fft_power, axis=1)

    # Total power
    tot_power = np.sum(fft_power, axis=1)

    # Mean frequency (sum of the product of spectrogram power and frequency, divided by total sum of spectrogram power)
    freqs_reshaped = freqs.reshape(1, freqs.shape[0], 1) #reshape for multiplication of spectrogram power and frequency 
    mean_frequency = np.sum(fft_power * freqs_reshaped, axis=1) / tot_power

    # Median frequency 
    cumulative_power = np.cumsum(fft_power, axis=1)
    total_power = cumulative_power[:, -1, :]
    median_frequency = np.zeros((EMG_windows.shape[0],EMG_windows.shape[2]))

    for i in range(EMG_windows.shape[0]):
        for j in range(EMG_windows.shape[2]):
            median_frequency[i,j] = freqs[np.where(cumulative_power[i, :, j] >= total_power[i,j] / 2)[0][0]]

    # Peak frequency (use np.argmax)
    peak_frequency = freqs[np.argmax(fft_power, axis=1)]

    threshold=0.01
    ssc = np.zeros((EMG_windows.shape[0],EMG_windows.shape[2]))
    for i in range(EMG_windows.shape[0]):
        for j in range(EMG_windows.shape[2]):
            # Calculate SSC with threshold
            ssc[i, j] = np.sum((np.abs(diff[i, :-1, j]) >= threshold) &
                           (np.abs(diff[i, 1:, j]) >= threshold) &
                           (np.sign(diff[i, :-1, j]) != np.sign(diff[i, 1:, j])))


    X = np.column_stack((mav, maxav, std, rms, wl, zc, ssc, mean_power, tot_power, mean_frequency, median_frequency, peak_frequency))

    return X