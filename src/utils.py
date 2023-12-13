"""
This file contains utility functions for the project.
These functions are used in both preprocessing and classification notebooks.
"""


# Data structures
import numpy as np
import pandas as pd

# Utils
import zipfile
import os
from tqdm import tqdm
import scipy.io as sio


def unzip_and_remove(zip_folder : str) -> None:
    """
    Unzips all zip files in the specified folder and removes the original zip files.

    Args:
        zip_folder (str): Path to the folder containing the zip files.   
    """
    # Get a list of all zip files in the specified folder
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]

    # Iterate through each zip file
    for zip_file in tqdm(zip_files):
        zip_path = os.path.join(zip_folder, zip_file)
        unzip_path = os.path.join(zip_folder, zip_file.replace('.zip', ''))

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        # Remove the original zip file
        os.remove(zip_path)

def parse_exercise_data(datapath : str, subject_ids : list, exercise_id : int) -> np.ndarray:
    """
    Parse the data for the given subject ids and exercise id.

    Args:
        datapath (str): path to the data folder
        subject_ids (list): list of subject ids
        exercise_id (int): exercise id
    
    Returns:
        all_data (np.ndarray): array of shape (n_subjects, emg (10) + stimulus + repetition = 12) containing the data
    """


    # Load the data for each subject
    all_data = []
    for subject_id in tqdm(subject_ids, desc=f"Loading data for exercise {exercise_id}"):

        # Load the data a dictionary
        filepath = os.path.join(datapath, f"s{subject_id}" ,f'S{subject_id}_A1_E{exercise_id}.mat')
        data = sio.loadmat(filepath)

        # Choose the relevant columns
        emg, stimulus, repetition, subject_id_arr = data['emg'], data['restimulus'], data['rerepetition'], np.repeat(subject_id, data['emg'].shape[0]).reshape(-1, 1)
        data = np.concatenate([emg, stimulus, repetition, subject_id_arr], axis=1)

        # Append to the list
        all_data.append(data)

    # Concatenate the list 
    all_data = np.concatenate(all_data, axis=0)

    return all_data


def train_val_test_split(df : pd.DataFrame, val : list, test: list) -> pd.DataFrame:
    """
    Adds a new column to the df denoting to which split given
    timepoint belongs.

    Args:
        df (pd.DataFrame): dataframe containing the data
        val (int): ids of repetitions used for validation
        test (int): ids of repetitions used for testing

    Returns:
        df (pd.DataFrame): dataframe containing the data with the new column 
    """

    # Create a new column and assign to all timepoints default value 'train'
    df['Split'] = 'train'

    # Set the split column to 'val' for the validation set
    df.loc[df['Repetition'].isin(val), 'Split'] = 'val'

    # Set the split column to 'test' for the test set
    df.loc[df['Repetition'].isin(test), 'Split'] = 'test'

    # Set the rest to a temporary value 'rest'
    df.loc[df['Repetition'] == 0, 'Split'] = None

    # Forward fill NaN values = fill the 'rest' values with the previous value
    # E.g. rest rest train train -> train train train train
    df['Split'] = df['Split'].ffill()

    # Drop the NaN values which will only drop the first rest period
    df = df.dropna()

    return df

def extract_trial_windows(emg: np.ndarray, df : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    This function is defined to extract the trial windows from the given data. The trial windows are defined as the 
    unique stimulus and repetition combinations.

    Args:
        emg (numpy array): array containing the EMG data of shape (n_samples, n_channels)
        data (pd.DataFrame): dataframe containing the data
    
    Returns:
        emg_windows (numpy array): array containing the EMG data in windows of shape (n_windows, win_len, n_channels)
        labels_windows (numpy array): array containing the labels in windows of shape (n_windows, )
    """

    # Get the labels and repetitions
    labels = df['Stimulus'].values
    repetitions = df['Repetition'].values

    # Init the stimuli and repetitions
    curr_stim, curr_rep = None, None

    # Init the window id as a list
    window_ids = []

    curr_id = 0
    for y, rep in zip(labels, repetitions):

        # Init the current stimulus and repetition only the first time
        if curr_stim is None:
            curr_rep, curr_stim = rep, y    

        # No change, we stick to the current id and save it
        if (y == curr_stim) and (rep == curr_rep):
            window_ids.append(curr_id)
        
        # Change 
        else:
            # Update the current stimulus and repetition
            curr_stim, curr_rep = y, rep

            # Update the window id
            curr_id += 1

            # Save the window id
            window_ids.append(curr_id)

    # Add the window ids to the dataframe
    df.loc[:, "Window"] = window_ids

    # Determine the minimum length across all windows
    min_length = df.groupby('Window').size().min()

    # Determine the number of windows, timepoints, and features
    n_windows = len(df['Window'].unique())
    feat_dim = emg.shape[1]

    # Create an empty 3D array
    result_array = np.empty((n_windows, min_length, feat_dim))
    labels = np.zeros((n_windows, ), dtype=int)

    # Iterate over each window and fill the 3D array
    for i, window_value in enumerate(df['Window'].unique()):
        # Extract indices of timepoints belonging to the current window
        window_indices = df[df['Window'] == window_value].index[:min_length]
        
        # Assign the corresponding rows from the 2D array to the 3D array
        result_array[i, :, :] = emg[window_indices, :]

        # Get the corresponding stimulus
        stimulus = df[df['Window'] == window_value]['Stimulus'].values[0]
        labels[i] = stimulus
    
    return result_array, labels

def extract_time_windows(emg : np.ndarray, labels : np.ndarray, sampling_frequency : int, win_len : int, step : int) -> tuple[np.array, np.array]:

    """
    This function is defined to extract time windows from the given data using the given window lenght
    and step. The labels are assigned to the windows based on the majority of labels in the window.

    Args:
        emg (numpy array): array containing the EMG data of shape (n_samples, n_channels)
        labels (numpy array): array containing the labels of shape (n_samples, )
        sampling_frequency (int): sampling frequency of the EMG data
        win_len (float): length of the window in seconds
        step (float): step between two consecutive windows in seconds
    
    Returns:
        emg_windows (numpy array): array containing the EMG data in windows of shape (n_windows, win_len, n_channels)
        labels_windows (numpy array): array containing the labels in windows of shape (n_windows, )
    """

    # Obtain the number of samples and channels
    n,m = emg.shape

    # Define the window length based on the sampling frequency
    win_len = int(win_len*sampling_frequency)

    # Definet the start and end points of the windows
    start_points = np.arange(0, n-win_len, int(step*sampling_frequency))
    end_points = start_points + win_len

    # Init the final variables to be returned
    emg_windows = np.zeros((len(start_points), win_len, m))
    labels_windows = [] 

    # Now assign the corresponding timepont to the given windows
    for i in range(len(start_points)):
        # Extract the EMG data
        emg_windows[i,:,:] = emg[start_points[i]:end_points[i],:]

        # Extract the labels
        labels_window = labels[start_points[i]:end_points[i]]

        # Get the most frequent label
        val, count = np.unique(labels_window, return_counts=True)
        most_frequent_label = val[np.argmax(count)]

        labels_windows.append(most_frequent_label)
     
    return emg_windows, np.array(labels_windows, dtype=int)

def calc_fft_power(emg_windows : np.array, sampling_frequency : int) -> tuple[np.array, np.array]:
    """
    This function calculates the power of the FFT of the given EMG data.

    Args:
        emg_windows (numpy array): EMG data in sliding of shape (n_windows, time, n_channels)
        sampling_frequency (float): sampling frequency of the EMG data
    
    Returns:
        freqs (numpy array): frequency bins of shape (n_freqs, )
        fft_power (numpy array): power spectrum of the FFT of the EMG data of shape (n_windows, n_freqs, n_channels)
    """
    # Number of points in each window
    N = emg_windows.shape[1]

    # Frequency bins
    freqs = np.fft.rfftfreq(N, 1/sampling_frequency)  

    # Fast Fourier Transform (FFT)
    fft_vals = np.fft.rfft(emg_windows, axis=1)
    fft_power = np.abs(fft_vals) ** 2  # Power spectrum

    return freqs[1:], fft_power[:,1:,:]


def downsample_rest_windows(data : tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    This function downsamples the rest windows to the average length of the stimulus windows.

    Args:
        data (tuple): tuple containing the EMG data and labels
    
    Returns:
        X (numpy array): Adjusted EMG data of shape (n_samples, n_channels)
        y (numpy array): Adjusted labels of shape (n_samples, )
    """

    # Parse the data
    X, y = data

    # Get the unique labels and corresponding counts
    _, count = np.unique(y, return_counts=True)

    # We know that the labels are sorted, and 0 = Rest, therefore we first compute average of the stimulus
    avg_count_stim = int(np.mean(count[1:]))

    # Compute the number of samples to remove
    rest_count = count[0]
    n_samples_to_remove = rest_count - avg_count_stim

    # Get the indexes of the rest periods
    rest_index = np.where(y == 0)[0]

    # Randomly choose the indexes to remove
    remove_index = np.random.choice(rest_index, size=n_samples_to_remove, replace=False)

    # Remove the samples
    X = np.delete(X, remove_index, axis=0)
    y = np.delete(y, remove_index, axis=0)

    return X, y 


def extract_features(emg_windows : np.ndarray, sampling_frequency : int) -> np.ndarray:
    """
    This function extracts features from raw EMG data. 
    
    Args:
        emg_windows (numpy array): EMG data in sliding of shape (n_windows, time, n_channels)
        sampling_frequency (int): sampling frequency of EMG data
    
    Returns:
        X (numpy array): features of shape (n_windows, n_features)
    """

    # ---- Time features
    # Mean absolute value (MAV), axis=1 means mean along the time axis for each window
    mav = np.mean(np.abs(emg_windows), axis=1)

    # Maximum absolute Value (MaxAV)
    maxav = np.max(np.abs(emg_windows), axis=1)

    # Standard Deviation (STD)
    std = np.std(emg_windows, axis=1)

    # Root mean square (RMS)
    rms = np.sqrt(np.mean(emg_windows ** 2, axis=1))

    # Wavelength (WL)
    wl = np.sum(np.abs(np.diff(emg_windows, axis=1)), axis=1)

    # ---- Frequency features
    # Get frequency and spectrogram power 
    freqs, fft_power = calc_fft_power(emg_windows, sampling_frequency=sampling_frequency)

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
    median_frequency = np.zeros((emg_windows.shape[0],emg_windows.shape[2]))

    for i in range(emg_windows.shape[0]):
        for j in range(emg_windows.shape[2]):
            median_frequency[i,j] = freqs[np.where(cumulative_power[i, :, j] >= total_power[i,j] / 2)[0][0]]

    # Peak frequency (use np.argmax)
    peak_frequency = freqs[np.argmax(fft_power, axis=1)]

    X = np.column_stack((mav, maxav, std, rms, wl, mean_power, tot_power, mean_frequency, median_frequency, peak_frequency))

    return X

def drop_missing_values(data : tuple[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    This function drops the samples with missing values.

    Args:
        data (tuple): tuple containing the EMG data and labels
    
    Returns:
        X (numpy array): Adjusted EMG data of shape (n_samples, n_channels)
        y (numpy array): Adjusted labels of shape (n_samples, )
    """

    # Parse the data
    X, y = data

    # Get the indexes of the missing values
    missing_index = np.where(np.isnan(X))[0]

    # Remove the samples
    X = np.delete(X, missing_index, axis=0)
    y = np.delete(y, missing_index, axis=0)

    return X, y

def collect_subject_data(subject_ids : list[int], subjects_features : dict) -> tuple[np.ndarray, np.ndarray]:

    """
    This function is defined to collect the data of all subjects in one array

    Args:
        subject_ids (list): list of subject ids
        subjects_features (dict): dictionary containing the features of each subject
    
    Returns:
        X_all (numpy array): array containing the EMG data of all subjects of shape (n_samples, n_channels)
        y_all (numpy array): array containing the labels of all subjects of shape (n_samples, )
    """

    X_all, y_all = None, None
    for subject in subject_ids:
        # Get the data for the subject
        X_train, y_train, X_val, y_val, X_test, y_test = subjects_features[subject]

        # Concatenate the all the data
        X = np.concatenate([X_train, X_val, X_test], axis=0)
        y = np.concatenate([y_train, y_val, y_test], axis=0)

        # Concatenate the data for all subjects
        if X_all is None:
            X_all, y_all = X, y
        else:
            X_all = np.concatenate([X_all, X], axis=0)
            y_all = np.concatenate([y_all, y], axis=0)
         
    return X_all, y_all
