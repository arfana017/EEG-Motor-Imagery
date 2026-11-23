import numpy as np
from scipy.signal import butter, lfilter

# SIMULATE EEG DATA (Multi-channel)

def simulate_eeg_data(num_samples=2500, fs=250, num_channels=8):
    """
    Simulates multi-channel EEG data (10 seconds at 250 Hz).
    Adds background noise + synthetic mu/beta oscillations for motor imagery.
    """
    time = np.linspace(0, num_samples / fs, num_samples)
    
    # Generate background pink/white noise across channels
    eeg_data = np.random.normal(0, 1, (num_channels, num_samples))
    
    # Simulate a motor imagery event (ex imagination of movement changes signal power)
    # Let's inject a modulated 12 Hz mu-rhythm into channels 3 and 4 (motor cortex area)
    mu_rhythm = np.sin(2 * np.pi * 12 * time)
    
    # Add event modulation (simulate left vs right hand imagery periods)
    eeg_data[3, :] += 2.0 * mu_rhythm * np.cos(2 * np.pi * 0.5 * time)
    eeg_data[4, :] -= 1.5 * mu_rhythm * np.sin(2 * np.pi * 0.5 * time)
    
    return eeg_data, time


# SIGNAL FILTERING (Bandpass Filter)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply filter along the time axis (axis=1)
    filtered_data = lfilter(b, a, data, axis=1)
    return filtered_data


# FEATURE EXTRACTION & CLASSIFICATION

def extract_band_power(filtered_data):
    """
    Extracts variance (power) of the signal across each channel 
    as a simple feature vector for classification.
    """
    return np.var(filtered_data, axis=1)

# MAIN
if __name__ == "__main__":
    FS = 250  # Sampling frequency (Hz)
    print("--- Initializing BCI EEG Simulation Pipeline ---")
    
    # Step 1: Generate simulated raw EEG
    raw_eeg, t = simulate_eeg_data(num_samples=2500, fs=FS, num_channels=8)
    print(f"Generated raw EEG shape: {raw_eeg.shape} (Channels x Samples)")
    
    # Step 2: Filter for Mu & Beta rhythms (8 Hz to 30 Hz)
    filtered_eeg = bandpass_filter(raw_eeg, lowcut=8.0, highcut=30.0, fs=FS, order=4)
    print("Applied 8-30 Hz Bandpass Filter (isolating sensorimotor rhythms).")
    
    # Step 3: Extract features (Band Power)
    features = extract_band_power(filtered_eeg)
    print(f"Extracted Feature Vector (Channel Variances): {np.round(features, 4)}")
    
    print("\nPipeline executed successfully! Ready to expand into classification models.")
