import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def calc_mean_erp(trial_points_file, ecog_data_file):
    # Load data
    trial_points = pd.read_csv(trial_points_file)
    ecog_data = pd.read_csv(ecog_data_file, header=None)

    # Ensure trial_points data is of type int
    trial_points = trial_points.astype(int)

    # Initialize variables
    num_fingers = 5
    signal_length = 1201  # 200 ms before, 1 ms at start, 1000 ms after
    fingers_erp_mean = np.zeros((num_fingers, signal_length))

    # Process each finger
    for finger in range(1, num_fingers + 1):
        # Filter trial points for the current finger
        finger_trials = trial_points[trial_points.iloc[:, 2] == finger]

        # Initialize list to collect signals for the current finger
        finger_signals = []

        for _, trial in finger_trials.iterrows():
            start_idx = trial[0] - 200
            end_idx = trial[0] + 1000

            # Extract signal if within bounds
            if start_idx >= 0 and end_idx < len(ecog_data):
                signal = ecog_data.iloc[start_idx:end_idx + 1, 0].values
                finger_signals.append(signal)

        # Calculate mean ERP for the current finger
        if finger_signals:
            fingers_erp_mean[finger - 1] = np.mean(finger_signals, axis=0)

    # Plot the averaged brain response for each finger
    time_axis = np.linspace(-200, 1000, signal_length)  # Time in ms
    plt.figure(figsize=(10, 8))
    for finger in range(num_fingers):
        plt.plot(time_axis, fingers_erp_mean[finger], label=f'Finger {finger + 1}')

    plt.title('Averaged Brain Response per Finger')
    plt.xlabel('Time (ms)')
    plt.ylabel('Brain Signal (uV)')
    plt.legend()
    plt.grid()
    plt.show()

    # Display the resulting matrix as an example
    fingers_erp_mean_df = pd.DataFrame(fingers_erp_mean, 
                                       index=[f'Finger {i+1}' for i in range(fingers_erp_mean.shape[0])],
                                       columns=[f'Time {i}' for i in range(fingers_erp_mean.shape[1])])
    print("Example of ERP Mean Matrix:")
    print(fingers_erp_mean_df.head())


    return fingers_erp_mean

# Example usage


fingers_erp_mean = calc_mean_erp("events_file_ordered.csv", "brain_data_channel_one.csv")

