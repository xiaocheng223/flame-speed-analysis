import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

def extract_radius_from_images(image_folder, calibration_factor, frame_rate):
    """Processes sequential .tif images to extract flame radius over time."""
    radii = []
    times = []
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])

    if not image_filenames:
        print(f"Error: No .tif images found in {image_folder}")
        return np.array([]), np.array([])

    for i, filename in enumerate(image_filenames):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the flame)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit a circle to the largest contour
            (x, y), radius_pixels = cv2.minEnclosingCircle(largest_contour)
            radii.append(radius_pixels * calibration_factor)
            times.append(i / frame_rate)
        else:
            print(f"Warning: No contours found in image {filename}. Skipping.")
            radii.append(np.nan) # Append NaN if no radius found
            times.append(i / frame_rate) # Still record time

    return np.array(times), np.array(radii)

def smooth_and_differentiate(time_data, radius_data, window_length, polyorder):
    """Applies Savitzky-Golay filter and differentiates radius data to get Sb."""
    # Filter out NaN values before smoothing
    valid_indices = ~np.isnan(radius_data)
    time_data_filtered = time_data[valid_indices]
    radius_data_filtered = radius_data[valid_indices]

    if len(radius_data_filtered) < window_length:
        print("Warning: Not enough valid data points for Savitzky-Golay filter. Skipping smoothing.")
        smoothed_radius_data = radius_data_filtered
    else:
        smoothed_radius_data = savgol_filter(radius_data_filtered, window_length, polyorder)

    # Calculate apparent flame propagation speed (Sb) by differentiation
    sb_data = np.gradient(smoothed_radius_data, time_data_filtered)
    
    return smoothed_radius_data, sb_data, time_data_filtered, radius_data_filtered

def compute_speeds_and_stretch(sb_data, smoothed_radius_data, Tu, Tb):
    """Computes unburned-gas flame speed (Su) and stretch rate (K)."""
    # Convert to unburned-gas flame speed (Su)
    su_data = (Tu / Tb) * sb_data

    # Compute stretch rate (K)
    # Avoid division by zero for initial radius values
    k_data = np.where(smoothed_radius_data != 0, 2 * sb_data / smoothed_radius_data, 0)
    
    return su_data, k_data

def fit_markstein_relation(su_data, k_data):
    """Performs linear regression for the Markstein relation."""
    # Filter out NaN values that might arise from initial data points or division by zero
    valid_indices = ~np.isnan(su_data) & ~np.isnan(k_data)
    su_data_filtered = su_data[valid_indices]
    k_data_filtered = k_data[valid_indices]

    if len(su_data_filtered) < 2:
        print("Warning: Not enough valid data points for linear regression. Returning None for S_L0 and L.")
        return None, None, None # Return None for S_L0, L, and regression line

    slope, intercept, r_value, p_value, std_err = linregress(k_data_filtered, su_data_filtered)
    s_l0 = intercept
    L = -slope
    
    # Generate regression line for plotting using the original (potentially unfiltered) K values
    regression_line = intercept + slope * k_data
    
    return s_l0, L, regression_line

def plot_results(time_data, raw_radius_data, smoothed_radius_data, sb_data, su_data, k_data, regression_line):
    """Generates and displays plots."""
    plt.figure(figsize=(18, 6))

    # Plot 1: Flame radius vs. time (raw + smoothed)
    plt.subplot(1, 3, 1)
    plt.plot(time_data, raw_radius_data, 'b.', label='Raw Radius')
    plt.plot(time_data, smoothed_radius_data, 'r-', label='Smoothed Radius')
    plt.xlabel('Time [s]')
    plt.ylabel('Radius [m]')
    plt.title('Flame Radius vs. Time')
    plt.legend()
    plt.grid(True)

    # Plot 2: Sb vs. time
    plt.subplot(1, 3, 2)
    plt.plot(time_data, sb_data, 'g-', label='Apparent Flame Speed (Sb)')
    plt.xlabel('Time [s]')
    plt.ylabel('Sb [m/s]')
    plt.title('Apparent Flame Propagation Speed vs. Time')
    plt.legend()
    plt.grid(True)

    # Plot 3: Su vs. K with regression line
    plt.subplot(1, 3, 3)
    plt.plot(k_data, su_data, 'k.', label='Su vs. K Data')
    if regression_line is not None:
        # Ensure k_data and regression_line have the same length for plotting
        # This might be an issue if k_data had NaN values that were filtered out during regression
        # For plotting, we use the original k_data and the generated regression_line based on it.
        plt.plot(k_data, regression_line, 'm--', label='Linear Regression')
    plt.xlabel('Stretch Rate K [1/s]')
    plt.ylabel('Unburned Flame Speed Su [m/s]')
    plt.title('Markstein Relation: Su vs. K')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def save_data_to_csv(filepath, time_data, raw_radius_data, smoothed_radius_data, sb_data, su_data, k_data):
    """Saves processed data to a CSV file."""
    data = {
        'time': time_data,
        'raw_radius': raw_radius_data,
        'smoothed_radius': smoothed_radius_data,
        'Sb': sb_data,
        'Su': su_data,
        'K': k_data
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")

def main():
    # --- Input Parameters ---
    image_folder = 'flame_images'  # Folder containing .tif images
    calibration_factor = 1e-5     # Example: 1 pixel = 1e-5 meters
    frame_rate = 5000             # Frames per second
    Tu = 300                      # Initial unburned gas temperature [K]
    Tb = 2000                     # Adiabatic flame temperature [K]

    # Savitzky-Golay filter parameters
    window_length = 11            # Must be odd
    polyorder = 3                 # Polynomial order
    
    output_csv_filepath = 'processed_flame_data.csv'

    # --- 1. Image processing to extract flame radius ---
    print(f"Processing images from {image_folder}...")
    times, raw_radii = extract_radius_from_images(image_folder, calibration_factor, frame_rate)

    if len(times) == 0 or np.all(np.isnan(raw_radii)):
        print("No valid radius data extracted. Exiting.")
        return

    # --- 2. Preprocessing (Smoothing and Differentiation) ---
    print("Smoothing data and calculating apparent flame speed...")
    smoothed_radii, sb_data, filtered_times, filtered_raw_radii = smooth_and_differentiate(times, raw_radii, window_length, polyorder)

    # --- 3, 4, 5. Calculate unburned-gas flame speed (Su) and stretch rate (K) ---
    print("Calculating unburned-gas flame speed and stretch rate...")
    su_data, k_data = compute_speeds_and_stretch(sb_data, smoothed_radii, Tu, Tb)

    # --- 6. Linear fit for Markstein relation ---
    print("Performing Markstein regression...")
    s_l0, L, regression_line = fit_markstein_relation(su_data, k_data)

    # --- 7. Outputs ---
    print("\n--- Results ---")
    if s_l0 is not None and L is not None:
        print(f"Unstretched Laminar Burning Velocity (S_L0): {s_l0:.4f} m/s")
        print(f"Markstein Length (L): {L:.4f} m")
    else:
        print("Could not determine S_L0 and L due to insufficient data for regression.")

    print("Generating plots...")
    plot_results(filtered_times, filtered_raw_radii, smoothed_radii, sb_data, su_data, k_data, regression_line)

    print("Saving processed data to CSV...")
    save_data_to_csv(output_csv_filepath, filtered_times, filtered_raw_radii, smoothed_radii, sb_data, su_data, k_data)

if __name__ == "__main__":
    main()
