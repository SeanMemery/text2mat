import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the CSV data
data = pd.read_csv('interpolate_gt.csv')

# Convert these strings into lists of floats
data['lpips_1'] = data['lpips_1'].apply(lambda x: list(map(float, x.split(','))))

# # Normalise data per line
# data['lpips_1'] = data['lpips_1'].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# Get x values, ranging from 0 to 1
x_values = np.linspace(0, 1, len(data['lpips_1'][0]))

# Compute mean and confidence interval for lpips_1
lpips_1_mean = np.mean(data['lpips_1'].tolist(), axis=0)
lpips_1_std = np.std(data['lpips_1'].tolist(), axis=0)
lpips_1_yerr = lpips_1_std / np.sqrt(len(data['lpips_1'])) * stats.t.ppf(1-0.05/2, len(data['lpips_1']) - 1)

# Plot mean lpips_1 with confidence interval
plt.figure(1)
plt.plot(x_values, lpips_1_mean, label='Interpolated Material')
plt.fill_between(x_values, lpips_1_mean - lpips_1_yerr, lpips_1_mean + lpips_1_yerr, alpha=0.2)
plt.title('Interpolation of Ground Truth and Interpolation of Predictions')
plt.xlabel('Interpolation Parameter')
plt.ylabel('LPIPS Distance')
plt.legend()

# Convert these strings into lists of floats
data['lpips_2_A'] = data['lpips_2_A'].apply(lambda x: list(map(float, x.split(','))))
data['lpips_2_B'] = data['lpips_2_B'].apply(lambda x: list(map(float, x.split(','))))

# # Normalise data per line
# data['lpips_2_A'] = data['lpips_2_A'].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# data['lpips_2_B'] = data['lpips_2_B'].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# Compute mean and confidence interval for lpips_2_A and lpips_2_B
lpips_2_A_mean = np.mean(data['lpips_2_A'].tolist(), axis=0)
lpips_2_A_std = np.std(data['lpips_2_A'].tolist(), axis=0)
lpips_2_A_yerr = lpips_2_A_std / np.sqrt(len(data['lpips_2_A'])) * stats.t.ppf(1-0.05/2, len(data['lpips_2_A']) - 1)

lpips_2_B_mean = np.mean(data['lpips_2_B'].tolist(), axis=0)
lpips_2_B_std = np.std(data['lpips_2_B'].tolist(), axis=0)
lpips_2_B_yerr = lpips_2_B_std / np.sqrt(len(data['lpips_2_B'])) * stats.t.ppf(1-0.05/2, len(data['lpips_2_B']) - 1)

# Plot mean lpips_2_A and lpips_2_B with confidence interval
plt.figure(2)
plt.plot(x_values, lpips_2_A_mean, label='Material 1')
plt.fill_between(x_values, lpips_2_A_mean - lpips_2_A_yerr, lpips_2_A_mean + lpips_2_A_yerr, alpha=0.2)

plt.plot(x_values, lpips_2_B_mean, label='Material 2')
plt.fill_between(x_values, lpips_2_B_mean - lpips_2_B_yerr, lpips_2_B_mean + lpips_2_B_yerr, alpha=0.2)

plt.title('Ground Truth and Interpolation of Predictions')
plt.xlabel('Interpolation Parameter')
plt.ylabel('LPIPS Distance')
plt.legend()

# # Load the CSV data
# data = pd.read_csv('interpolate_t.csv')

# # Convert these strings into lists of floats
# data['clip_score1'] = data['clip_score1'].apply(lambda x: list(map(float, x.split(','))))
# data['clip_score2'] = data['clip_score2'].apply(lambda x: list(map(float, x.split(','))))

# # Compute mean and confidence interval for lpips_2_A and lpips_2_B
# clip1_mean = np.mean(data['clip_score1'].tolist(), axis=0)
# clip1_std = np.std(data['clip_score1'].tolist(), axis=0)
# clip1_yerr = clip1_std / np.sqrt(len(data['clip_score1'])) * stats.t.ppf(1-0.05/2, len(data['clip_score1']) - 1)

# clip2_mean = np.mean(data['clip_score2'].tolist(), axis=0)
# clip2_std = np.std(data['clip_score2'].tolist(), axis=0)
# clip2_yerr = clip2_std / np.sqrt(len(data['clip_score2'])) * stats.t.ppf(1-0.05/2, len(data['clip_score2']) - 1)

# # Plot mean clip_score1 and clip_score2 with confidence interval
# plt.figure(3)
# plt.plot(x_values, clip1_mean, label='clip_score1')
# plt.fill_between(x_values, clip1_mean - clip1_yerr, clip1_mean + clip1_yerr, alpha=0.2)

# plt.plot(x_values, clip2_mean, label='clip_score2')
# plt.fill_between(x_values, clip2_mean - clip2_yerr, clip2_mean + clip2_yerr, alpha=0.2)

# plt.title('CLIP Similarity Score of Interpolation of Text Embeddings')
# plt.xlabel('Interpolation Parameter')
# plt.ylabel('CLIP Similarity Score')
# plt.legend()

# Display the plots
plt.show()