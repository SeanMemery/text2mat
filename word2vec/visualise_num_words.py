import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open('lpips.json', 'r') as f:
    data = json.load(f)

N = 20

# Initialize an empty dict to hold the averages
averages = {i: [] for i in range(1, N + 1)}

# Iterate over each material in the JSON data
for material, adj_noun in data.items():
    # Iterate over each adjective-noun pair
    for a_n, values in adj_noun.items():
            values = np.array(list(values.values()))
            values = (values - values.min()) / (values.max() - values.min())
            for i, value in enumerate(values):
                # Add the value to the list for this number of words
                averages[i+1].append(value)

# for i in range(1, 21):
#     print(f"Number of words: {i}, Average LPIPS: {sum(averages[i]) / len(averages[i])}")
#     print(f"Max LPIPS: {max(averages[i])}, Min LPIPS: {min(averages[i])}")

# Compute the average for each number of words
averages = {num_words: (sum(values) / len(values)) for num_words, values in averages.items()}

# Convert to a pandas DataFrame for easier manipulation
df1 = pd.DataFrame(list(averages.items()), columns=['Number of Words', 'Average Value'])


# # Create a bar plot
# plt.figure(figsize=(10,6))
# plt.bar(df['Number of Words'], df['Average Value'], color='blue')
# plt.xticks(df['Number of Words'])

# # Add labels and title
# plt.xlabel('Number of Words')
# plt.ylabel('Average Value')
# plt.title('Model: Average Value for Each Number of Words')

# Load the JSON data
with open('w2v.json', 'r') as f:
    data = json.load(f)

N = 20

# Initialize an empty dict to hold the averages
averages = {i: [] for i in range(1, N + 1)}

# Iterate over each material in the JSON data
for material, adj in data.items():
    values = 1.0 - np.array([v[1] for v in adj][:N])
    values = (values - values.min()) / (values.max() - values.min())
    for i, v in enumerate(values):
        averages[i+1].append(v)

# Compute the average for each number of words
averages = {num_words: (sum(values) / len(values)) for num_words, values in averages.items()}

# Convert to a pandas DataFrame for easier manipulation
df2 = pd.DataFrame(list(averages.items()), columns=['Number of Words', 'Average Value'])

# plt.figure(figsize=(10,6))
# plt.bar(df['Number of Words'], df['Average Value'], color='blue')
# plt.xticks(df['Number of Words'])

# # Add labels and title
# plt.xlabel('Number of Words')
# plt.ylabel('Average Value')
# plt.title('W2V: Average Value for Each Number of Words')


# Compare df1 and df2
df = pd.merge(df1, df2, on='Number of Words')
df.columns = ['Number of Words', 'Model', 'W2V']


# Create a scatter plot
plt.figure(figsize=(10,6))
# Scatter dots with x = df['W2V'] and y = df['Model']
plt.scatter(df['W2V'], df['Model'], color='blue', marker='x')
# Add correlation
plt.text(0.1, 0.9, f"Correlation: {df['W2V'].corr(df['Model'])}", transform=plt.gca().transAxes)
# Add line of best fit
m, b = np.polyfit(df['W2V'], df['Model'], 1)
plt.plot(df['W2V'], m*df['W2V'] + b, color='red')
# Add labels and title
plt.xlabel('W2V')
plt.ylabel('Model')
plt.title('Model vs W2V')
# plt.scatter(df['Number of Words'], df['Model'], color='blue', label='Model')
# plt.scatter(df['Number of Words'], df['W2V'], color='red', label='W2V')
#plt.xticks(df['Number of Words'])

# Add labels and title
# plt.xlabel('Number of Words')
# plt.ylabel('Average Value')
# plt.title('Average Value for Each Number of Words')


# Show the plot
plt.show()
