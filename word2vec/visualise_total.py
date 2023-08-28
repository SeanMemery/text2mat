import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_vals(N, norm=True):
    #################################################################################################################

    # Load the JSON data
    with open('lpips_50.json', 'r') as f:
        data = json.load(f)

    num_nouns = 25
    num_adjs = 10

    lpip_values = np.zeros((N*num_adjs))

    # Iterate over each material in the JSON data
    for material, adj_noun in data.items():
        noun_values = []
        for a_n, values_v in adj_noun.items():
                values = np.array(list(values_v.values())[:N])
                values = (values - values.min()) / (values.max() - values.min()) if norm else values
                for i, value in enumerate(values):
                    # Add the value to the list for this number of words
                    noun_values.append(value)
        lpip_values += np.array(noun_values)

    ### Need to average over all nouns as w2v is only adjectives
    lpip_values /= num_nouns

    #################################################################################################################

    # Load the JSON data
    with open('w2v.json', 'r') as f:
        data = json.load(f)

    w2v_values = []

    # Iterate over each material in the JSON data
    for a_n, values_v in data.items():
        values = np.array(list(values_v.values()))
        values = (values - values.min()) / (values.max() - values.min()) if norm else values
        values = values[:N]
        for i, v in enumerate(values):
            w2v_values.append(v)

    w2v_values = np.array(w2v_values)

    #################################################################################################################

    return w2v_values, lpip_values

for N in range(2, 21):

    w2v_values, lpip_values = get_vals(N=N, norm=False)
    corr = np.corrcoef(w2v_values, lpip_values)[0,1]
    t = corr * np.sqrt((len(w2v_values)-2)/(1-corr**2))
    p = 1 - stats.t.cdf(t, len(w2v_values)-2)
    print('N = {}, Correlation: {:.3f}, p-value: {:.3f}'.format(N, corr, p))


w2v_values, lpip_values = get_vals(N=3, norm=False)

# Create a scatter plot
plt.figure(figsize=(9,6))
plt.scatter(w2v_values, lpip_values, marker='x', color=(84/255, 187/255, 249/255))
# Add text of correlation score
corr = np.corrcoef(w2v_values, lpip_values)[0,1]
plt.text(0.02, 0.95, 'Correlation: {:.3f}'.format(corr), transform=plt.gca().transAxes)
# Calculate p value
t = corr * np.sqrt((len(w2v_values)-2)/(1-corr**2))
p = 1 - stats.t.cdf(t, len(w2v_values)-2)
# Add text of p value
plt.text(0.02, 0.9, 'P-Value: {:.3f}'.format(p), transform=plt.gca().transAxes)
# Add line of best fit
plt.plot(np.unique(w2v_values), np.poly1d(np.polyfit(w2v_values, lpip_values, 1))(np.unique(w2v_values)), color=(249/255, 84/255, 84/255))
# Add labels and title
plt.xlabel('Word2Vec Distance')
plt.ylabel('LPIPS Distance')
plt.title('Correlation of Word2Vec and LPIPS for 3 Neighbours')

# Show the plot
plt.show()
