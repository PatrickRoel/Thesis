import pandas as pd
import os
import matplotlib.pyplot as plt

# Read the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "beam_comparison_results.csv")

df = pd.read_csv(csv_path)

# Create the plot
fig, ax = plt.subplots(figsize=(4,4))
ax.grid()
ax.scatter(df['Element Length (m)'], df['Deflection error (%)'], marker='+',color = "black",zorder=20)
ax.set_xlabel('Element length (m)')
ax.set_ylabel('Deflection error (%)')
# plt.title('Error vs Element Length')
fig.tight_layout()
fig.savefig(os.path.join(script_dir, 'beamcomparison.pdf'), format='pdf')
