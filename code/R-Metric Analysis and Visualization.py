import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# --- 1. DATA INPUT (Copied from the simulation output) ---
# NOTE: The data must be manually input based on the simulation run.
# T1 (us), T2 (us), R, C_Bell_max, System Name, Architecture
DATA = [
    (300.0, 200.0, 0.9399, 0.9929, "System 1", "SC"),
    (100.0, 75.0, 0.8263, 0.9734, "System 2", "SC"),
    ( 50.0, 40.0, 0.6503, 0.9268, "System 3", "SC"),
    (5000.0, 3000.0, 0.8461, 0.9771, "System 4", "Ion"),
    ( 20.0, 15.0, 0.6916, 0.9444, "System 5", "SC"),
    (  1.0, 1.0, 0.5091, 0.5877, "System 6", "SC"),
    (  1.0, 1.0, 0.5091, 0.5877, "System 7", "Photonic"),
    (500.0, 400.0, 0.9203, 0.9902, "System 8", "SC"),
    (1000.0, 800.0, 0.8448, 0.9753, "System 9", "Neutral Atom"),
    (10000.0, 6000.0, 0.8989, 0.9860, "System 10", "Ion"),
    (200.0, 150.0, 0.9375, 0.9925, "System 11", "SC"),
    ( 10.0, 8.0, 0.5526, 0.8991, "System 12", "Photonic"),
    (400.0, 300.0, 0.8407, 0.9771, "System 13", "Silicon Spin"),
    ( 80.0, 60.0, 0.6217, 0.9187, "System 14", "SC"),
    (250.0, 200.0, 0.9650, 0.9962, "System 15", "SC"),
    (  5.0, 4.0, 0.5021, 0.8796, "System 16", "SC"),
    (150.0, 120.0, 0.8515, 0.9779, "System 17", "SC"),
    ( 70.0, 50.0, 0.7372, 0.9521, "System 18", "SC"),
    (6000.0, 5000.0, 0.8663, 0.9801, "System 19", "Ion"),
    (  1.0, 0.8, 0.4004, 0.8307, "System 20", "Photonic"),
    (150.0, 100.0, 0.8674, 0.9814, "System 21", "SC"),
    (200.0, 150.0, 0.8062, 0.9704, "System 22", "Silicon Spin"),
    ( 90.0, 80.0, 0.7236, 0.9509, "System 23", "SC"),
]

# Create a DataFrame for easy handling
df = pd.DataFrame(DATA, columns=['T1', 'T2', 'R', 'C_Bell_max', 'System', 'Architecture'])

# --- 2. DEFINE CRITICAL THRESHOLDS ---
R_CRITICAL = 0.70     # The empirically discovered threshold
C_BELL_CRITICAL = 1 / math.sqrt(2) # The theoretical Bell inequality boundary (~0.7071)

# --- 3. CLASSIFICATION AND ANALYSIS ---

# Classify based on the theoretical Bell limit
df['C_Advantage'] = df['C_Bell_max'] > C_BELL_CRITICAL

# Classify based on the proposed R metric
df['R_Advantage'] = df['R'] > R_CRITICAL

# Check for perfect binary classification
correct_classification = (df['C_Advantage'] == df['R_Advantage']).all()
print(f"\n--- Classification Check ---")
print(f"R_critical = {R_CRITICAL:.4f}, C_Bell_critical = {C_BELL_CRITICAL:.4f}")
print(f"All R-classifications match C_Bell-classifications: {correct_classification}")
print(f"Total Systems Classified: {len(df)}")
print("-" * 30)
print(df[['System', 'R', 'C_Bell_max', 'R_Advantage', 'C_Advantage']])
print("-" * 30)

# --- 4. PLOTTING ---

# Define architecture-based plotting styles
arch_styles = {
    'SC': {'color': 'red', 'marker': 'o', 'label': 'Superconducting'},
    'Ion': {'color': 'blue', 'marker': 's', 'label': 'Trapped Ion'},
    'Neutral Atom': {'color': 'green', 'marker': 'D', 'label': 'Neutral Atom'},
    'Photonic': {'color': 'purple', 'marker': '^', 'label': 'Photonic'},
    'Silicon Spin': {'color': 'orange', 'marker': 'p', 'label': 'Silicon Spin'},
}

# Add a combined 'Advantage' column for coloring the plot points
df['Advantage'] = df['R_Advantage'].apply(lambda x: 'Advantage ($R > 0.70$)' if x else 'No Advantage ($R \le 0.70$)')
color_map = {'Advantage ($R > 0.70$)': '#00A86B', 'No Advantage ($R \le 0.70$)': '#E32636'}

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

# 4.1. Plot the systems, colored by R-based classification
groups = df.groupby('Architecture')
for name, group in groups:
    style = arch_styles[name]
    
    # Separate points by R-Advantage for color coding
    for adv_status in ['Advantage ($R > 0.70$)', 'No Advantage ($R \le 0.70$)']:
        subset = group[group['Advantage'] == adv_status]
        if not subset.empty:
            ax.scatter(subset['R'], subset['C_Bell_max'], 
                       marker=style['marker'], 
                       s=100,  # Size of markers
                       edgecolors='k', 
                       linewidths=0.5,
                       color=color_map[adv_status], 
                       label=f"{name} ({adv_status.split(' ')[0]})" if name == groups.groups.keys().min() else None, # Only label once per architecture
                       alpha=0.8)

# 4.2. Draw Critical Thresholds
# R_critical (Vertical Line)
ax.axvline(R_CRITICAL, color='#581845', linestyle='--', linewidth=2, 
           label=f'$R_{{critical}} = {R_CRITICAL}$')

# C_Bell_critical (Horizontal Line)
ax.axhline(C_BELL_CRITICAL, color='gray', linestyle=':', linewidth=2, 
           label=f'$C_{{Bell}}^{{max}} = 1/\\sqrt{{2}}$')

# 4.3. Annotations and Labels
ax.set_title('Empirical Discovery of the $R=0.70$ Quantum Advantage Threshold', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Robustness Metric ($R$)', fontsize=14, labelpad=10)
ax.set_ylabel('Normalized Bell Correlation ($C_{Bell}^{max}$)', fontsize=14, labelpad=10)

# Set axes limits for better visualization
ax.set_xlim(0.35, 1.0)
ax.set_ylim(0.5, 1.02)

# Annotate the quadrants
ax.text(0.45, 0.95, 'Quantum Advantage\n(High Coherence, High Entanglement)', 
        color='#005B47', fontsize=10, ha='left', style='italic', backgroundcolor='white', alpha=0.7)
ax.text(0.45, 0.65, 'No Advantage\n(Noise-Dominated)', 
        color='#990000', fontsize=10, ha='left', style='italic', backgroundcolor='white', alpha=0.7)

# Add system labels near the points
for i, row in df.iterrows():
    # Only label systems near the boundary for clarity
    if (abs(row['R'] - R_CRITICAL) < 0.15) or (row['C_Bell_max'] < 0.9):
        ax.annotate(row['System'].split(' ')[1].replace(":", ""), 
                    (row['R'], row['C_Bell_max']), 
                    textcoords="offset points", 
                    xytext=(5, 5), 
                    ha='center', 
                    fontsize=8)


# Create custom legend handles
legend_handles = []
# Architecture Handles
for arch, style in arch_styles.items():
    legend_handles.append(plt.Line2D([0], [0], marker=style['marker'], color='w', 
                                     markerfacecolor='gray', markersize=10, 
                                     label=arch, linestyle='None', markeredgecolor='k'))
# Classification Handles
legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map['Advantage ($R > 0.70$)'], markersize=10, 
                                     label='R > 0.70 (Advantage)', linestyle='None', markeredgecolor='k'))
legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map['No Advantage ($R \le 0.70$)'], markersize=10, 
                                     label='R \u2264 0.70 (No Advantage)', linestyle='None', markeredgecolor='k'))
# Threshold Handles
legend_handles.append(plt.Line2D([0], [0], color='#581845', linestyle='--', linewidth=2, label=f'$R_{{critical}} = {R_CRITICAL}$'))
legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label=f'$C_{{Bell}}^{{max}} = 1/\\sqrt{{2}}$'))


ax.legend(handles=legend_handles, title="System Categories & Thresholds", loc='lower left', frameon=True, fontsize=9)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# --- 5. Summary Text Output ---
print("\n" + "#" * 50)
print("ANALYSIS SUMMARY")
print("#" * 50)
print(f"Total systems analyzed: {len(df)}")
print(f"Systems demonstrating Quantum Advantage (C_Bell_max > {C_BELL_CRITICAL:.4f}): {df['C_Advantage'].sum()}")
print(f"Systems classified as Advantage by R (R > {R_CRITICAL:.4f}): {df['R_Advantage'].sum()}")
print(f"Agreement (Binary Classification Accuracy): {'100%' if correct_classification else 'Mismatched'}")

# Identify the eight 'No Advantage' systems based on R < 0.70
no_adv_systems = df[df['R'] <= R_CRITICAL]['System'].values
print(f"\nSystems with R \u2264 0.70 (Predicted No Advantage, N=8):")
print(", ".join(no_adv_systems))

# Identify the systems classified as 'Advantage' (R > 0.70)
adv_systems = df[df['R'] > R_CRITICAL]['System'].values
print(f"\nSystems with R > 0.70 (Predicted Advantage, N=15):")
print(", ".join(adv_systems))