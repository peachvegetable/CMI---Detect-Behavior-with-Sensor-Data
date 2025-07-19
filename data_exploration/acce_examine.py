import pandas as pd
import matplotlib.pyplot as plt
import random
import os

df = pd.read_csv('data/train.csv')

BFRB_gestures = ["Cheek - pinch skin", "Forehead - pull hairline", "Neck - scratch", "Neck - pinch skin", "Eyelash - pull hair"]
non_BFRB_gestures = ["Write name on leg", "Feel around in tray and pull out an object", "Wave hello", "Write name in air", "Text on phone"]

n_sequences_per_gesture = 10   # plot at most this many distinct sequence_id’s per gesture
random.seed(0)                # reproducible sampling

base_output_dir = 'accel_plots_by_orientation'

def plot_gesture_orientation_accel(gesture, orientation, col_name):
    subset = df[(df['gesture'] == gesture) & (df['orientation'] == orientation)]
    seq_ids = subset['sequence_id'].unique()
    seq_ids = random.sample(list(seq_ids), min(len(seq_ids), n_sequences_per_gesture))

    plt.figure(figsize=(10, 4))
    for sid in seq_ids:
        g = subset[subset['sequence_id'] == sid]
        plt.plot(g['sequence_counter'], g[col_name], alpha=0.5, label=f'seq {sid}')
    
    title = f"{col_name} of '{gesture}' ({orientation})"
    plt.title(title)
    plt.xlabel('sequence_counter')
    plt.ylabel(col_name)
    plt.legend(fontsize=6, ncol=2, frameon=False)
    plt.tight_layout()
    
# Make sure folder names are safe
    safe_gesture = gesture.replace('/', '-')
    safe_orientation = orientation.replace('/', '-')
    
    # Create subfolder: gesture / orientation
    output_dir = os.path.join(base_output_dir, safe_gesture, safe_orientation)
    os.makedirs(output_dir, exist_ok=True)

    # Save with axis name
    plt.savefig(os.path.join(output_dir, f"{col_name}.png"))
    plt.close()

# Loop through gestures, orientations, and axes
for gesture in BFRB_gestures + non_BFRB_gestures:
    orientations = df[df['gesture'] == gesture]['orientation'].unique()
    for orientation in orientations:
        for col in ['acc_x', 'acc_y', 'acc_z']:
            plot_gesture_orientation_accel(gesture, orientation, col)