import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, 'test')
csv_path = os.path.join(base_dir, 'result.csv')
output_dir = os.path.join(base_dir, 'assets')

# Read results
df = pd.read_csv(csv_path)

# Select random samples
num_samples = 16
if len(df) > num_samples:
    samples = df.sample(n=num_samples, random_state=42)
else:
    samples = df

# Create figure
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Model Predictions on Test Set', fontsize=16)

for idx, (i, row) in enumerate(samples.iterrows()):
    ax = axes[idx // 4, idx % 4]
    
    img_path = os.path.join(test_dir, row['filename'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, 'Image not found', ha='center')
        
    # Color code title: Green if correct, Red if wrong
    color = 'green' if row['actual'] == row['predicted'] else 'red'
    title = f"True: {row['actual']}\nPred: {row['predicted']}"
    
    ax.set_title(title, color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Save
output_path = os.path.join(output_dir, 'sample_predictions.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Generated {output_path}")
