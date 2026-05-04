import matplotlib.pyplot as plt
import numpy as np

# dados hardcoded da primeira run. 
# TODO: talvez deva rodar de novo se preciso
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
training_vals = [99.83, 99.83, 99.83, 99.83]
validation_vals = [93.75, 94.47, 93.75, 93.72]
test_vals = [91.67, 92.08, 91.67, 91.62]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, training_vals, width, label='train', color='#2ecc71')
rects2 = ax.bar(x, validation_vals, width, label='val', color='#3498db')
rects3 = ax.bar(x + width, test_vals, width, label='test', color='#e74c3c')

ax.set_ylabel('score (%)')
ax.set_title('MLP performance (8 neuronios)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 110)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('mlp_performance.png') 
plt.show()