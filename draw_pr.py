import os 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


### prepare data
with open("pr.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(len(lines))

names = []
pr = []
item = []

for line in lines:
    
    if line[0]=='=':
        names.append(line.strip()[1:-1].upper())
        item = []
    elif line[0] in 'PR[':
        nums = line.strip().split("[")[1].split("]")[0].split(", ")
        nums = [float(num) for num in nums]
        item.append(nums)

    if len(item)==4:
        pr.append(item)
        item = []

print(names, len(names))
print(pr[0], len(pr), len(pr[0]))



# Generate some random PR data for 10 algorithms  
# precisions = [np.random.rand(10) for _ in algos]
# recalls = np.linspace(0, 1, 10)

# Plot styling  
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

markers = ['^', 's', 'o', '*', 'P', 'X', 'D', '+', 'h', 'p', '^']  
colors = plt.cm.rainbow(np.linspace(0, 1, len(names)))  

# Plot PR curves
for i in range(len(names)):
    ax1.plot(pr[i][0], pr[i][1], marker=markers[i], color=colors[i], linestyle='--', label=names[i])
    
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')   
ax1.set_title('PR Curves')
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))   

# Draw legend in extra axis
for i in range(len(names)):
    ax2.plot([], [], marker=markers[i], color=colors[i], linestyle='--', label=names[i])

ax2.legend(loc='center')
ax2.axis('off')  

# Adjust layout  
fig.subplots_adjust(wspace=0.3)
plt.show()