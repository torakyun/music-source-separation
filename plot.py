import numpy as np
import argparse
import os
from pathlib import Path
import json

parser = argparse.ArgumentParser("plot", description="Discriminator Loss Plot")
parser.add_argument("--logs",
                        type=Path,
                        default=Path("logs"),
                        help="Folder where to store logs")

json_open = open('logs/raw=raw musdb=musdb18 use_gan=True input_D=outputs.json', 'r')
json_load = json.load(json_open)
# プロットするデータを用意
x = []
y1 = []
y2 = []
for epoch, d in enumerate(json_load):
    x.append(epoch)
    y1.append(d['D_fake'])
    y2.append(d['D_fake']+d['D_real'])
y1 = np.array(y1)
y2 = np.array(y2)

import matplotlib as mpl
mpl.use('Agg') # この行を追記
import matplotlib.pyplot as plt

# FigureとAxesの設定
fig = plt.figure(figsize=(8, 6), dpi=100) # (x, y)=(8*100, 6*100)
ax = fig.add_subplot(111) # 分割数 1×1 の1つ目
ax.grid()
ax.set_xlabel("Epoch", fontsize = 14)
ax.set_ylabel("Discriminator Loss", fontsize = 14)
ax.set_xlim(0, 120) # ｘ範囲
ax.set_ylim(0.0, 2.0) # ｙ範囲
#ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi]) # 目盛り間隔
#ax.set_xticklabels(["0", "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"], fontsize = 12) # 目盛り

# Axesにグラフをプロット
#ax.plot(x, y1, color = "blue", linestyle='solid', linewidth = 3.0, label='Discriminator Loss')
ax.plot(x, y2, color = "black", linestyle='solid', linewidth = 3.0)

# 塗り潰す
ax.fill_between(x, y1, y2, facecolor='orange', alpha=0.3, label='D=real')
ax.fill_between(x, y1, facecolor='blue', alpha=0.3, label='D_fake')

# legend and title
#ax.legend(loc='best')
#ax.set_title('Plot of sine and cosine')

#plt.show()
plt.savefig("D_loss.png") # この行を追記