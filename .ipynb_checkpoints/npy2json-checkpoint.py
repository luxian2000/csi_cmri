import json
import numpy as np

data = np.load('../DataSpace/csi_cmri/CSI_channel_30km.npy')

with open('../DataSpace/csi_cmri/CSI_channel_30km.jsonl', 'w', encoding='utf-8') as f:
    for row in data:
        # 每一行是一个样本（列表或数组），转为 JSON 写入一行
        f.write(json.dumps(row.tolist()) + '\n')  # 转为 list 再转 JSON

print("已保存为 CSI_channel_30km.jsonl")
