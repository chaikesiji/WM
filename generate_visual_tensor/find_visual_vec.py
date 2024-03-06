import os
import torch
import json
import time
import numpy as np
from tqdm import tqdm

# 生成单字符视觉特征替换表
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"#把1改成0**
    config = json.load(open('/home/xy/DB-WM/csv_word_modify/data/config_file.json', 'r'))
    # 这里是加载对应的tokenizer编码后的结果
    suffix_id_list = np.load(config['suffix_id_path']).tolist()
    # 这里加载视觉特征
    suffix_tensor_map = torch.load(config['visual_suffix_tensor_path'])
    result = []
    for v_id in suffix_id_list:
        result.append(suffix_tensor_map[v_id])
    result = torch.stack(result).cuda()
    multi_label = torch.zeros((998, 998))
    multi_count = torch.zeros((998,))
    m = 2
    for i in tqdm(range(len(suffix_id_list))):
        tmp = find(i, result)
        scores, indices = torch.topk(tmp, k=998, largest=False)
        for idx in indices[0]:
            if i == idx:
                multi_label[i, i] = 1
                multi_count[i] += 1
                continue
            if multi_count[idx] >= (m - 1):
                continue
            if multi_count[i] == m:
                break
            multi_label[i, idx] = 1
            multi_label[idx, i] = 1
            multi_count[i] += 1
            multi_count[idx] += 1
    torch.save(multi_label, config['visual_tensor_path'])


def find(sf_id, visual_vec):
    sf_vec = visual_vec[sf_id]
    num = 998
    list_score = torch.zeros(1, num)
    for i in range(num):
        if i == sf_id:
            list_score[0, i] = -9999
        else:
            list_score[0, i] = torch.norm(sf_vec - visual_vec[i], dim=0)

    return list_score


if __name__ == '__main__':
    s = time.time()
    main()
    print(f"Total time {(time.time() - s) / 60}")
