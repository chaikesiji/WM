'''
嵌入水印的代码,仅需要确定修改文件和权重即可
'''
import copy
import json
from transformers import AutoTokenizer
import torch
import pandas as pd
from datasets import Dataset
import os
import numpy as np
import random
from model_primarykey_col import VPKCOL
from utils_motifs_pk import modify
from tqdm import tqdm
import csv


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config_file(config_path):
    assert 'json' in config_path, "file type is not json"
    return json.load(open(config_path, 'r'))

#BERT模型的词汇表中提取特定格式的后缀单词及其对应的ID，并将这些数据保存为numpy数组。
def get_suffix_vocab_list(tokenizer, adding_pad=False, model_name='bert-base-uncased'):
    words_info = [(tok, ids) for tok, ids in tokenizer.vocab.items()]
    words_info = sorted(words_info)
    suffix_id_list = []
    suffix_list = []
    assert model_name == 'bert-base-uncased'
    for word, word_id in words_info:
        if word.startswith('##') and (len(word) == 3 or len(word) == 4):
            suffix_id_list.append(word_id)
            suffix_list.append(word)
    np.save('./data/suffix_id_one_char_list.npy', suffix_id_list)
    np.save('./data/suffix_one_char_list.npy', suffix_list)
    return suffix_id_list, suffix_list


def get_vocab_visual_tensor(vocab_ids, visual_path):
    suffix_tensor_map = torch.load(visual_path)
    result = []
    for v_id in vocab_ids:
        result.append(suffix_tensor_map[v_id])
    result = torch.stack(result).cuda()
    return result


def load_data(config):
    df = pd.read_csv(config['csv_path']).dropna()
    df = pd.DataFrame(df)
    # 选取多少行数据
    train_df = df.tail(10000)
    test_dataset = Dataset.from_pandas(train_df, split="train")
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenzier_cache_path'],
        add_prefix_space=True,
    )
    tqdm.write("Dateset loading complete!")
    return test_dataset, tokenizer


def preprocess_char_data(examples):
    tmp_dataset = {}
    for key, values in examples.items():
        tmp_dataset[key] = values
    return tmp_dataset

#用于评估（或测试）一个预训练的模型
def evaluate(model, tokenizer, val_dataset, col_name, device, config):
    vis = torch.load(config["visual_tensor_path"])
    labels = np.load(config["suffix_path"]).tolist()
    bsz = config['val_bsz']
    vpk_value = torch.zeros([val_dataset.num_rows, 32], dtype=torch.float32, device=device)
    wm_list = np.load(config['wm_path']).tolist()
    wm_len = len(wm_list)

    # 划分数据集
    single_num = (val_dataset.num_rows // wm_len)
    split_num = [i * single_num for i in range(1, wm_len)]
    if split_num[-1] != val_dataset.num_rows:#数据集不能被完整地分成wm_len个等份
        split_num.append(val_dataset.num_rows)#确保最后一个子集包含剩余的所有数据
    data_dict_ori = {value: [] for value in col_name}
    j = 0
    # 不进行梯度计算
    with torch.no_grad():#上下文管理器
        bsz_num = val_dataset.num_rows // bsz
        if bsz_num * bsz == val_dataset.num_rows:
            range_list = range(0, (val_dataset.num_rows // bsz))
        else:
            range_list = range(0, (val_dataset.num_rows // bsz) + 1)
        for bsz_num in tqdm(range_list):
            col_choice, select_inputs, select_words, select_sentences = [], [], [], []
            select_sentences_part = []
            data_tensor_dict = {}
            data_dict = copy.deepcopy(data_dict_ori)
            if bsz_num != range_list[-1]:
                start_index = bsz_num * bsz
                end_index = (bsz_num + 1) * bsz
            else:
                start_index = end_index + 1
                end_index = val_dataset.num_rows

            # 对提取出来的行进行编码并放到对应device上
            encode_dict = {key: np.array(value).astype(str).tolist() for key, value in
                           val_dataset[start_index:end_index].items()}#每个值都变成了字符串列表
            for col in col_name:
                data_tensor = tokenizer(encode_dict[col], max_length=512, padding=True, truncation=True,
                                        return_tensors='pt')
                if device != 'cpu':
                    data_tensor_dict[col] = {k: v.cuda() for k, v in data_tensor.items()}

            # vpk生成
            model.eval()
            sentence_fe, sentence_dg = model.forward_(data_tensor_dict, col_name)
            sentence_dg_copy = copy.deepcopy(sentence_dg.detach())
            # 生成的结果是float,需要四舍五入到0或者1,sigmoid是限制范围在0和1之间
            sentence_dg_sig = torch.sigmoid(sentence_dg_copy)
            sentence_dg_round = torch.round(sentence_dg_sig)
            vpk_value[start_index:end_index] = sentence_dg_round

        # VPK从2进制转换为10进制
        decimal_range = torch.arange(vpk_value.shape[1], device='cuda:0').expand(vpk_value.shape[0], -1)
        decimal_value = 2 ** decimal_range
        all_words_vpk_decimal = torch.sum(vpk_value * decimal_value, dtype=torch.long, dim=1)
        # 通过all_words_vpk_decimal能够判断是否嵌入水印
        torch.save(all_words_vpk_decimal, config['tuple_vpk_path'])
        col_idx = tokenizer(col_name, add_special_tokens=False, return_tensors='pt', padding=True, max_length=512,
                            truncation=True).data['input_ids']
        col_emb = torch.sum(col_idx, dim=-1).to(device)
        # VPK具体到元组上
        all_sk_score = all_words_vpk_decimal.unsqueeze(1) + col_emb.unsqueeze(0)
        #处理嵌入水印的值，并将其与某个输入列的嵌入值结合，得到一个最终的分数或表示

        # 当元组属性的VPK % 9 = 4时嵌入水印,这个条件可以换.值不同嵌入量不同
        remainder = 4
        condition = (all_sk_score % 9 == remainder)
#找到all_sk_score中模9余数为4的所有元素的位置索引，并将这些位置的值保存为1（满足条件）或0（不满足条件）
        choose_col = torch.where(condition)
        choose_col_one = torch.where(condition, torch.tensor(1), torch.tensor(0))
        torch.save(choose_col_one, config['attribute_vpk_path'])

        # 可能会导致csv文件显示成#name?的词
        csv_data = [val_dataset.column_names]
        special_characters = ['-', '=', '+', '@', '{', '}', '#', ',', '>', '<', '*', '/']
        count_num = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for row in tqdm(range(val_dataset.num_rows)):
            # 分层进行水印嵌入,例如前1000行仅嵌入第一位水印
            if row < split_num[j]:
                wm = wm_list[j]
            else:
                wm = wm_list[j + 1]#跨越分段时，使用新的水印
                j += 1
            # 遍历所有需要嵌入的单元格,先找行,再找列
            if row in choose_col[0]:
                tensor_index = torch.where(choose_col[0] == row)[0]
                col = choose_col[1][tensor_index]#根据row的值，从某个二维张量中选取特定的列
                non_first = torch.nonzero(col)
                non_first_col = col[non_first].squeeze()
                row_ori_data = list(val_dataset[row].values())#基于当前的行号，从某个数据集中提取特定的行数据
                for idx in col:
                    row_score = all_sk_score[row][idx]
                    wait_change_data = str(row_ori_data[idx]).lstrip()
                    # 根据水印长度来选取单元格的字符位置
                    data_len = len(wait_change_data)
                    change_idx = row_score.item() % data_len
                    # 所选字符不可以是' '
                    if wait_change_data[change_idx] == ' ':
                        continue
                    # 嵌入水印,依据视觉特征生成的表
                    modify_char = modify(wait_change_data[change_idx], wm, vis, labels)
                    # 如果想仅在数字之间转换可以取消这里的注释(该注释后四行)
                    # if wait_change_data[change_idx] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    #     change_idx += 1
                    # modify_char = modify_digital(wait_change_data[change_idx], wm)
                    # count_num[modify_char] += 1

                    # 如果修改第一个字符并且第一个字符是特殊字符则放弃修改,这里如果不在乎CSV文件可能显示错误可以注释
                    if change_idx == 0 and modify_char in special_characters:
                        modify_char = wait_change_data[change_idx]#将修改后的字符重置为原始字符
                    modify_str_list = list(wait_change_data)

                    # 修改内容
                    modify_str_list[change_idx] = modify_char
                    modify_str = ''.join(modify_str_list)#修改后的字符列表重新组合成字符串
                    row_ori_data[idx] = modify_str#将修改后的字符串存储回原始数据中的相应位置
                row_data = row_ori_data
            else:
                row_data = list(val_dataset[row].values())
            csv_data.append(row_data)

        # 输出CSV
        with open(config["output_path"], mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)#使用CSV写入器将csv_data中的数据写入到文件中

        print('Finished')


def main(device):
    # 配置文件
    config = load_config_file('/home/xy/DB-WM/csv_word_modify/data/config_file.json')#*
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"#*1改成0
    setup_seed(config["random_seed"])
    # 加载数据集，tokenzier，单字符、单字符对应的序号，单字符的视觉特征
    test_dataset, tokenizer = load_data(config)
    # 读取到数据集，preprocess_char_data是对数据的预处理。这里仅简单转换为字典形式.
    encode_test_dataset = test_dataset.map(preprocess_char_data,
                                           batched=True,
                                           remove_columns=test_dataset.column_names,
                                           load_from_cache_file=False
                                           )

    col_name = encode_test_dataset.column_names

    # 加载模型
    model = VPKCOL(config).to(device)

    # forest是纯数字类型的数据权重,另一个是混合型
    #*加载预训练权重
    checkpoint_path = config['checkpoint_path']
    checkpoint = torch.load(checkpoint_path)

    # 加载权重并判断哪些是模型修改的地方,仅加载名字相同的部分
    #*检查预训练权重与当前模型中参数的匹配情况，并仅加载匹配的权重
    #*准备加载模型权重
    state_dict = checkpoint.get('model_state', checkpoint)
    #*初始化一些变量以跟踪不匹配和缺失的键
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)#用于跟踪哪些参数在当前模型中存在，但尚未在预训练权重中找到
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data#检查当前预训练权重的键是否存在于missing_keys集合中
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    print("\tUnexpected_keys:", list(unexpected_keys))
    print("\tMissing_keys:", list(missing_keys))
    model.load_state_dict(matched_state_dict, strict=False)#将加载的权重加载到模型中

    evaluate(model, tokenizer, encode_test_dataset, col_name, device, config)


if __name__ == '__main__':
    # 设置设备和具体用哪个GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"#*1改成0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
