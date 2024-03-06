'''
提取水印的代码,仅需要确定文件和权重即可
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
from utils_motifs_pk import extract
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config_file(config_path):
    assert 'json' in config_path, "file type is not json"
    return json.load(open(config_path, 'r'))


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
    return suffix_id_list, suffix_list#返回后缀词汇的ID列表和词汇列表


def get_vocab_visual_tensor(vocab_ids, visual_path):
    suffix_tensor_map = torch.load(visual_path)
    result = []
    for v_id in vocab_ids:
        result.append(suffix_tensor_map[v_id])
    result = torch.stack(result).cuda()
    return result#返回一个堆叠后的张量，其中包含了与vocab_ids中ID对应的视觉特征


def load_data(config):
    df = pd.read_csv(config['output_path']).dropna()
    df = pd.DataFrame(df)
    train_dataset = Dataset.from_pandas(df, split="train")
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenzier_cache_path'],
        add_prefix_space=True,
    )
    tqdm.write("Dateset loading complete!")
    return train_dataset, tokenizer


def preprocess_char_data(examples):
    tmp_dataset = {}
    for key, values in examples.items():
        tmp_dataset[key] = values
    return tmp_dataset


def evaluate(model, tokenizer, val_dataset, col_name, device, config):
    vis = torch.load(config["visual_tensor_path"])
    labels = np.load(config["suffix_path"]).tolist()
    # 一次处理bsz个元组
    bsz = config['val_bsz']
    all_words_sk = torch.zeros([val_dataset.num_rows, 32], dtype=torch.float32, device=device)
    # 这里是加载之前的水印,为了计算准确率,以narray的类型进行存储
    wm_real = np.load(config['wm_path']).tolist()
    wm_len = len(wm_real)
    choose_col_one_c = torch.load(config['attribute_vpk_path'])
    single_num = (choose_col_one_c.shape[0] // wm_len)
    split_num = [i * single_num for i in range(1, wm_len)]
    if split_num[-1] != val_dataset.num_rows:
        split_num.append(val_dataset.num_rows)
    j = 0
    with torch.no_grad():
        # 划分数据
        bsz_num = val_dataset.num_rows // bsz
        if bsz_num * bsz == val_dataset.num_rows:
            range_list = range(0, (val_dataset.num_rows // bsz))
        else:
            range_list = range(0, (val_dataset.num_rows // bsz) + 1)
        for bsz_num in tqdm(range_list):
            data_tensor_dict = {}
            if bsz_num != range_list[-1]:
                start_index = bsz_num * bsz
                end_index = (bsz_num + 1) * bsz
            else:
                start_index = end_index
                end_index = val_dataset.num_rows
            # 对提取出来的数据进行编码
            encode_dict = {key: np.array(value).astype(str).tolist() for key, value in val_dataset[start_index:end_index].items()}
            for col in col_name:
                data_tensor = tokenizer(encode_dict[col], max_length=512, padding=True, truncation=True,
                                        return_tensors='pt')
                if device != 'cpu':
                    data_tensor_dict[col] = {k: v.cuda() for k, v in data_tensor.items()}

            # 与嵌入做相同操作,给每个元组生成VPK
            model.eval()
            sentence_fe, sentence_dg = model.forward_(data_tensor_dict, col_name)
            sentence_dg_copy = copy.deepcopy(sentence_dg.detach())
            sentence_dg_sig = torch.sigmoid(sentence_dg_copy)
            sentence_dg_round = torch.round(sentence_dg_sig)
            all_words_sk[start_index:end_index] = sentence_dg_round

        # VPK从32位2进制->10进制
        decimal_range = torch.arange(all_words_sk.shape[1], device='cuda:0').expand(all_words_sk.shape[0], -1)
        decimal_value = 2 ** decimal_range
        all_words_sk_decimal = torch.sum(all_words_sk * decimal_value, dtype=torch.long, dim=1)
        torch.save(all_words_sk_decimal, config['tuple_recover_vpk_path'])
        # 计算每个单元格(属性)的VPK并判断是否要嵌入
        col_idx = tokenizer(col_name, add_special_tokens=False, return_tensors='pt', padding=True, max_length=512,
                            truncation=True).data['input_ids']
        col_emb = torch.sum(col_idx, dim=-1).to(device)
        all_sk_score = all_words_sk_decimal.unsqueeze(1) + col_emb.unsqueeze(0)
        remainder = 4
        condition = (all_sk_score % 9 == remainder)

        choose_col = torch.where(condition)
        choose_col_one = torch.where(condition, torch.tensor(1), torch.tensor(0))

        # 这里是为了计算当嵌入单元格嵌入水印后,提取时能够判断出这里嵌入水印的准确率,实际使用时可以注释(这句注释后九行,到tqdm.write)

        choose_col_sum = choose_col_one + choose_col_one_c
        same_number = torch.where(choose_col_sum == 2)
        if (torch.where(choose_col_one_c == 1)[0].shape[0]):
            same_rate = same_number[0].shape[0] / (torch.where(choose_col_one_c == 1)[0].shape[0])
        else:
            tqdm.write(f'水印列完全被删除')
            same_rate = 0
        tqdm.write(f'same_rate: {same_rate}')

        csv_data = [val_dataset.column_names]
        extract_wm_list = []
        row_split = []
        count = 0
        wm = []
        # 提取水印
        for row in tqdm(range(val_dataset.num_rows)):
            if row >= split_num[j]:
                j += 1
                row_split.append(count)
                if len(extract_wm_list) != 0:
                    # 多次提取结果取平均并四舍五入
                    wm.append(round((sum(extract_wm_list) / len(extract_wm_list))))
                else:
                    # 为能够有效提取则为-1
                    wm.append(-1)
                extract_wm_list = []
            if row in choose_col[0]:
                tensor_index = torch.where(choose_col[0] == row)[0]
                col = choose_col[1][tensor_index]
                row_ori_data = list(val_dataset[row].values())
                for idx in col:
                    row_score = all_sk_score[row][idx]
                    wait_change_data = str(row_ori_data[idx])
                    data_len = len(wait_change_data)
                    change_idx = row_score.item() % data_len
                    if wait_change_data[change_idx] == ' ':
                        continue
                    wm_row = extract(wait_change_data[change_idx], vis, labels)
                    # 如果想仅在数字之间转换可以取消这里的注释(该注释后三行),规则可以自己制定
                    # if wait_change_data[change_idx] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    #     change_idx += 1
                    # wm_row = extract_digital(wait_change_data[change_idx])
                    count += 1
                    extract_wm_list.append(wm_row)
                row_data = row_ori_data
            else:
                row_data = list(val_dataset[row].values())
            csv_data.append(row_data)

        # 由于写法的问题,这里需要再做一次将多次结果求平均并四舍五入
        if len(extract_wm_list) != 0:
            wm.append(round((sum(extract_wm_list) / len(extract_wm_list))))
        else:
            wm.append(-1)
        # 这里是计算准确率,非必须
        wm_tensor = torch.tensor(wm_real)
        wm_ex_tensor = torch.tensor(wm)
        same = len(torch.where(wm_tensor == wm_ex_tensor)[0])
        tqdm.write(f"准确率:{same/len(wm_real)}")
        print(f"水印信息为：{wm}")
        print('Finished')
        return same/len(wm_real)


def main(device):
    config = load_config_file('/home/xy/DB-WM/csv_word_modify/data/config_file.json')
    # 加载数据集
    train_dataset, tokenizer = load_data(config)
    encode_train_dataset = train_dataset.map(preprocess_char_data,
                                             batched=True,
                                             remove_columns=train_dataset.column_names,
                                             load_from_cache_file=False
                                             )
    col_name = train_dataset.column_names

    model = VPKCOL(config).to(device)

    checkpoint_path = config['checkpoint_path']
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get('model_state', checkpoint)
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    print("\tUnexpected_keys:", list(unexpected_keys))
    print("\tMissing_keys:", list(missing_keys))
    model.load_state_dict(matched_state_dict, strict=False)

    cur_acc = evaluate(model, tokenizer, encode_train_dataset, col_name, device, config)
    tqdm.write(f"准确率:{cur_acc}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
