import os
os.environ['CURL_CA_BUNDLE'] = ''#*新加
import copy
import json
from transformers import AutoTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datasets import Dataset

import numpy as np
import random
from model_primarykey_col import VPKCOL
from utils_pk_loss_function import feature_loss, msg_loss
from utils_motifs_pk import metric_acc
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

#https://www.cnblogs.com/happyNLP/p/16880691.html**
#设置种子，生成随机数**
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config_file(config_path):
    assert 'json' in config_path, "file type is not json"
    return json.load(open(config_path, 'r'))


def load_data(config):
    df = pd.read_csv(config['train_file']).dropna()
    df = pd.DataFrame(df)
    # 训练文件取前30000行,验证集取最后9000行
    train_df = df.head(30000)
    val_df = df.tail(9000)
    train_dataset = Dataset.from_pandas(train_df, split="train")
    val_dataset = Dataset.from_pandas(val_df, split="val")
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenzier_cache_path'],
        add_prefix_space=True,
    )
    tqdm.write("Dateset loading complete!")
    return train_dataset, val_dataset, tokenizer


def preprocess_char_data(examples):
    tmp_dataset = {}
    for key, values in examples.items():
        tmp_dataset[key] = values
    return tmp_dataset


def train(model, tokenizer, config, encoded_dataset, val_dataset, col_name, log_txt, val_log_txt, device):
    tb_writer = SummaryWriter(log_dir=config['tensorboard_path'])#将训练过程中的信息写入TB
    bsz = config["batch_size"]
    params_model = model.parameters()#获取模型的权重和偏差等
    # 优化器采用AdamW
    #学习率决定模型参数更新的步长，较大的学习率导致模型训练不稳定，较小的学习率导致训练速度缓慢
    #权重衰减，防止模型过拟合
    optimizer = torch.optim.AdamW(params_model, lr=0.000005, weight_decay=0.00001)#初始化优化器对象，用于更新模型参数
    # 学习率策略
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    #学习率调度器，验证损失不再改善时减小学习率
    #min：关注验证损失最小化   验证损失停止下降后，1个epoch再降低学习率 新的学习率是原来的一半
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    total_pos_acc, total_neg_acc, total_feat_loss, total_dg_loss, total_model_loss = 0, 0, 0, 0, 0
    update_num = 0#训练轮数
    best_score = 0.
    data_dict_ori = {value: [] for value in col_name}
    # 开始训练
    for i in tqdm(range(config['max_iter'])):
        for bsz_num in range(0, (encoded_dataset.num_rows // bsz)):#*遍历每一批数据 row:行 column：列
            data_tensor_dict = {}
            data_dict = copy.deepcopy(data_dict_ori)#data_dict和data_dict_ori是两个完全独立的字典，对data_dict的修改不会影响data_dict_ori。
            part_data_dict = copy.deepcopy(data_dict_ori)
            part_data_tensor_dict = {}
            row_range = [bsz_num * bsz, (bsz_num + 1) * bsz]
            # 随机删除一列,select_col是选取的列,用字典保存
            select_col = random.choice(col_name)
            part_data_dict.pop(select_col) #随机删除一列数据
            for row in range(row_range[0], row_range[1]):
                sentence = ''
                for key, values in encoded_dataset[row].items():
                    data_dict[key].append(str(values) + ' ')
                    sentence += str(values) + ' '
                    if key != select_col:#检查当前的键（列名）是否不等于之前随机选择的列名select_col
                        part_data_dict[key].append(str(values) + ' ')

            # 对删除前后的数据进行编码并放到对应device上, max_length是编码最长长度,padding是为了batch中长度一致
            # return_tensors是为了得到tensor类型数据
            for col in col_name:
                data_tensor = tokenizer(data_dict[col], max_length=512, padding=True, truncation=True,
                                        return_tensors='pt')
                if device != 'cpu':
                    data_tensor_dict[col] = {k: v.cuda() for k, v in data_tensor.items()}
                if col != select_col:#只关心除select_col之外的其他列
                    part_data_tensor = tokenizer(part_data_dict[col], max_length=512, padding=True, truncation=True,
                                                 return_tensors='pt')#这个参数可以确保返回的张量格式是PyTorch兼容的
                    if device != 'cpu':#v.cuda(): 这会将张量v从当前设备（可能是CPU）移动到CUDA设备（通常是GPU）。
                        part_data_tensor_dict[col] = {k: v.cuda() for k, v in part_data_tensor.items()}

            # 开始训练
            model.train()
            # 清空梯度   这是在进行反向传播之前必须要做的，因为PyTorch会累积梯度
            optimizer.zero_grad()
            # 进行计算得到元组特征,元组VPK,p表示残缺,部分的意思(即受到删除攻击后的元组生成的特征和VPK)
            sentence_fe, sentence_p_fe, sentence_dg, sentence_p_dg = model(data_tensor_dict, part_data_tensor_dict,
                                                                           col_name, select_col, is_pooling=True)#使用模型对数据进行前向传播
            # 在特征级别和VPK级别都采用了损失
            feat_loss = feature_loss(sentence_fe, sentence_p_fe, margin=1.2, use_cos_sim=False)
            dg_loss = msg_loss(sentence_dg, sentence_p_dg, margin=1.8, use_sig=True, use_contra=True, beta=3)
            # 模型总损失,config['feat_loss']是这个loss占的权重,一个是特征级别一个VPK级别,意义不同
            model_loss = feat_loss * config['feat_loss'] + dg_loss * config['dg_loss']

            # 梯度回传并更新权重
            model_loss.backward()#自动计算损失函数关于模型参数的梯度，并将这些梯度存储在参数的.grad属性
            optimizer.step()

            # 计算同一个batch中计算正负样本之间的准确率
            # 例如10个元组x,删除一个属性后得到x',那么x和x'的第i行互为正样本要输出相同VPK,但x和x'的第j(i != j)行相对于x和x'的i都为负样本
            # 应该输出不同的VPK
            pos_acc, neg_acc = metric_acc(sentence_dg, sentence_p_dg)

            # 下面是可视化的一些结果,item()是为了获取数值
            total_pos_acc += (pos_acc.item())
            total_neg_acc += (neg_acc.item())
            total_feat_loss += (feat_loss.item()) * config['feat_loss']
            total_dg_loss += (dg_loss.item()) * config['dg_loss']
            total_model_loss += (model_loss.item())

            # 当迭代log_interval时则打印此使的损失和准确率等,tb_writer.add_scalar是增加了tensorboard模块来用图形可视化训练结果
            # 命令行为tensorborad --logdir='#' --port='#',具体命令可官网查询
            if (bsz_num + 1) % config['log_interval'] == 0 and bsz_num > 0:
                cur_pos_acc = total_pos_acc / config['log_interval']
                cur_neg_acc = total_neg_acc / config['log_interval']
                cur_feat_loss = total_feat_loss / config['log_interval']
                cur_dg_loss = total_dg_loss / config['log_interval']
                cur_model_loss = total_model_loss / config['log_interval']

                tb_writer.add_scalar("cur_disc_loss", cur_pos_acc, update_num)
                tb_writer.add_scalar("errG_loss", cur_neg_acc, update_num)
                tb_writer.add_scalar("msg_loss", cur_feat_loss, update_num)
                tb_writer.add_scalar("en_msg_loss", cur_dg_loss, update_num)
                tb_writer.add_scalar("en_zero_loss", cur_model_loss, update_num)
                update_num += 1

                run_info = '| epoch {:3d} | {:3d}/{:3d} batches | lr {:05.8f}'.format(i, bsz_num + 1,
                                                                                      encoded_dataset.num_rows // bsz,
                                                                                      optimizer.param_groups[0]['lr'])
                acc_info = 'pos_msg_acc {:5.2f}| neg_msg_acc {:5.2f} |'.format(cur_pos_acc, cur_neg_acc)
                loss_info = 'feat_loss {:5.5f} | dg_loss {:5.5f} | model_loss {:5.5f}'.format(cur_feat_loss,
                                                                                              cur_dg_loss,
                                                                                              cur_model_loss)
                log_info = run_info + acc_info + loss_info
                tqdm.write('*' * 60 + run_info + '*' * 60)
                tqdm.write(acc_info + loss_info)
                log_txt.write(log_info + '\n')
                total_pos_acc, total_neg_acc, total_feat_loss, total_dg_loss, total_model_loss = 0, 0, 0, 0, 0

        # 当运行vali_epoch后则进行验证,验证时不进行梯度传递,仅遍历计算一次完整验证数据集得到最后的结果
        if (i + 1) % config['valid_epoch'] == 0 and i >= (config['valid_epoch'] - 1):
            eval_fe_loss, eval_dg_loss, eval_score = evaluate(model, tokenizer, config, val_dataset, col_name, val_log_txt)
            if eval_score > best_score:
                torch.save(model.state_dict(), config['root_path'] + "model_best.pth")
                best_score = cur_neg_acc + cur_neg_acc
            scheduler.step(eval_fe_loss)
    log_txt.close()
    val_log_txt.close()


def evaluate(model, tokenizer, config, val_dataset, col_name, val_log_txt):
    tb_writer = SummaryWriter(log_dir=config['tensorboard_path'])
    bsz = config["batch_size"]

    total_pos_acc, total_neg_acc, total_feat_loss, total_dg_loss, total_model_loss = 0, 0, 0, 0, 0
    data_dict_ori = {value: [] for value in col_name}
    update_num = 0
    cal_num = 0
    with torch.no_grad():
        for bsz_num in range(0, (val_dataset.num_rows // bsz)):
            data_tensor_dict = {}
            data_dict = copy.deepcopy(data_dict_ori)
            part_data_dict = copy.deepcopy(data_dict_ori)
            part_data_tensor_dict = {}
            row_range = [bsz_num * bsz, (bsz_num + 1) * bsz]
            select_col = random.choice(col_name)
            part_data_dict.pop(select_col)
            for row in range(row_range[0], row_range[1]):
                sentence = ''
                for key, values in val_dataset[row].items():
                    data_dict[key].append(str(values) + ' ')
                    sentence += str(values) + ' '
                    if key != select_col:
                        part_data_dict[key].append(str(values) + ' ')
            
            
            sentence_fe, sentence_p_fe, sentence_dg, sentence_p_dg = model(data_tensor_dict, part_data_tensor_dict,
                                                                           col_name, select_col, is_pooling=True)
            pos_acc, neg_acc = metric_acc(sentence_dg, sentence_p_dg)
            feat_loss = feature_loss(sentence_fe, sentence_p_fe, margin=1, use_cos_sim=False)
            dg_loss = msg_loss(sentence_dg, sentence_p_dg)

            total_feat_loss += (feat_loss.item()) * config['feat_loss']
            total_dg_loss += (dg_loss.item()) * config['dg_loss']
            total_pos_acc += (pos_acc.item())
            total_neg_acc += (neg_acc.item())
            cal_num += 1

        cur_pos_acc = total_pos_acc / cal_num
        cur_neg_acc = total_neg_acc / cal_num
        cur_feat_loss = total_feat_loss / cal_num
        cur_dg_loss = total_dg_loss / cal_num
        cur_model_loss = total_model_loss / cal_num

        tb_writer.add_scalar("eval_pos_acc", cur_pos_acc, update_num)
        tb_writer.add_scalar("eval_neg_acc", cur_neg_acc, update_num)
        update_num += 1

        acc_info = 'pos_msg_acc {:5.2f}| neg_msg_acc {:5.2f} |'.format(cur_pos_acc, cur_neg_acc)
        loss_info = 'feat_loss {:5.5f} | dg_loss {:5.5f} | model_loss {:5.5f}'.format(cur_feat_loss,
                                                                                      cur_dg_loss,
                                                                                      cur_model_loss)
        log_info = acc_info + loss_info
        tqdm.write('*' * 60 + "EVAL" + '*' * 60)
        tqdm.write(acc_info + loss_info)
        tqdm.write('*' * 60 + "EVAL" + '*' * 60)
        val_log_txt.write(log_info + '\n')
        return cur_feat_loss, cur_dg_loss, cur_pos_acc + cur_neg_acc


def main(device):
    config = load_config_file('/home/xy/DB-WM/csv_word_modify/data/config_file.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"#2改为0*
    setup_seed(config["random_seed"])
    train_dataset, val_dataset, tokenizer = load_data(config)
    last_two_data = {}
    encode_train_dataset = train_dataset.map(preprocess_char_data,
                                             batched=True,
                                             remove_columns=train_dataset.column_names,
                                             load_from_cache_file=False
                                             )
    encode_val_dataset = val_dataset.map(preprocess_char_data,
                                         batched=True,
                                         remove_columns=val_dataset.column_names,
                                         load_from_cache_file=False
                                         )
    col_name = train_dataset.column_names

    model = VPKCOL(config).to(device)
    train_log = open(config['train_log_path'], 'w')
    val_log = open(config['train_log_path'], 'w')
    if config['load_checkpoint']:
        # 加载预训练权重
        checkpoint = torch.load(config['checkpoint_path'])
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

    # 是否进行xavier初始化,是否回传梯度
    # for name, param in model.named_parameters():
    #     if (not name.startswith('bert')) and param.dim() > 1:
    #         nn.init.xavier_uniform_(param)
    #         tqdm.write(name)
    #         if name.startswith('classifier') and param.dim() > 1:
    #             param.requires_grad = False

    # 冻结部分层
    # for name, param in model.named_parameters():
    #     if (not name.startswith('classifier')) and param.dim() > 1:
    #         param.requires_grad = False
    #         tqdm.write(f'frozen layer: {name}')
    model.load_state_dict(matched_state_dict, strict=False)
    train(model, tokenizer, config, encode_train_dataset, encode_val_dataset, col_name, train_log, val_log, device)
    # 保存权重, 注意修改Load和Save的权重路径，容易覆盖之前训练好的权重
    torch.save(model.state_dict(), config['root_path'] + "model_final.pth")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
