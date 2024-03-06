import torch
import copy


def extract_digital(change_char):
    if change_char in ['0', '2', '4', '6', '8']:
        return 0
    else:
        return 1


def extract(change_char, vis, labels):
    # 根据选择的字符是ASCII码的相对大小来判断0和1
    change_char_prefix = '##' + change_char
    if change_char_prefix in labels:
        char_idx = labels.index(change_char_prefix)
        char_vis = vis[char_idx]
        candidate_idx = torch.nonzero(char_vis)[:, 0].tolist()
        candidate_char = list(map(lambda x: labels[x][2:], candidate_idx))
        candidate_char_ord = list(map(ord, candidate_char))
        if change_char == chr(min(candidate_char_ord)):
            return 1
        else:
            return 0
    else:
        return 0


def modify(change_char, wm, vis, labels):
    # 找到对应的字符,然后根据ASCII码和水印01来确定是否修改
    change_char_prefix = '##' + change_char
    if change_char_prefix in labels:
        char_idx = labels.index(change_char_prefix)
        char_vis = vis[char_idx]
        candidate_idx = torch.nonzero(char_vis)[:, 0].tolist()
        candidate_char = list(map(lambda x: labels[x][2:], candidate_idx))
        candidate_char_ord = list(map(ord, candidate_char))
        if wm:
            return chr(min(candidate_char_ord))
        else:
            return chr(max(candidate_char_ord))
    else:
        print(change_char)
        return change_char


def modify_digital(change_char, wm):
    if change_char in ['0', '1']:
        if wm == 0:
            return '0'
        else:
            return '1'
    elif change_char in ['2', '3']:
        if wm == 0:
            return '2'
        else:
            return '3'
    elif change_char in ['4', '5']:
        if wm == 0:
            return '4'
        else:
            return '5'
    elif change_char in ['6', '7']:
        if wm == 0:
            return '6'
        else:
            return '7'
    elif change_char in ['8', '9']:
        if wm == 0:
            return '8'
        else:
            return '9'


def metric_acc(sentence_dg, sentence_p_dg):
    '''
    计算正负样本之间的准确率(必须32位完全正确才算正确)
    正样本准确率仅仅在对应行之间计算,所以如果batch_size=10,则10次比较求平均
    正负样本的准确率,先计算正负样本之间有多少个一样的,这其实是错误概率,再用1-错误概率得到最终正确的准确率
    需要注意的是如果batch_size=100,那么对x中第i行来说,总共有18个负样本
    '''
    # 先限制范围到0和1之间,然后四舍五入
    sentence_dg = copy.deepcopy(sentence_dg.detach())
    sentence_p_dg = copy.deepcopy(sentence_p_dg.detach())
    bsz, num = sentence_dg.shape[0], sentence_dg.shape[1]
    device = sentence_dg.device
    sentence_dg_sig = torch.sigmoid(sentence_dg)
    sentence_dg_round = torch.round(sentence_dg_sig)
    sentence_p_dg_sig = torch.sigmoid(sentence_p_dg)
    sentence_p_dg_round = torch.round(sentence_p_dg_sig)
    pos_acc_bool = sentence_dg_round.eq(sentence_p_dg_round)
    pos_acc = (torch.all(pos_acc_bool, dim=1).sum()) / bsz
    neg_acc = torch.tensor([0], dtype=torch.float32, device=device)
    # 计算准确率,为了效率,这里多采样矩阵之间的运算
    for i in range(bsz):
        self_expand = sentence_dg_round[i].expand(bsz - 1, -1)
        other_expand = sentence_p_dg_round[i].expand(bsz - 1, -1)
        self_labels = torch.ones([bsz], device=device, dtype=torch.bool)
        self_labels[i] = False
        neg_self_matrix = sentence_dg_round[self_labels]
        neg_other_matrix = sentence_p_dg_round[self_labels]
        neg_self_acc_bool = self_expand.eq(neg_self_matrix)
        neg_other_acc_bool = other_expand.eq(neg_other_matrix)
        neg_self_acc = (torch.all(neg_self_acc_bool, dim=1).sum()) / (bsz - 1)
        neg_other_acc = (torch.all(neg_other_acc_bool, dim=1).sum()) / (bsz - 1)
        neg_avg_acc = 2 - neg_self_acc - neg_other_acc
        neg_avg_acc = 0.5 * neg_avg_acc
        neg_acc += neg_avg_acc
    neg_acc /= bsz
    return pos_acc, neg_acc.squeeze(0)


# def learing_rate_scheduler(step_num):
#     d_model = 768
#     warm_up = 4000
#     lr_scheduler = np.power(d_model, -0.8) * min(np.power(step_num, -0.5), step_num * np.power(warm_up, -1.5))
#     return lr_scheduler


