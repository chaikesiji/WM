import torch
import torch.nn.functional as F


def msg_loss(sentence_dg, sentence_part_dg, margin=1, use_sig=True, use_contra=True, beta=2):
    # use_sig是判断是否要将值限制到0和1之间
    if use_sig:
        sentence_dg_sig = torch.sigmoid(sentence_dg)
        sentence_part_dg_sig = torch.sigmoid(sentence_part_dg)
    else:
        sentence_dg_sig = sentence_dg
        sentence_part_dg_sig = sentence_part_dg
    # margin是为了设置正负样本之间的距离, beta是为了设置更加偏向于生成正样本之间生成相同VPK还是正负样本之间生成不同VPK
    loss = feature_loss(sentence_dg_sig, sentence_part_dg_sig, margin=margin, use_contra=use_contra, beta=beta)
    return loss


def feature_loss(sentence_feat, sentence_part_feat, margin=2, use_cos_sim=True, use_contra=False, beta=2):
    bsz = sentence_feat.shape[0]
    device = sentence_feat.device
    # use_contra是判断是使用对比损失还是triplet loss
    if use_contra:
        sentence_feat_norm = sentence_feat
        sentence_part_feat_norm = sentence_part_feat
    else:
        sentence_feat_norm = F.normalize(sentence_feat, p=2, dim=-1)
        sentence_part_feat_norm = F.normalize(sentence_part_feat, p=2, dim=-1)
    label_ones = torch.ones(bsz, bsz).to(device)
    # 使用余弦距离还是欧氏距离
    if use_cos_sim:
        label = label_ones - torch.diag(torch.diag(label_ones))
        similarity_AA_matrix = F.cosine_similarity(sentence_feat_norm.unsqueeze(1), sentence_feat_norm.unsqueeze(0), dim=-1)
        similarity_AB_matrix = F.cosine_similarity(sentence_feat_norm.unsqueeze(1), sentence_part_feat_norm.unsqueeze(0), dim=-1)
        similarity_AA_matrix_mask = similarity_AA_matrix * label
        similarity_AB_matrix_mask = similarity_AB_matrix * label
    else:
        label = torch.diag(torch.diag(label_ones)) * 10000000000
        similarity_AA_matrix = torch.cdist(sentence_feat_norm, sentence_feat_norm, p=2)
        similarity_AB_matrix = torch.cdist(sentence_feat_norm, sentence_part_feat_norm, p=2)
        similarity_AA_matrix_mask = similarity_AA_matrix + label
        similarity_AB_matrix_mask = similarity_AB_matrix + label

    loss = torch.tensor([0.], device='cuda:0')
    # 遍历正负样本得到损失
    for i in range(bsz):
        pos_score = similarity_AB_matrix[i, i]
        if use_cos_sim:
            pos_distance = 1 - pos_score
            AA_neg_scores = torch.max(similarity_AA_matrix_mask[i], dim=-1)[0]
            AB_neg_scores = torch.max(similarity_AB_matrix_mask[i], dim=-1)[0]
            if AA_neg_scores > AB_neg_scores:
                neg_scores = AA_neg_scores
            else:
                neg_scores = AB_neg_scores
            neg_distance = 1 - neg_scores
        else:
            pos_distance = pos_score
            AA_neg_scores = torch.min(similarity_AA_matrix_mask[i], dim=-1)[0]
            AB_neg_scores = torch.min(similarity_AB_matrix_mask[i], dim=-1)[0]
            if AA_neg_scores < AB_neg_scores:
                neg_scores = AA_neg_scores
            else:
                neg_scores = AB_neg_scores
            neg_distance = neg_scores
        if use_contra:
            loss += 0.5 * torch.pow(pos_distance, 2) + beta * 0.5 * torch.pow(torch.clamp(margin - neg_distance, min=0.), 2)
        else:
            loss += torch.clamp(pos_distance - neg_distance + margin, min=0.)
    loss /= bsz
    return loss
