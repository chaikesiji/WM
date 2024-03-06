import transformers
import torch
from transformers import BertModel
import torch.nn.functional as F
import torch.nn as nn
from utils_co_attention import Self_Attention_Encoder

path='/home/xy/DB-WM/tokenizer_config'#*加
class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg["hidden_dim"]
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg["dropout_rate"]
        self.num_head = self.cfg["num_head"]
        self.inner_dim = self.cfg["inner_dim"]
        self.k_dim = self.cfg["k_dim"]
        self.v_dim = self.cfg["v_dim"]

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                                             self.v_dim, self.hidden_dim, self.inner_dim,
                                                             self.dropout_rate)

    def forward(self, x, textual_feats=None, mask=None):
        assert mask is not None
        outp = self.SA_transformer_encoder(x, mask)

        return outp


class VPKCOL(torch.nn.Module):
    def __init__(self, config):
        super(VPKCOL, self).__init__()
        self.bert = BertModel.from_pretrained(
            path#*
           # 'bert-base-uncased'
            )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 384), nn.LayerNorm(384), nn.LeakyReLU(),
            nn.Linear(384, 192), nn.LayerNorm(192), nn.LeakyReLU(),
            nn.Linear(192, 32)
        ])
        self.max_seq_length = 80
        self.SA_layer = Self_Attention_Cell(config)

    def forward(self, sentence, sentence_part, col_name, select_col, is_pooling=False):
        tensor_output = torch.tensor([], device='cuda:0')
        part_tensor_output = torch.tensor([], device='cuda:0')
        tensor_output_dict = {}
        part_tensor_output_dict = {}
        for col in col_name:
            last_output = self.bert(**sentence[col]).last_hidden_state
            last_output_pooling = self.avg(last_output)
            tensor_output = torch.cat([tensor_output, last_output_pooling], dim=1)
            # tensor_output_dict[col] = last_output_pooling
            if col != select_col:
                # part_last_output = self.bert(**sentence_part[col]).last_hidden_state
                # part_last_output_pooling = self.avg(part_last_output)
                part_tensor_output = torch.cat([part_tensor_output, last_output_pooling], dim=1)
            # part_tensor_output_dict[col] = part_last_output_pooling
        data_mask = torch.ones([tensor_output.shape[0], tensor_output.shape[1]], device=tensor_output.device)
        part_data_mask = torch.ones([part_tensor_output.shape[0], part_tensor_output.shape[1]], device=tensor_output.device)
        tensor_output_sa = self.SA_layer(tensor_output, mask=data_mask)
        part_tensor_output_sa = self.SA_layer(part_tensor_output, mask=part_data_mask)
        pooled_output1 = self.avg(tensor_output_sa).squeeze(1)
        pooled_output2 = self.avg(part_tensor_output_sa).squeeze(1)
        logits1 = self.classifier(pooled_output1)
        logits2 = self.classifier(pooled_output2)
        return pooled_output1, pooled_output2, logits1, logits2

    def forward_(self, sentence, col_name):
        tensor_output = torch.tensor([], device=sentence[col_name[0]]['input_ids'].device)
        for col in col_name:
            last_output = self.bert(**sentence[col]).last_hidden_state
            last_output_pooling = self.avg(last_output)
            tensor_output = torch.cat([tensor_output, last_output_pooling], dim=1)
        data_mask = torch.ones([tensor_output.shape[0], tensor_output.shape[1]], device=tensor_output.device)
        tensor_output_sa = self.SA_layer(tensor_output, mask=data_mask)
        pooled_output1 = self.avg(tensor_output_sa).squeeze(1)
        logits1 = self.classifier(pooled_output1)
        return pooled_output1, logits1

    def avg(self, input_tensor):
        m = nn.AdaptiveAvgPool1d(1)
        return m(input_tensor.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
