"""这个model的构图方法是每个visit构建一个图，然后把所有patient所有visit聚合为一个batch，一个batch_data聚合为一个batch_graph"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.grpah_netV1 import HeteroGNN, process_visit_graphs, process_patient_graphs
from HiNet3D.models.grpah_netV1 import HeteroGNN, process_visit_graphs, process_patient_graphs

def extract_and_transpose(tensor_list):
    """
    提取每个张量的最后一个序列并转置
    
    :param tensor_list: list of tensors
    :return: 处理后的张量列表
    """
    processed_tensors = []
    for tensor in tensor_list:
        last_seq = tensor[:, -1:, :]  # 提取最后一个序列
        transposed_seq = last_seq.transpose(0, 1)  # 转置
        processed_tensors.append(transposed_seq)
    return processed_tensors


def split_tensor(tensor, lengths, max_len):
    """
    将聚合的张量拆分为原始形状
    :param tensor: 聚合的张量
    :param lengths: 每个batch的长度
    :param max_len: 最大长度
    :return: 拆分后的张量列表
    """
    index = 0
    outputs = []

    for length in lengths:
        output_tensor = tensor[index:index + length]
        outputs.append(output_tensor)
        index += length

    outputs = [x[:, :max_len, :] for x in outputs]
    return outputs


def aggregate_tensors(tensors, device):
    """
    将多个张量聚合到最大长度
    :param tensors: list of tensors
    :return: 聚合后的张量, 每个batch的长度
    """
    max_len = max([x.size(1) for x in tensors])
    padded_inputs = []
    lengths = []

    for x in tensors:
        lengths.append(x.size(0))
        padding = torch.zeros(x.size(0), max_len - x.size(1), x.size(2)).to(device)
        padded_x = torch.cat((x, padding), dim=1)
        padded_inputs.append(padded_x)

    aggregated_tensor = torch.cat(padded_inputs, dim=0)
    return aggregated_tensor, lengths


def concatenate_patient_embeddings(patient_emb_dict):
    """
    将 patient_emb_dict 中的所有张量在最后一个维度上拼接起来。

    参数:
    patient_emb_dict (dict of torch.Tensor): 包含每个特征的患者嵌入字典

    返回:
    torch.Tensor: 拼接后的张量
    """
    # 提取所有张量
    emb_list = list(patient_emb_dict.values())
    # 在最后一个维度上拼接
    patient_emb = torch.cat(emb_list, dim=-1)
    return patient_emb


def pad_tensors(tensors):
    """
    将不同尺寸的tensor填充成相同尺寸的大tensor。

    参数:
    tensors (list of torch.Tensor): 输入的多个张量列表，每个张量的形状为 (visit, monitor, dim)

    返回:
    torch.Tensor: 填充后的大张量，形状为 (batch, max_visits, max_monitors, dim)
    """
    # 找到各个维度的最大值
    max_visits = max(t.size(0) for t in tensors)
    max_monitors = max(t.size(1) for t in tensors)
    dim = tensors[0].size(2)

    # 初始化一个大的tensor，用0填充
    batch_size = len(tensors)
    padded_tensor = torch.zeros((batch_size, max_visits, max_monitors, dim))

    # 将各个tensor填充到大tensor中
    for i, tensor in enumerate(tensors):
        v, m, d = tensor.size()
        padded_tensor[i, :v, :m, :] = tensor

    return padded_tensor


def expand_tensors(visit_emb_list, target_shape):
    """
    将visit_emb_list中的每个张量扩展到目标形状。

    参数:
    visit_emb_list (list of torch.Tensor): 输入的张量列表，每个张量的形状为 (16, 22, 128)
    target_shape (tuple): 目标张量形状 (16, 22, 77, 128)

    返回:
    list of torch.Tensor: 扩展后的张量列表，每个张量的形状为 (16, 22, 77, 128)
    """
    expanded_list = []
    for tensor in visit_emb_list:
        # 在第三个维度上增加一个维度 (16, 22, 1, 128)
        tensor_expanded = tensor.unsqueeze(2)
        # 将第三个维度扩展到目标形状 (16, 22, 77, 128)
        tensor_expanded = tensor_expanded.expand(-1, -1, target_shape[2], -1)
        expanded_list.append(tensor_expanded)
    return expanded_list


class HiNet3D(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7
    ):
        super(HiNet3D, self).__init__()
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.monitor_event_token = Tokenizers_monitor_event

        self.feature_visit_event_keys = Tokenizers_visit_event.keys()
        self.feature_monitor_event_keys = Tokenizers_monitor_event.keys()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.device = device

        self.embeddings = nn.ModuleDict()
        # 为每种event（包含monitor和visit）添加一种嵌入
        for feature_key in self.feature_visit_event_keys:
            tokenizer = self.visit_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        for feature_key in self.feature_monitor_event_keys:
            tokenizer = self.monitor_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        self.visit_gru = nn.ModuleDict()
        # 为每种visit_event添加一种gru
        for feature_key in self.feature_visit_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in self.feature_monitor_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in ['weight', 'age']:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.monitor_gru = nn.ModuleDict()
        # 为每种monitor_event添加一种gru
        for feature_key in self.feature_monitor_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in self.feature_visit_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.patient_info_fc = nn.ModuleDict()
        # 为每种病人信息添加一个全连接层
        for feature_key in ['weight', 'age']:
            self.patient_info_fc[feature_key] = nn.Linear(1, self.embedding_dim)

        item_num = int(len(Tokenizers_monitor_event.keys()) / 2) + 3 + 2
        self.fc_visit = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(5 * self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )

        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(item_num * self.embedding_dim, output_size)
        )

        self.gnn = nn.ModuleDict()
        num_nodes_dict = {}
        for feature_key in self.feature_visit_event_keys:
            num_nodes_dict[feature_key] = self.visit_event_token[feature_key].get_vocabulary_size()

        # 给visit-monitoring交互的图构图
        num_nodes_dict['lab_item'] = self.monitor_event_token['lab_item'].get_vocabulary_size()
        num_nodes_dict['inj_item'] = self.monitor_event_token['inj_item'].get_vocabulary_size()
        names = list(self.feature_visit_event_keys) + ['lab_item', 'inj_item']
        self.gnn['monitoring_visit_graph'] \
            = HeteroGNN(['lab_item', 'inj_item'], names, num_nodes_dict, embedding_dim, device)

        # 给两个patient构图
        for feature_key in ['weight', 'age']:
            num_nodes_dict[feature_key] = 61
            names = list(self.feature_visit_event_keys) + [feature_key]
            self.gnn[feature_key] = HeteroGNN([feature_key], names, num_nodes_dict, embedding_dim, device)

        self.patient_state_gru = torch.nn.GRU(1, 1, batch_first=True)

        # === 新 Feature Interaction 的模块（pairwise visit fusion + 学习型池化）
        # 每个 visit 的向量由 len(feature_visit_event_keys)+2 个 embedding_dim 片段拼接而成
        self._parts_per_visit = len(self.feature_visit_event_keys) + 2  # +2 对应 lab_item, inj_item
        # 两种 fusion MLP：一种用于最初 concat 的大输入（initial），一种用于后续 fusion（pairwise，输入为 2*embedding_dim）
        self.fusion_in_dim_init = 2 * self._parts_per_visit * self.embedding_dim  # 初始 cat(visit, visit) 的输入维度
        self.fusion_in_dim_pair = 2 * self.embedding_dim  # 后续 pairwise 融合的输入维度
        self.visit_fusion_mlp_init = nn.Sequential(
            nn.Linear(self.fusion_in_dim_init, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.visit_fusion_mlp_pair = nn.Sequential(
            nn.Linear(self.fusion_in_dim_pair, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        # 备用投影（若出现其它尺寸，投影到 pair size）
        self.visit_fusion_proj_to_pair = nn.Linear(self.fusion_in_dim_init, self.fusion_in_dim_pair)

        self.pool_scorer = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 将最终得到的 patient vector 投影到 fc_patient 期望的维度 (item_num * embedding_dim)
        self.patient_proj = nn.Linear(self.embedding_dim, item_num * self.embedding_dim)

    def forward(self, batch_data):

        batch_size = len(batch_data['visit_id'])
        # patient_emb_list = []
        patient_emb_dict_origin = {}
        patient_emb_dict = {}
        patient_emb_dict_reg = {}

        """处理cond, proc, drug"""
        for feature_key in self.feature_visit_event_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(
                batch_data[feature_key], max_length=(400, 1024)
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event)

            x = self.dropout(self.embeddings[feature_key](x))
            # (patient, visit, event, embedding_dim)

            x = torch.sum(x, dim=2)
            # (patient, visit, embedding_dim)

            patient_emb_dict_origin[feature_key] = x
            # dict{feature_key: (patient, visit, embedding_dim)}

        """处理lab, inj"""
        # lab_key = 'lab_item'
        # inj_key = 'inj_item'
        # graph_key = 'monitoring_visit_graph'

        # 初始化多就诊字典 最后里面装的是dict{5 *list[patient * (visit, monitor, embedding_dim)]}
        patient_emb_list_dict = {'lab_item': [], 'inj_item': []}
        for feature_key in self.feature_visit_event_keys:
            patient_emb_list_dict[feature_key] = []

        visit_lengths = []
        max_monitors = {'lab_item': None, 'inj_item': None}
        # 迭代处理每一对
        feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))

        for feature_key1, feature_key2 in feature_paris:
            monitor_emb_list = []
            # 先聚合monitor层面，生成batch_size个病人的多次就诊的表征，batch_size * (1, visit, monitor, embedding)
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(400, 1024)
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(400, 1024)
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
                # (visit, monitor, event)

                x1 = self.embeddings[feature_key1](x1)
                x2 = self.embeddings[feature_key2](x2)
                # (visit, monitor, event, embedding_dim)

                x = self.dropout(torch.mul(x1, x2))
                # (visit, monitor, event, embedding_dim)

                x = torch.sum(x, dim=2)
                # (visit, monitor, embedding_dim)

                monitor_emb_list.append(x)

            """把monitor的数据变成指定样式"""
            aggregated_monitor_tensor, visit_lengths = aggregate_tensors(monitor_emb_list, self.device)
            # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加

            _, max_monitor_temp, _ = aggregated_monitor_tensor.size()
            max_monitors[feature_key1] = max_monitor_temp

            """把visit的数据变成指定样式"""
            patient_emb_list_dict[feature_key1] = aggregated_monitor_tensor

        # 处理其他feature key的数据并加入到patient_emb_list_dict中
        for feature_key in self.feature_visit_event_keys:
            x = patient_emb_dict_origin[feature_key]
            # (patient, visit, embedding_dim)
            padded_inputs = []
            for i, length in enumerate(visit_lengths):
                patient_visit_tensor = x[i, :length, :]  # 获取每个patient的所有visit
                # (visit, embedding_dim)
                repeated_tensor = patient_visit_tensor.unsqueeze(1).repeat(1, max(max_monitors.values()), 1)
                # (visit, monitor, embedding_dim)
                padded_inputs.append(repeated_tensor)

            aggregated_feature_tensor = torch.cat(padded_inputs, dim=0).to(self.device)
            # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加
            patient_emb_list_dict[feature_key] = aggregated_feature_tensor

        # ===== 替换原来的图构建 / GNN 处理，改为按 visit 节点的 pairwise 融合 + 学习型池化 =====
        # 构建每个 patient 的 visit 节点：把每个 visit 的 visit-event embeddings 与 lab/inj 的 monitor summary 拼接
        # patient_emb_list_dict[monitor_key] 的形状为 (patient*visit, monitor, embedding_dim)
        # patient_emb_dict_origin[feature_key] 的形状为 (patient, visit, embedding_dim)

        # 1) 计算每个 visit 的向量列表（按 patient）
        visit_nodes_per_patient = []  # list of list, 每个内层元素为 (visit_vec) tensor
        # 计算每个 patient 的 visit 基础索引偏移，用于从 patient_emb_list_dict 取 monitor 向量
        offsets = [0]
        for l in visit_lengths[:-1]:
            offsets.append(offsets[-1] + l)

        for p in range(batch_size):
            num_visits = visit_lengths[p]
            nodes = []
            base_idx = offsets[p]
            for v in range(num_visits):
                parts = []
                # visit-event 特征拼接
                for fk in self.feature_visit_event_keys:
                    parts.append(patient_emb_dict_origin[fk][p, v, :])  # (embedding_dim,)
                # monitor 特征：对对应 (patient*visit, monitor, dim) 做 sum -> (dim,)
                for monitor_key in ['lab_item', 'inj_item']:
                    agg = patient_emb_list_dict[monitor_key]  # (patient*visit, monitor, dim)
                    idx = base_idx + v
                    mon_vec = agg[idx].sum(dim=0)  # (embedding_dim,)
                    parts.append(mon_vec)
                visit_vec = torch.cat(parts, dim=-1)  # ((parts_per_visit * embedding_dim),)
                nodes.append(visit_vec)
            visit_nodes_per_patient.append(nodes)

        # 2) 对每个 patient 执行 pairwise 融合（1&2, 3&4, ...），迭代直到得到若干节点
        patient_reprs = []
        for nodes in visit_nodes_per_patient:
            if len(nodes) == 0:
                patient_reprs.append(torch.zeros(self.embedding_dim, device=self.device))
                continue
            cur = nodes
            # 如果单个 visit，先缩放为 embedding_dim
            if len(cur) == 1:
                single = cur[0]
                single_cat = torch.cat([single, single], dim=-1)  # 2 * parts_per_visit * embedding_dim
                single_in = single_cat.unsqueeze(0)
                fused_single = self.visit_fusion_mlp_init(single_in).squeeze(0)
                cur = [fused_single]
            else:
                # 迭代 pairwise 融合
                while len(cur) > 1:
                    nxt = []
                    for i in range(0, len(cur), 2):
                        if i + 1 < len(cur):
                            a = cur[i]
                            b = cur[i + 1]
                            ab = torch.cat([a, b], dim=-1)
                            # 统一成 (1, D)
                            ab_in = ab.unsqueeze(0) if ab.dim() == 1 else ab
                            D = ab_in.size(-1)
                            if D == self.fusion_in_dim_init:
                                fused_tensor = self.visit_fusion_mlp_init(ab_in).squeeze(0)
                            elif D == self.fusion_in_dim_pair:
                                fused_tensor = self.visit_fusion_mlp_pair(ab_in).squeeze(0)
                            else:
                                # 备用投影到 pair dim，再用 pair MLP
                                proj = self.visit_fusion_proj_to_pair(ab_in)
                                proj = proj[:, :self.fusion_in_dim_pair] if proj.size(-1) >= self.fusion_in_dim_pair else F.pad(proj, (0, self.fusion_in_dim_pair - proj.size(-1)))
                                fused_tensor = self.visit_fusion_mlp_pair(proj).squeeze(0)
                            nxt.append(fused_tensor)
                        else:
                            # 奇数个节点：对最后一个做 self-fusion 以保证降维为 embedding_dim
                            last = cur[i]
                            last_cat = torch.cat([last, last], dim=-1)
                            last_cat_in = last_cat.unsqueeze(0) if last_cat.dim() == 1 else last_cat
                            if last_cat_in.size(-1) == self.fusion_in_dim_init:
                                fused_last = self.visit_fusion_mlp_init(last_cat_in).squeeze(0)
                            elif last_cat_in.size(-1) == self.fusion_in_dim_pair:
                                fused_last = self.visit_fusion_mlp_pair(last_cat_in).squeeze(0)
                            else:
                                proj_last = self.visit_fusion_proj_to_pair(last_cat_in)
                                proj_last = proj_last[:, :self.fusion_in_dim_pair] if proj_last.size(-1) >= self.fusion_in_dim_pair else F.pad(proj_last, (0, self.fusion_in_dim_pair - proj_last.size(-1)))
                                fused_last = self.visit_fusion_mlp_pair(proj_last).squeeze(0)
                            nxt.append(fused_last)
                    cur = nxt

            # cur 现在为长度 >=1 的 embedding_dim 向量列表，做学习型池化（attention weight）
            stack = torch.stack(cur, dim=0)  # (num_nodes, embedding_dim)
            scores = self.pool_scorer(stack).squeeze(-1)  # (num_nodes,)
            weights = torch.softmax(scores, dim=0)  # (num_nodes,)
            patient_vec = (weights.unsqueeze(-1) * stack).sum(dim=0)  # (embedding_dim,)
            patient_reprs.append(patient_vec)

        # 3) 投影到 fc_patient 期望的维度，保持后续逻辑不变
        patient_emb = torch.stack(patient_reprs, dim=0)  # (batch_size, embedding_dim)
        patient_emb = self.patient_proj(patient_emb)  # -> (batch_size, item_num * embedding_dim)

        # ===== 后续保持原逻辑（使用 fc_patient 得到 logits） =====
        logits = self.fc_patient(patient_emb)

        # 为兼容训练流程，返回 (logits_reg, logits)
        # logits_reg 作为占位：长度为所有 visits 总数（若没有 visit_lengths 则退化为 batch_size）
        try:
            total_visits = sum(visit_lengths)
        except Exception:
            total_visits = batch_size
        logits_reg = torch.zeros(total_visits, device=self.device)
        return logits_reg, logits






