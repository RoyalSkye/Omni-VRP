
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)  # 1187712 (1.19 million)
        self.decoder = TSP_Decoder(**model_params)  # 82048 (0.08 million)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state, weights=None):
        if weights is not None and self.model_params["meta_update_encoder"]:
            self.encoded_nodes = self.encoder(reset_state.problems, weights=weights)
        else:
            self.encoded_nodes = self.encoder(reset_state.problems, weights=None)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes, weights=weights)

    def forward(self, state, weights=None, selected=None, return_probs=False):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)  # for torch.gather(dim=1)
            prob = torch.ones(size=(batch_size, pomo_size))
            probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node, weights=weights)  # pre-compute fixed part of the context embedding
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, weights=weights)
            # shape: (batch, pomo, problem)
            if selected is None:
                while True:
                    if self.training or self.model_params['eval_type'] == 'softmax':
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                    else:
                        selected = probs.argmax(dim=2)
                        # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        if return_probs:
            return selected, prob, probs
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, weights=None):
        if weights is None:
            # data.shape: (batch, problem, 2)
            embedded_input = self.embedding(data)
            # shape: (batch, problem, embedding)
            out = embedded_input
            for layer in self.layers:
                out = layer(out)
        else:
            embedded_input = F.linear(data, weights['encoder.embedding.weight'], weights['encoder.embedding.bias'])
            out = embedded_input
            for idx, layer in enumerate(self.layers):
                out = layer(out, weights=weights, index=idx)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, weights=None, index=0):
        # input1.shape: (batch, problem, EMBEDDING_DIM)
        if weights is None:
            head_num = self.model_params['head_num']
            q = reshape_by_heads(self.Wq(input1), head_num=head_num)
            k = reshape_by_heads(self.Wk(input1), head_num=head_num)
            v = reshape_by_heads(self.Wv(input1), head_num=head_num)
            # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
            out_concat = multi_head_attention(q, k, v)
            # shape: (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)
            # shape: (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            head_num = self.model_params['head_num']
            q = reshape_by_heads(F.linear(input1, weights['encoder.layers.{}.Wq.weight'.format(index)], bias=None), head_num=head_num)
            k = reshape_by_heads(F.linear(input1, weights['encoder.layers.{}.Wk.weight'.format(index)], bias=None), head_num=head_num)
            v = reshape_by_heads(F.linear(input1, weights['encoder.layers.{}.Wv.weight'.format(index)], bias=None), head_num=head_num)
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = F.linear(out_concat, weights['encoder.layers.{}.multi_head_combine.weight'.format(index)], weights['encoder.layers.{}.multi_head_combine.bias'.format(index)])
            if self.model_params['norm'] is None:
                out1 = self.addAndNormalization1(input1, multi_head_out)
            elif self.model_params["norm"] == "rezero":
                out1 = self.addAndNormalization1(input1, multi_head_out, weights={'weight': weights['encoder.layers.{}.addAndNormalization1.norm'.format(index)]})
            else:
                out1 = self.addAndNormalization1(input1, multi_head_out, weights={'weight': weights['encoder.layers.{}.addAndNormalization1.norm.weight'.format(index)], 'bias': weights['encoder.layers.{}.addAndNormalization1.norm.bias'.format(index)]})
            out2 = self.feedForward(out1, weights={'weight1': weights['encoder.layers.{}.feedForward.W1.weight'.format(index)], 'bias1': weights['encoder.layers.{}.feedForward.W1.bias'.format(index)],
                                                   'weight2': weights['encoder.layers.{}.feedForward.W2.weight'.format(index)], 'bias2': weights['encoder.layers.{}.feedForward.W2.bias'.format(index)]})
            if self.model_params['norm'] is None:
                out3 = self.addAndNormalization2(out1, out2)
            elif self.model_params["norm"] == "rezero":
                out3 = self.addAndNormalization2(out1, out2, weights={'weight': weights['encoder.layers.{}.addAndNormalization2.norm'.format(index)]})
            else:
                out3 = self.addAndNormalization2(out1, out2, weights={'weight': weights['encoder.layers.{}.addAndNormalization2.norm.weight'.format(index)], 'bias': weights['encoder.layers.{}.addAndNormalization2.norm.bias'.format(index)]})

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes, weights=None):
        if weights is None:
            # encoded_nodes.shape: (batch, problem, embedding)
            head_num = self.model_params['head_num']
            self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
            self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
            # shape: (batch, head_num, pomo, qkv_dim)
            self.single_head_key = encoded_nodes.transpose(1, 2)
            # shape: (batch, embedding, problem)
        else:
            head_num = self.model_params['head_num']
            self.k = reshape_by_heads(F.linear(encoded_nodes, weights['decoder.Wk.weight'], bias=None), head_num=head_num)
            self.v = reshape_by_heads(F.linear(encoded_nodes, weights['decoder.Wv.weight'], bias=None), head_num=head_num)
            self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1, weights=None):
        if weights is None:
            # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
            head_num = self.model_params['head_num']
            self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
            # shape: (batch, head_num, n, qkv_dim)
        else:
            head_num = self.model_params['head_num']
            self.q_first = reshape_by_heads(F.linear(encoded_q1, weights['decoder.Wq_first.weight'], bias=None), head_num=head_num)

    def forward(self, encoded_last_node, ninf_mask, weights=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        if weights is None:
            head_num = self.model_params['head_num']
            #  Multi-Head Attention
            #######################################################
            q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
            # shape: (batch, head_num, pomo, qkv_dim)
            q = self.q_first + q_last
            # shape: (batch, head_num, pomo, qkv_dim)
            out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            # shape: (batch, pomo, head_num*qkv_dim)
            mh_atten_out = self.multi_head_combine(out_concat)
            # shape: (batch, pomo, embedding)
            #  Single-Head Attention, for probability calculation
            #######################################################
            score = torch.matmul(mh_atten_out, self.single_head_key)
            # shape: (batch, pomo, problem)
            sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
            logit_clipping = self.model_params['logit_clipping']
            score_scaled = score / sqrt_embedding_dim
            # shape: (batch, pomo, problem)
            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask
            probs = F.softmax(score_masked, dim=2)
            # shape: (batch, pomo, problem)
        else:
            head_num = self.model_params['head_num']
            q_last = reshape_by_heads(F.linear(encoded_last_node, weights['decoder.Wq_last.weight'], bias=None), head_num=head_num)
            q = self.q_first + q_last
            out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            mh_atten_out = F.linear(out_concat, weights['decoder.multi_head_combine.weight'], weights['decoder.multi_head_combine.bias'])
            score = torch.matmul(mh_atten_out, self.single_head_key)
            sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
            logit_clipping = self.model_params['logit_clipping']
            score_scaled = score / sqrt_embedding_dim
            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask
            probs = F.softmax(score_masked, dim=2)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1, input2, weights=None):
        # input.shape: (batch, problem, embedding)
        if weights is None:
            if isinstance(self.norm, nn.InstanceNorm1d):
                added = input1 + input2
                transposed = added.transpose(1, 2)
                # shape: (batch, embedding, problem)
                normalized = self.norm(transposed)
                # shape: (batch, embedding, problem)
                back_trans = normalized.transpose(1, 2)
                # shape: (batch, problem, embedding)
            elif isinstance(self.norm, nn.BatchNorm1d):
                added = input1 + input2
                batch, problem, embedding = added.size()
                normalized = self.norm(added.reshape(batch * problem, embedding))
                back_trans = normalized.reshape(batch, problem, embedding)
            elif isinstance(self.norm, nn.Parameter):
                back_trans = input1 + self.norm * input2
            else:
                back_trans = input1 + input2
        else:
            if isinstance(self.norm, nn.InstanceNorm1d):
                added = input1 + input2
                transposed = added.transpose(1, 2)
                normalized = F.instance_norm(transposed, weight=weights['weight'], bias=weights['bias'])
                back_trans = normalized.transpose(1, 2)
            elif isinstance(self.norm, nn.BatchNorm1d):
                added = input1 + input2
                batch, problem, embedding = added.size()
                normalized = F.batch_norm(added.reshape(batch * problem, embedding), running_mean=self.norm.running_mean, running_var=self.norm.running_var, weight=weights['weight'], bias=weights['bias'], training=True)
                back_trans = normalized.reshape(batch, problem, embedding)
            elif isinstance(self.norm, nn.Parameter):
                back_trans = input1 + weights['weight'] * input2
            else:
                back_trans = input1 + input2

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1, weights=None):
        # input.shape: (batch, problem, embedding)
        if weights is None:
            return self.W2(F.relu(self.W1(input1)))
        else:
            output = F.relu(F.linear(input1, weights['weight1'], bias=weights['bias1']))
            return F.linear(output, weights['weight2'], bias=weights['bias2'])
