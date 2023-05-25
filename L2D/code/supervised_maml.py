import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.nn as gnn
from collections import OrderedDict

from util import *


def load_data(args, path):
    data = np.load(path)
    xs = VRFullProblem.process_node_counts(data['xs'], args.ptype, use_count=args.use_count_feature)
    ts = data['ts']
    n_routes = data['n_routes']
    routes = data['routes']
    route_neighbors = data['route_neighbors']
    unique_masks = data['unique_masks']
    labels = data['labels']

    max_n_routes = n_routes.max()
    ts = ts[:, :max_n_routes]
    _, n_nodes, d_node = xs.shape
    d_route = ts.shape[-1]

    return Namespace(
        # dimensions
        N=len(n_routes), n_nodes=n_nodes, d_node=d_node, d_route=d_route,
        max_n_routes=max_n_routes,

        # matrices
        xs=xs, ts=ts,
        n_routes=n_routes,
        routes=np.pad(routes, [(0, 0), (1, 0)])[:, :n_nodes + max_n_routes],
        route_neighbors=route_neighbors[:, :max_n_routes],
        labels=labels[:, :max_n_routes],
        unique_masks=unique_masks[:, :max_n_routes],
    )


def load_subproblem_data(path):
    data = np.load(path)
    xs, offsets, node_idxs, n_subp_nodes, lkh_dists, prev_dists = data['xs'], data['offsets'], data['subp_node_idxs'], data['n_subp_nodes'], data['lkh_dists'], data['prev_dists']
    node_idxs = np.split(node_idxs, np.cumsum(n_subp_nodes[:-1]))
    return Namespace(
        N=len(n_subp_nodes), d_node=xs.shape[-1],
        xs=xs, offsets=offsets, node_idxs=pad_each(node_idxs), n_subp_nodes=n_subp_nodes, lkh_dists=lkh_dists, prev_dists=prev_dists
    )


def load_subproblem_statistics(path):
    npz = np.load(path)
    data = Namespace({k: npz[k] for k in npz.files})
    data.N = len(data.statistics)
    return data


def stack_froms_tos(froms, tos, inverse=False, both=False):
    if inverse:
        froms, tos = tos, froms
    pairs = np.stack([froms, tos], axis=-2)
    if both:
        return np.concatenate([pairs, np.stack([tos, froms], axis=-2)], axis=-1)
    return pairs


def build_edge_neighbors(neighbors, inverse=False, both=False):
    n_batch, n_nodes, n_neighbors = neighbors.shape
    froms = np.tile(np.repeat(np.arange(n_nodes), n_neighbors), (n_batch, 1))
    tos = neighbors.reshape(n_batch, -1)
    return stack_froms_tos(froms, tos, inverse, both)


def build_edge_routes(routes, inverse=False, both=False):
    return stack_froms_tos(routes[:, :-1], routes[:, 1:], inverse, both)


def build_edge_agent2route(routes, inverse=False, both=False):
    is_depot = routes == 0
    tos = np.cumsum(is_depot, axis=1) - is_depot
    return stack_froms_tos(routes, tos, inverse, both)


def build_edge_cluster2route(routes, route_neighbors, inverse=False, both=False):
    n_batch, max_n_routes, _ = route_neighbors.shape
    routes = [np.array(unpack_routes(rs), dtype=object) for rs in routes]
    nodes = [[np.concatenate(rs[rns_i]) for rns_i in rns] for rs, rns in zip(routes, route_neighbors)]
    n_nodes = [[len(ns_i) for ns_i in ns] for ns in nodes]
    nodes = [np.concatenate(ns) for ns in nodes]
    max_n_nodes = max(len(ns) for ns in nodes)
    froms = np.array([pad_to(ns, max_n_nodes) for ns in nodes])
    tos = np.array([pad_to(np.repeat(np.arange(max_n_routes), nns), max_n_nodes, constant_values=max_n_routes) for nns in n_nodes])
    return stack_froms_tos(froms, tos, inverse, both)


def batch_edges(e, inc, no_from_zeros=False, no_to_zeros=False, max_froms=None, max_tos=None, lengths=None):
    # Batch and increment the edges
    n_batch, two, n_e = e.shape
    inc = np.arange(n_batch).reshape(-1, 1, 1) * np.reshape(inc, (1, -1, 1))
    e_inc = np.transpose(e + inc, axes=(1, 0, 2)).reshape(2, -1)

    need_from_to_zeros = no_from_zeros or no_to_zeros
    need_from_to_e = max_froms is not None or max_tos is not None
    if need_from_to_zeros or need_from_to_e or lengths is not None:
        mask = np.ones(n_batch * n_e, dtype=np.bool)
        if need_from_to_zeros:
            from_zeros, to_zeros = np.transpose(e == 0, axes=(1, 0, 2)).reshape(2, -1)
            if no_from_zeros:
                mask[from_zeros] = False
            if no_to_zeros:
                mask[to_zeros] = False
        if need_from_to_e:
            froms, tos = e[:, 0], e[:, 1]
            if max_froms is not None:
                mask[(froms >= max_froms.reshape(n_batch, 1)).reshape(-1)] = False
            if max_tos is not None:
                mask[(tos >= max_tos.reshape(n_batch, 1)).reshape(-1)] = False
        if lengths is not None:
            tails = np.tile(np.arange(n_e), (n_batch, 1)) >= lengths.reshape(n_batch, 1)
            mask[tails.reshape(-1)] = False
        e_inc = e_inc[:, mask]
    return e_inc


def to_tensor(x, device='cpu'):
    if x is None: return None
    dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32 if np.issubdtype(x.dtype, np.floating) else None
    return torch.tensor(x, dtype=dtype, device=device)


def get_prepare(args, d=None, rotate=False, flip=False, perturb_node=False, perturb_route=False):
    def prep_tensors(b):
        arrays = dict(x=b.xs, t=b.ts, unique_mask=b.unique_masks, labels=b.get('labels'),
            e_xct=batch_edges(build_edge_cluster2route(b.routes, b.route_neighbors), inc_xt, max_tos=b.n_routes), # routes[:, 1:] remove the leading 0
            e_t=batch_edges(build_edge_neighbors(b.route_neighbors, inverse=True), inc_t, max_froms=b.n_routes, max_tos=b.n_routes),
        )
        b_t = Namespace((k, to_tensor(v, device=args.device)) for k, v in arrays.items())
        b_t.x, b_t.t = VRFullProblem.transform_features(b_t.x, b_t.t, rotate=rotate, flip=flip, perturb_node=perturb_node, perturb_route=perturb_route)
        return b_t

    if d is None: # Generation time
        inc_xt = inc_x, inc_t = [0, 0]
        return prep_tensors
    else: # Training and evaluation time
        inc_xt = inc_x, inc_t = [d.n_nodes, d.max_n_routes]
        keys = ['xs', 'ts', 'n_routes', 'routes', 'route_neighbors', 'unique_masks', 'labels']
        return lambda idxs: prep_tensors(Namespace((k, d[k][idxs]) for k in keys))


def get_prepare_subproblem(args, d=None, rotate=False, flip=False, perturb_node=False, perturb_route=False):
    def prep_batch(b):
        b.labels = b.get('lkh_dists', None)
        if args.fit_statistics:
            b.x = b.statistics
        else:
            b.x = b.xs[b.offsets.reshape(-1, 1) + b.node_idxs[:, :b.n_subp_nodes.max()]]
        b_t = Namespace((k, to_tensor(b[k], device=args.device)) for k in model_keys)
        if not args.fit_statistics:
            b_t.x, _ = VRFullProblem.transform_features(b_t.x, None, rotate=rotate, flip=flip, perturb_node=perturb_node)
        return b_t

    model_keys = ['x', 'labels'] if args.fit_statistics else ['x', 'n_subp_nodes', 'labels', 'prev_dists']
    if d is None: # Generation time
        return prep_batch
    else: # Training and evaluation time
        if args.fit_statistics:
            return lambda idxs: prep_batch(Namespace((k, d[k][idxs]) for k in ['statistics', 'lkh_dists']))
        return lambda idxs: prep_batch(Namespace(((k, d[k][idxs]) for k in ['offsets', 'node_idxs', 'n_subp_nodes', 'lkh_dists', 'prev_dists']), xs=d.xs))


def restore(args, net, opt=None):
    if args.step is None:
        models = list(args.model_save_dir.glob('*.pth'))
        if len(models) == 0:
            print('No model checkpoints found')
            return None
        step, load_path = max((int(p.stem), p) for p in models) # Load the max step
    else:
        step, load_path = args.step, args.model_save_dir / f'{args.step}.pth'
    ckpt = torch.load(load_path, map_location=args.device)
    net.load_state_dict(ckpt['net'])
    if opt is not None:
        opt.load_state_dict(ckpt['opt'])
    print(f'Loaded network{"" if opt is None else " and optimizer"} from {load_path}')
    return ckpt['step']


class Block(nn.Module):
    def __init__(self, args, d_hidden):
        super(Block, self).__init__()
        Mod = getattr(gnn, args.gnn_module)
        if Mod == gnn.TransformerConv:
            Mod = lambda d_in, d_out: gnn.TransformerConv(d_in, d_out, heads=args.transformer_heads, concat=False)
        elif Mod == gnn.GINConv:
            Mod = lambda d_in, d_out: gnn.GINConv(nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.ReLU(inplace=True),
                nn.Linear(d_out, d_out),
            ))
        elif Mod == gnn.PNAConv:
            Mod = lambda d_in, d_out: gnn.PNAConv(d_in, d_out, ['mean', 'min', 'max', 'std'], scalers=['linear'], deg=torch.tensor(1, device='cuda:0'))
        self.gat_tt = Mod(d_hidden, d_hidden)
        self.gat_xct = Mod(d_hidden, d_hidden)
        self.out = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden) if args.use_layer_norm else nn.Identity()
        )
        self.fc_x = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        ) if args.use_x_fc else lambda x: 0

    def forward(self, x, t, e_t, e_xct):
        t = (t + self.gat_tt(t, e_t) + self.gat_xct((x, t), e_xct)).relu()
        return x + self.fc_x(x), t + self.out(t)


class Network(nn.Module):
    def __init__(self, args, d):
        super(Network, self).__init__()
        self.args = args
        self.d_hidden = d_hidden = args.d_hidden

        if args.normalize_features:
            xs = d.xs.reshape(-1, d.d_node).astype(np.float32)
            ts = d.ts.reshape(-1, d.d_route)
            ts = ts[(ts != 0).any(axis=1)].astype(np.float32)
            d.xs_mean, d.xs_std = xs.mean(axis=0), xs.std(axis=0)
            d.ts_mean, d.ts_std = ts.mean(axis=0), ts.std(axis=0)
            for name in 'xs_mean', 'xs_std', 'ts_mean', 'ts_std':
                size = d.d_node if name.startswith('xs') else d.d_route
                self.register_buffer(name, torch.tensor(d.get(name, np.zeros(size))))

        self.fc_x = nn.Linear(d.d_node, d_hidden)
        self.fc_t = nn.Linear(d.d_route, d_hidden)
        self.blocks = nn.ModuleList(Block(args, d_hidden) for _ in range(args.n_layers))

        self.fc_out = nn.Linear(d_hidden, 1)

    def forward(self, d):
        """
        Contents of d:
        x: shape (n_batch, n_nodes, d_node)
        t: shape (n_batch, max_n_routes, d_route)
        e_n, e_r, e_xct, e_xt, e_t: shape (2, num_edges)
        labels: shape (n_batch, max_n_routes)
        """
        args = self.args
        x, t, e_t, e_xct, unique_mask, labels = d.x, d.t, d.e_t, d.e_xct, d.unique_mask, d.get('labels')
        n_batch, n_nodes, d_node = x.shape
        _, max_n_routes, d_route = t.shape

        x = x.view(-1, d_node)
        t = t.view(-1, d_route)
        if args.normalize_features:
            x = (x - self.xs_mean) / self.xs_std
            t = (t - self.ts_mean) / self.ts_std
        x = self.fc_x(x).relu()
        t = self.fc_t(t).relu()

        for block in self.blocks:
            x, t = block(x, t, e_t, e_xct)

        scores = self.fc_out(t.view((n_batch, max_n_routes, self.d_hidden))).squeeze(-1)

        # If provided, negative labels indicate improvements in distance by the LKH action, positive labels indicate worse changes
        if args.loss.startswith('MSE'):
            if labels is None:
                scores[~unique_mask] = -np.inf
                return scores # Predicted improvement in distance
            labels = -labels # Flip the labels so that positive is good
            if args.loss == 'MSE_clip':
                labels[labels < 0] = 0
                scores[(scores < 0) & (labels == 0)] = 0
            loss = ((scores - labels) ** 2).mean()
        else: # Softmax cross entropy (CE) loss
            scores[~unique_mask] = -np.inf
            logps = scores.log_softmax(dim=-1)
            if labels is None:
                return logps.exp()
            labels[~unique_mask] = np.inf
            labels_logps = (-labels / args.temperature).log_softmax(dim=-1)

            loss = (labels_logps.exp() * (labels_logps - logps))[unique_mask].sum() / n_batch # KL loss
        return loss


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, weights=None, id=0):
        if weights is None:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                weights['layers.layers.{}.self_attn.in_proj_weight'.format(id)], weights['layers.layers.{}.self_attn.in_proj_bias'.format(id)],
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, weights['layers.layers.{}.self_attn.out_proj.weight'.format(id)], weights['layers.layers.{}.self_attn.out_proj.bias'.format(id)],
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, weights=None):
        output = src
        for id, mod in enumerate(self.layers):
            if weights is None:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, weights=weights, id=id)
        if self.norm is not None:
            output = self.norm(output)  # norm is None

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, weights=None, id=0):
        if weights is None:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, weights=weights, id=id)[0]
            src = src + self.dropout1(src2)
            src = F.layer_norm(src, (self.d_model,), weight=weights["layers.layers.{}.norm1.weight".format(id)], bias=weights['layers.layers.{}.norm1.bias'.format(id)])
            src2 = F.linear(self.dropout(self.activation(F.linear(src, weights['layers.layers.{}.linear1.weight'.format(id)], weights['layers.layers.{}.linear1.bias'.format(id)]))), weights['layers.layers.{}.linear2.weight'.format(id)], weights['layers.layers.{}.linear2.bias'.format(id)])
            src = src + self.dropout2(src2)
            src = F.layer_norm(src, (self.d_model, ), weight=weights['layers.layers.{}.norm2.weight'.format(id)], bias=weights['layers.layers.{}.norm2.bias'.format(id)])
        return src


class SubproblemNetwork(nn.Module):
    def __init__(self, args, d):
        super(SubproblemNetwork, self).__init__()
        self.args = args
        self.d_hidden = d_hidden = args.d_hidden

        self.register_buffer('mean_lkh_dist', torch.tensor(d.lkh_dists.mean()))
        if args.fc_only:
            self.fc_in = nn.Sequential(
                nn.Linear(d.d_node + args.use_prev_dist_feature, d_hidden),
                getattr(nn, args.activation)(),
                nn.Linear(d_hidden, d_hidden),
            )
            self.fc_out = nn.Sequential(
                getattr(nn, args.activation)(),
                nn.Linear(d_hidden, d_hidden),
                getattr(nn, args.activation)(),
                nn.Linear(d_hidden, 1),
            )
        else:
            self.fc = nn.Linear(d.d_node + args.use_prev_dist_feature, d_hidden)
            layer = TransformerEncoderLayer(d_model=d_hidden, nhead=args.transformer_heads, dim_feedforward=d_hidden * 4, dropout=args.dropout)
            self.layers = TransformerEncoder(layer, num_layers=args.n_layers)

            self.fc_out = nn.Linear(d_hidden, 1)

    def forward(self, d, weights=None):
        """
        Contents of d:
        x: shape (n_batch, max_n_subp_nodes, d_node)
        n_subp_nodes: shape (n_batch,)
        labels: shape (n_batch,)
        """
        args = self.args
        x, n_subp_nodes, prev_dists, labels = d.x, d.n_subp_nodes, d.prev_dists, d.get('labels')
        n_batch, max_n_subp_nodes, _ = x.shape

        mask = torch.arange(max_n_subp_nodes, device=x.device).expand(n_batch, max_n_subp_nodes) >= n_subp_nodes.unsqueeze(1) # (n_batch, max_n_subp_nodes)
        if args.use_prev_dist_feature:
            x = torch.cat((x, prev_dists.view(n_batch, 1, 1).expand(n_batch, max_n_subp_nodes, 1)), dim=2)
        if args.fc_only:
            x = self.fc_in(x).sum(dim=1) / n_subp_nodes.reshape(-1, 1)
            preds = self.fc_out(x).flatten() + self.mean_lkh_dist # (n_batch,)
        else:
            if weights is None:
                x = self.fc(x)
                x = self.layers(x.transpose(0, 1), src_key_padding_mask=mask) # Transformer takes in (max_n_subp_nodes, n_batch, d_node)
                outs = self.fc_out(x).squeeze(-1) # (max_n_subp_nodes, n_batch)
            else:
                x = F.linear(x, weights['fc.weight'], weights['fc.bias'])
                x = self.layers(x.transpose(0, 1), src_key_padding_mask=mask, weights=weights)
                outs = F.linear(x, weights['fc_out.weight'], weights['fc_out.bias']).squeeze(-1)

            outs[mask.T] = 0
            preds = outs.sum(dim=0) / n_subp_nodes + self.mean_lkh_dist  # (n_batch,)

        if labels is None:
            return preds # Predicted distance
        if args.loss.endswith('clip'):
            clipped = labels >= prev_dists
            labels[clipped] = prev_dists[clipped]
            clipped = (preds > prev_dists) & clipped
            preds[clipped] = prev_dists[clipped]
        if args.loss.startswith('MSE'):
            return F.mse_loss(preds, labels)
        if args.loss.startswith('MAE'):
            return F.l1_loss(preds, labels)
        return F.smooth_l1_loss(preds, labels, beta=1.0)


class FCNetwork(nn.Module):
    def __init__(self, args, d):
        super(FCNetwork, self).__init__()
        self.args = args

        self.register_buffer('mean_lkh_dist', torch.tensor(d.lkh_dists.mean()))
        fc = []
        in_size = d.statistics.shape[1]
        for i in range(args.n_layers):
            is_last = i < args.n_layers - 1
            out_size = args.d_hidden if is_last else 1
            fc.append(nn.Linear(in_size, out_size))
            not is_last and fc.append(getattr(nn, args.activation)())
            in_size = out_size
        self.fc = nn.Sequential(*fc)

    def forward(self, d):
        """
        Contents of d:
        x: shape (n_batch, n_statistics)
        labels: shape (n_batch,)
        """
        preds = self.fc(d.x).flatten() + self.mean_lkh_dist # (n_batch,)
        labels = d.labels

        if labels is None:
            return preds # Predicted distance
        if args.loss.endswith('clip'):
            clipped = labels >= prev_dists
            labels[clipped] = prev_dists[clipped]
            clipped = (preds > prev_dists) & clipped
            preds[clipped] = prev_dists[clipped]
        if args.loss.startswith('MSE'):
            return F.mse_loss(preds, labels)
        if args.loss.startswith('MAE'):
            return F.l1_loss(preds, labels)
        return F.smooth_l1_loss(preds, labels, beta=1.0)


def train(args, d, d_eval, d_generate):
    start_time = time()
    writer = SummaryWriter(log_dir=args.train_dir, flush_secs=10)

    net = (FCNetwork if args.fit_statistics else SubproblemNetwork if args.fit_subproblem else Network)(args, d).to(args.device)
    opt = Adam(net.parameters(), lr=args.lr)
    start_step = restore(args, net, opt=opt)

    scheduler = CosineAnnealingLR(opt, args.n_steps, last_epoch=-1 if start_step is None else start_step)
    start_step = start_step or 0

    prep = (get_prepare_subproblem if args.fit_subproblem else get_prepare)(args, d, rotate=args.augment_rotate, flip=args.augment_flip, perturb_node=args.augment_perturb_node, perturb_route=args.augment_perturb_route)
    def log(text, **kwargs):
        print(f'Step {step}: {text}', flush=True)
        [writer.add_scalar(k, v, global_step=step, walltime=time() - start_time) for k, v in kwargs.items()]

    start, end = 0, 75000
    meta_grad_dict = {(i, j): 0 for i, group in enumerate(opt.param_groups) for j, _ in enumerate(group['params'])}
    for step in range(start_step, args.n_steps + 1):
        if step % args.n_step_save == 0 or step == args.n_steps:
            args.model_save_dir.mkdir(exist_ok=True)
            ckpt = dict(step=step, net=net.state_dict(), opt=opt.state_dict())
            torch.save(ckpt, args.model_save_dir / f'{step}.pth')
        if step % args.n_step_eval == 0:
            evaluate(args, d_eval, net, log)
        if step % args.n_step_generate == 0 and (step > 0 or args.generate_step_zero):
            generate(args, d_generate, net, step)
        if step == args.n_steps: break
        if step > 10000:  # simple curriculm learning strategy, note: could simply use random task scheduler instead,
            start, end = 75000, d.N

        opt.zero_grad()
        net.train()

        for b in range(1):  # inner-loop
            # maml
            fast_weight = OrderedDict(net.named_parameters())

            # fomaml
            # task_net = copy.deepcopy(net)
            # task_opt = Adam(task_net.parameters(), lr=args.lr)
            # # task_opt.load_state_dict(opt.state_dict())

            for k in range(1):  # outer-loop
                train_data = prep(np.random.choice(range(start, end), size=args.n_batch))
                # task_net.train()
                # loss = task_net(train_data)
                # task_opt.zero_grad()
                # loss.backward()
                # task_opt.step()

                # for maml
                loss = net(train_data, weights=fast_weight)
                gradients = torch.autograd.grad(loss, fast_weight.values(), create_graph=True)  # allow_unused=True
                w_t, (beta1, beta2), eps = [], opt.param_groups[0]['betas'], opt.param_groups[0]['eps']
                lr, weight_decay = args.lr, 0
                for i, ((name, param), grad) in enumerate(zip(fast_weight.items(), gradients)):
                    if opt.state_dict()['state'] != {}:
                        # (with batch/instnace norm layer): i \in [0, 85], where encoder \in [0, 79] + decoder \in [80, 85]
                        # (with rezero norm layer): i \in [0, 73], where encoder \in [0, 67] + decoder \in [68, 73]
                        # (without norm layer): i \in [0, 61], where encoder \in [0, 55] + decoder \in [56, 61]
                        # i = i if self.model_params['meta_update_encoder'] else i + 80
                        state = opt.state_dict()['state'][i]
                        steps, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']
                        steps += 1
                        steps = steps.item() if isinstance(steps, torch.Tensor) else steps
                        # compute grad based on Adam source code using in-place operation
                        # update Adam stat (step, exp_avg and exp_avg_sq have already been updated by in-place operation)
                        # may encounter RuntimeError: (a leaf Variable that requires grad) / (the tensor used during grad computation) cannot use in-place operation.
                        grad = grad.add(param, alpha=weight_decay)
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        bias_correction1 = 1 - beta1 ** steps
                        bias_correction2 = 1 - beta2 ** steps
                        step_size = lr / bias_correction1
                        bias_correction2_sqrt = math.sqrt(bias_correction2)
                        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                        # param.addcdiv_(exp_avg, denom, value=-step_size)
                        param = param - step_size * exp_avg / denom
                        opt.state_dict()['state'][i]['exp_avg'] = exp_avg.clone().detach()
                        opt.state_dict()['state'][i]['exp_avg_sq'] = exp_avg_sq.clone().detach()
                    else:
                        param = param - lr * grad
                    w_t.append((name, param))
                fast_weight = OrderedDict(w_t)

            val_data = prep(np.random.choice(range(start, end), size=args.n_batch))
            # loss1 = task_net(val_data)
            # loss1 = loss1 * w=1
            # task_opt.zero_grad()
            # loss1.backward()

            loss1 = net(train_data, weights=fast_weight)
            opt.zero_grad()
            loss1.backward()
            for i, group in enumerate(opt.param_groups):
                for j, p in enumerate(group['params']):
                    meta_grad_dict[(i, j)] += p.grad

            # for i, group in enumerate(task_opt.param_groups):
            #     for j, p in enumerate(group['params']):
            #         meta_grad_dict[(i, j)] += p.grad

        # opt.zero_grad()
        # for i, group in enumerate(opt.param_groups):
        #     for j, p in enumerate(group['params']):
        #         p.grad = meta_grad_dict[(i, j)]
        # opt.step()

        # loss = net(prep(np.random.choice(d.N, size=args.n_batch)))
        # loss = net(data)
        # opt.zero_grad()
        # loss.backward()
        # opt.step()

        opt.zero_grad()
        for i, group in enumerate(opt.param_groups):
            for j, p in enumerate(group['params']):
                p.grad = meta_grad_dict[(i, j)]
        opt.step()

        loss = loss.item()
        lr = scheduler._last_lr[0]
        log(f'loss={loss:.4f} loss1={loss1:.4f} lr={lr:.4f}', loss=loss, loss1=loss1, lr=lr)
        scheduler.step()

    writer.close()


def train_sklearn(args, d, d_eval, d_generate):
    assert args.fit_statistics
    import sklearn
    import joblib

    start = time()
    skargs = Namespace(args.sklearn_parameters)
    if skargs.model == 'RandomForestRegressor':
        import sklearn.ensemble
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=skargs.get('n_estimators', 100), min_samples_split=skargs.get('min_samples_split', 2), max_features=skargs.get('max_features', 'auto'), n_jobs=args.n_cpus)
    elif skargs.model == 'ElasticNet':
        model = sklearn.linear_model.ElasticNet(alpha=skargs.get('alpha', 1), l1_ratio=skargs.get('l1_ratio', 0.5))
    elif skargs.model == 'MLPRegressor':
        import sklearn.neural_network
        model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=tuple(skargs.get('hidden_layer_sizes', (100,))), activation=skargs.get('activation', 'relu'), alpha=skargs.get('alpha', 0.0001))
    model.fit(d.statistics, d.lkh_dists)
    fit_time = time() - start

    pred = model.predict(d.statistics)
    train_mse = ((pred - d.lkh_dists) ** 2).mean()

    pred = model.predict(d_eval.statistics)
    np.save(args.train_dir / 'predictions.npy', pred)
    eval_mse = ((pred - d_eval.lkh_dists) ** 2).mean()
    print(f'Train MSE: {train_mse}')
    print(f'Eval MSE: {eval_mse}')

    result = {'fit_time': fit_time, 'train_mse': train_mse, 'evaluation_mse': eval_mse}

    with open(args.train_dir / 'eval.json', 'w+') as f:
        json.dump(result, f)

    joblib.dump(model, args.train_dir / 'model.joblib')


def evaluate(args, d, net, log=None):
    print(f'Evaluating on {d.N} problems...')
    eval_start_time = time()
    net.eval()
    prep = (get_prepare_subproblem if args.fit_subproblem else get_prepare)(args, d)
    total_loss = 0
    with torch.no_grad():
        for idxs in np.split(range(d.N), range(args.n_batch, d.N, args.n_batch)):
            loss = net(prep(idxs))
            total_loss += loss.item() * len(idxs)
    loss = total_loss / d.N
    eval_time = time() - eval_start_time

    if log is not None:
        log(f'eval_loss={loss:.4f} eval_time={eval_time:.1f}s', eval_loss=loss, eval_time=eval_time)


class NetAC(ActionCallback):
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model
        self.traj_probs = []
        self.subproblem_cache = {}
        self.x = None
        self.prep = (get_prepare_subproblem if args.fit_subproblem else get_prepare)(args)

    def action_order(self, p):
        args = self.args
        if self.x is None or args.use_count_feature:
            self.x = VRFullProblem.process_node_counts(p.get_node_features(), p.ptype, use_count=args.use_count_feature)
        if args.fit_subproblem:
            cache = self.subproblem_cache
            subps = p.get_subproblems(n_subproblems=args.n_subproblems, n_route_neighbors=args.n_route_neighbors, temperature=args.subproblem_temperature)
            subps_todo = [subp for subp in set(subps) if subp not in cache]

            gen_batch_size = len(subps_todo) if args.use_sklearn else args.n_batch
            if len(subps_todo):
                for start in range(0, len(subps_todo), gen_batch_size):
                    end = start + gen_batch_size
                    subps_batch = subps_todo[start: end]
                    prev_dists = np.array([subp.total_dist for subp in subps_batch])
                    if args.use_sklearn:
                        preds = self.model.predict(np.array([subp.get_features() for subp in subps_batch]))
                    else:
                        node_idxs = [subp.node_idxs for subp in subps_batch]
                        n_subp_nodes = np.array([len(ni) for ni in node_idxs])
                        offsets = np.zeros_like(n_subp_nodes)
                        data = Namespace(xs=self.x, offsets=offsets, node_idxs=pad_each(node_idxs), n_subp_nodes=n_subp_nodes, prev_dists=prev_dists)
                        with torch.no_grad():
                            preds = self.model(self.prep(data)).cpu().numpy()
                    for subp, pred in zip(subps_batch, preds):
                        cache[subp] = pred
            preds = np.array([subp.total_dist - cache[subp] for subp in subps])
        else:
            data = dict(xs=self.x, ts=p.get_route_features(),
                n_routes=len(p.routes), routes=pack_routes(p.routes, left_pad=1, right_pad=0),
                route_neighbors=p.route_neighbors[:, :args.n_route_neighbors],
                unique_masks=p.unique_mask,
            )
            data = Namespace((k, np.expand_dims(v, axis=0)) for k, v in data.items())
            with torch.no_grad():
                preds = self.model(self.prep(data)).cpu().numpy()[0]
        self.traj_probs.append(preds)
        if args.sample:
            assert not args.sample
            return np.random.choice(len(preds), size=(args.beam_width,), p=scipy.special.softmax(preds), replace=False)
        return np.argsort(-preds) # preds is the improvements (higher is better)

    def kwargs_fn(self):
        return dict(probs=pad_each(self.traj_probs), **super().kwargs_fn())


def generate_ij(task):
    (i, j), (nodes, demands, capacity, dist, init_routes, pkwargs), model, save_dir, args = task
    save_path = save_dir / f'{i}{f"_{j}" if j else ""}.npz'
    if save_path.exists():
        print(f'Skipping {save_path} since already generated', flush=True)
        return
    print(f'Generating {save_path}...', flush=True)
    mask = demands > 0
    mask[0] = True
    nodes, demands = nodes[mask], demands[mask]

    not args.use_sklearn and model.eval()
    start_time = time()
    cb = NetAC(args, model)
    if args.ptype == 'CVRPTW':
        pkwargs.update(window_distance_scale=args.window_distance_scale)
    save_beam_search(save_path, nodes, demands, capacity, init_routes, args, pkwargs=pkwargs, n_cpus=1,
        action_fn=cb.action_fn, feedback_fn=cb.feedback_fn, kwargs_fn=cb.kwargs_fn
    )
    print(f'Generated {save_path} in {time() - start_time:.3f}s', flush=True)


def generate(args, d, model, step):
    print(f'Generating {args.n_trajectories} trajectories each for problems {args.generate_index_start} to {args.generate_index_end}...')
    save_dir = args.generate_save_dir / f'{step}' if step else args.generate_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    *d, pkwargs = d
    pkwargs = {k: pkwargs[k] for k in dict(CVRP=[], CVRPTW=['window', 'service_time'], VRPMPD=['is_pickup'])[args.ptype]}
    tasks = [[
        (i, j),
        (*[x[i] for x in d], {k: v[i] for k, v in pkwargs.items()}),
        model, save_dir, args
    ] for i in range(args.generate_index_start, args.generate_index_end) for j in range(args.n_trajectories)]
    multiprocess(generate_ij, tasks, cpus=args.n_cpus if args.device == 'cpu' else 1)


parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('train_dir', type=Path)
parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
parser.add_argument('--window_distance_scale', type=float, default=0)
parser.add_argument('--data_suffix', type=str, default='')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fit_subproblem', action='store_true')
parser.add_argument('--fit_statistics', action='store_true')
parser.add_argument('--use_sklearn', action='store_true')
parser.add_argument('--sklearn_parameters', type=yaml.safe_load, default={})
parser.add_argument('--fc_only', action='store_true')
parser.add_argument('--augment_rotate', action='store_true')
parser.add_argument('--augment_flip', action='store_true')
parser.add_argument('--augment_perturb_node', type=float, default=0.0)
parser.add_argument('--augment_perturb_route', type=float, default=0.0)
parser.add_argument('--use_count_feature', action='store_true')
parser.add_argument('--use_prev_dist_feature', action='store_true')
parser.add_argument('--normalize_features', action='store_true')
parser.add_argument('--n_steps', type=int, default=40000)
parser.add_argument('--n_step_save', type=int, default=5000)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--n_batch', type=int, default=256)
parser.add_argument('--n_route_neighbors', type=int, default=15)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--d_hidden', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.01)
parser.add_argument('--use_layer_norm', action='store_true')
parser.add_argument('--use_x_fc', action='store_true')
parser.add_argument('--gnn_module', type=str, default='GATConv')
parser.add_argument('--transformer_heads', type=int, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--activation', type=str, default='ReLU')

parser.add_argument('--step', type=int, default=None)

parser.add_argument('--eval', action='store_true')
parser.add_argument('--eval_partition', type=str, default='val', choices=['train', 'val', 'test'])
parser.add_argument('--n_step_eval', type=int, default=1000)

parser.add_argument('--generate', action='store_true')
parser.add_argument('--save_dir', type=Path, default=None)
parser.add_argument('--save_suffix', type=str, default=None)
parser.add_argument('--generate_partition', type=str, default='val', choices=['train', 'val', 'test'])
parser.add_argument('--generate_partition_suffix', type=str, default='')
parser.add_argument('--n_step_generate', type=int, default=None)
parser.add_argument('--generate_step_zero', action='store_true')
parser.add_argument('--generate_index_start', type=int, default=0)
parser.add_argument('--generate_index_end', type=int, default=2)
parser.add_argument('--n_trajectories', type=int, default=1)
parser.add_argument('--generate_depth', type=int, default=30)
parser.add_argument('--n_subproblems', type=int, default=None)
parser.add_argument('--subproblem_temperature', type=float, default=0)
parser.add_argument('--sample', action='store_true')
parser.add_argument('--beam_width', type=int, default=None)
parser.add_argument('--improve_threshold', type=float, default=-float('inf'))
parser.add_argument('--detect_duplicate', action='store_true')
parser.add_argument('--no_cache', action='store_true')
parser.add_argument('--solver', type=str, choices=['LKH', 'HGS'], default='HGS')
parser.add_argument('--n_lkh_trials', type=int, default=None) # LKH
parser.add_argument('--time_threshold', type=int, default=None) # HGS
parser.add_argument('--n_cpus', type=int, default=None)
parser.add_argument('--init_tour', action='store_true')


def get_generate(args):
    generate_name = f'generations'
    if args.save_suffix or args.dataset_dir.parent.name != args.save_dir.name:
        generate_name += args.save_suffix or f'_{args.save_dir.name}' # Transfer to another dataset
    labels = dict(generate_partition='', beam_width='beam', generate_depth='depth', sample='sample', improve_threshold='improve', detect_duplicate='nodup', no_cache='nocache', init_tour='init', n_lkh_trials='lkh', time_threshold='hgsthres', n_subproblems='nsubp', subproblem_temperature='subptemp')
    for arg_name, arg_label in labels.items():
        arg_value = getattr(args, arg_name)
        if arg_value != parser.get_default(arg_name):
            generate_name += f'_{arg_label}{"" if isinstance(arg_value, bool) else arg_value}'
    return args.train_dir / generate_name


type_map = {a.dest: a.type for a in parser._actions}


if __name__ == '__main__':
    args = parser.parse_args()
    args.generate_partition += args.generate_partition_suffix
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.model_save_dir = args.train_dir / 'models'

    config = args.train_dir / 'config.yaml'
    if (args.eval or args.generate):
        assert config.exists()
        obj = load_yaml(config)
        for k, v in obj.items():
            if getattr(args, k) == parser.get_default(k):
                type_ = type_map[k]
                setattr(args, k, type_(v) if type_ is not None else v)
        print(f'Loaded args from {config}')
    else:
        obj = {k: v if isinstance(v, yaml_types) else str(v) for k, v in args.__dict__.items() if v != parser.get_default(k)}
        if config.exists():
            prev_obj = load_yaml(config)
            assert sorted(prev_obj.items()) == sorted(obj.items()), f'Previous training configuration at {config} is different than current training run\'s configs. Either use the same configs or delete {config.parent}.'
        else:
            save_yaml(config, obj)
            print(f'Saved args to {config}')
    print(args, flush=True)

    args.generate_save_dir = get_generate(args) if args.generate or args.n_step_generate else None
    args.beam_width = args.beam_width or 1
    args.n_step_generate = args.n_step_generate or np.inf
    args.n_cpus = args.n_cpus or args.beam_width

    if args.fit_subproblem:
        args.loss = args.loss or 'Huber'
        load, suffix = (load_subproblem_statistics, 'subproblem_statistics') if args.fit_statistics else (load_subproblem_data, 'subproblems')
        path_eval = args.dataset_dir / f'{args.eval_partition}{args.data_suffix}_{suffix}.npz'
        d_eval = load(path_eval)
        print(f'Loaded evaluation data from {path_eval}. {d_eval.N} total labeled subproblems')
    else:
        args.loss = args.loss or 'CE'
        path_eval = args.dataset_dir / f'{args.eval_partition}{args.data_suffix}.npz'
        d_eval = load_data(args, path_eval)
        print(f'Loaded evaluation data from {path_eval}. {d_eval.N} total subproblems with {d_eval.n_routes.sum()} total labels')

    d_generate = None
    if args.generate_save_dir:
        d_generate = load_problems(args.save_dir / f'problems_{args.generate_partition}.npz')
        print(f'Loaded {len(d_generate[0])} evaluation problem instances from {args.save_dir / f"problems_{args.generate_partition}.npz"} for generating trajectories to {args.generate_save_dir}', flush=True)

    if args.eval or args.generate:
        if args.use_sklearn:
            import joblib
            model = joblib.load(args.train_dir / 'model.joblib')
            step = None
        else:
            model = (FCNetwork if args.fit_statistics else SubproblemNetwork if args.fit_subproblem else Network)(args, d_eval)
            step = restore(args, model)
            model = model.to(args.device)

        if args.eval:
            evaluate(args, d_eval, model, log=lambda *args, **kwargs: print(*args))

        if args.generate:
            assert args.solver == 'LKH' and args.n_lkh_trials or args.solver == 'HGS' and args.time_threshold
            generate(args, d_generate, model, step)
    else:
        print(f'Saving experiment progress in {args.train_dir}')
        if args.fit_subproblem:
            path_train = args.dataset_dir / f'train{args.data_suffix}_{suffix}.npz'
            d = load(path_train)
            print(f'Loaded training data from {path_train}. {d.N} total labeled subproblems')
        else:
            path_train = args.dataset_dir / f'train{args.data_suffix}.npz'
            d = load_data(args, path_train)
            print(f'Loaded training data from {path_train}. {d.N} total problems with {d.n_routes.sum()} total labels.')
        (train_sklearn if args.use_sklearn else train)(args, d, d_eval, d_generate)
