import copy
import time
import random
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer

        self.len = len(data['train']['edge_index_list'])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength

        x = data['x'].to(args.device)
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x

        # fixed internal constants: not exposed as user hyperparameters
        self.route_div_temp = 0.7
        self.intervention_scale = 0.5
        self.aux_decay = 0.99

        # running statistics for self-normalized auxiliary objective
        self.aux_stats = {
            'variant': 1.0,
            'kl': 1.0,
            'proto': 1.0,
            'mask': 1.0,
            'orth': 1.0
        }

        setup_seed(args.seed)
        print('total length: {}, train length: {}, val length: {}, test length: {}'.format(
            self.len, self.len_train, self.len_val, self.len_test))

    def run(self):
        args = self.args
        best_val_auc = -1.0
        best_test_auc = -1.0
        best_train_auc = -1.0
        best_epoch = 0
        best_state = None
        patience = 0
        best_ood_results = (0, 0, 0, 0)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        t_total0 = time.time()
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                average_epoch_loss, average_train_auc, average_val_auc, average_test_auc = \
                    self.train(epoch, self.data['train'])

                if average_val_auc > best_val_auc:
                    best_val_auc = average_val_auc
                    best_test_auc = average_test_auc
                    best_train_auc = average_train_auc
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_ood_results = self.test(epoch, self.data['test'])
                    patience = 0
                else:
                    patience += 1
                    if epoch > args.min_epoch and patience > args.patience:
                        break

                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                        epoch, average_epoch_loss, time.time() - t0))
                    print(f"(IID) Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, "
                          f"Val AUC:{average_val_auc:.4f}, Test AUC:{average_test_auc:.4f}")
                    print(f"(IID) Best: Epoch:{best_epoch}, Train AUC:{best_train_auc:.4f}, "
                          f"Val AUC:{best_val_auc:.4f}, Test AUC:{best_test_auc:.4f}")
                    print(f"(OOD) Best_OOD_Test: Epoch:{best_ood_results[0]}, "
                          f"Train AUC:{best_ood_results[1]:.4f}, "
                          f"Val AUC:{best_ood_results[2]:.4f}, "
                          f"Test AUC:{best_ood_results[3]:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        epoch_time = (time.time() - t_total0) / max(1, best_epoch)
        return (best_train_auc, best_val_auc, best_test_auc,
                best_ood_results, epoch_time, best_epoch,
                average_train_auc, average_val_auc, average_test_auc)

    def train(self, epoch, data):
        self.model.train()
        optimizer = self.optimizer

        out = self.model(data['edge_index_list'], self.x, seq_len=self.len_train)
        causal_embeddings = out['causal_embeddings']
        variant_embeddings = out['variant_embeddings']
        route_probs = out['route_probs']

        snapshot_routes = route_probs.mean(dim=0)   # [T,K]
        dominant_expert = route_probs.argmax(dim=-1)  # [N,T]

        causal_losses = []
        variant_losses = []
        invariance_losses = []

        for t in range(self.len_train - 1):
            causal_z = causal_embeddings[:, t, :]
            variant_z = variant_embeddings[:, t, :]

            pos_edge_index, neg_edge_index = self._get_edges(data, t + 1)

            # causal branch
            causal_pos_pred, causal_neg_pred = self.cal_pred(
                causal_z, pos_edge_index, neg_edge_index, self.model.cs_decoder
            )
            causal_losses.append(self.loss(causal_pos_pred, causal_neg_pred))

            # variant branch as auxiliary pressure to avoid degeneration
            variant_pos_pred, variant_neg_pred = self.cal_pred(
                variant_z, pos_edge_index, neg_edge_index, self.model.ss_decoder
            )
            variant_losses.append(self.loss(variant_pos_pred, variant_neg_pred))

            # distribution-aware intervention
            inter_losses = []
            for _ in range(self.args.intervention_times):
                src_t = self.sample_intervention_time(t, snapshot_routes, self.len_train)

                intervened_variant = self.distribution_aware_variant_replace(
                    variant_embeddings[:, t, :],
                    variant_embeddings[:, src_t, :],
                    dominant_expert[:, t],
                    dominant_expert[:, src_t]
                )

                conf_pos_pred, conf_neg_pred = self.cal_pred(
                    intervened_variant, pos_edge_index, neg_edge_index, self.model.ss_decoder
                )

                mixed_pos = self.mix_predictions(causal_pos_pred, conf_pos_pred.detach())
                mixed_neg = self.mix_predictions(causal_neg_pred, conf_neg_pred.detach())
                inter_losses.append(self.loss(mixed_pos, mixed_neg))

            inter_losses = torch.stack(inter_losses)
            invariance_losses.append(inter_losses.mean() + inter_losses.var(unbiased=False))

        causal_loss = torch.stack(causal_losses).mean() if len(causal_losses) else torch.tensor(0., device=self.args.device)
        variant_loss = torch.stack(variant_losses).mean() if len(variant_losses) else torch.tensor(0., device=self.args.device)
        inv_loss = torch.stack(invariance_losses).mean() if len(invariance_losses) else torch.tensor(0., device=self.args.device)

        aux_dict = {
            'variant': variant_loss,
            'kl': out['loss_kl'],
            'proto': out['loss_proto'],
            'mask': out['loss_mask'],
            'orth': out['loss_orth']
        }
        aux_reg = self.compose_aux_loss(aux_dict)

        warm = self.args.weight1 if epoch >= self.args.warm_epoch else 0.0
        loss = causal_loss + warm * (inv_loss + aux_reg)

        optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        optimizer.step()

        average_epoch_loss = loss.item()
        average_train_auc, average_val_auc, average_test_auc = self.eval_iid(data)

        return average_epoch_loss, average_train_auc, average_val_auc, average_test_auc

    @torch.no_grad()
    def eval_iid(self, data):
        self.model.eval()
        out = self.model(data['edge_index_list'], self.x, seq_len=self.len)
        causal_embeddings = out['causal_embeddings']

        train_auc_list, val_auc_list, test_auc_list = [], [], []
        for t in range(self.len - 1):
            z = causal_embeddings[:, t, :]
            pos_edge, neg_edge = self._get_edges(data, t + 1)
            if neg_edge.size(1) == 0 or pos_edge.size(1) == 0:
                continue

            auc, _ = self.predict(z, pos_edge, neg_edge, self.model.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        return (np.mean(train_auc_list) if len(train_auc_list) else 0.0,
                np.mean(val_auc_list) if len(val_auc_list) else 0.0,
                np.mean(test_auc_list) if len(test_auc_list) else 0.0)

    @torch.no_grad()
    def test(self, epoch, data):
        self.model.eval()
        local_len = len(data['edge_index_list'])
        x = self.x[:local_len] if isinstance(self.x, list) else self.x

        out = self.model(data['edge_index_list'], x, seq_len=local_len)
        causal_embeddings = out['causal_embeddings']

        train_auc_list, val_auc_list, test_auc_list = [], [], []

        for t in range(local_len - 1):
            z = causal_embeddings[:, t, :]
            pos_edge, neg_edge = self._get_edges(data, t + 1)
            if neg_edge.size(1) == 0 or pos_edge.size(1) == 0:
                continue

            auc, _ = self.predict(z, pos_edge, neg_edge, self.model.cs_decoder)

            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        return (epoch,
                np.mean(train_auc_list) if len(train_auc_list) else 0.0,
                np.mean(val_auc_list) if len(val_auc_list) else 0.0,
                np.mean(test_auc_list) if len(test_auc_list) else 0.0)

    def _get_edges(self, data, t):
        if self.args.dataset == 'act':
            pos_edge_index = data['pedges'][t].long().to(self.args.device)
            neg_edge_index = negative_sampling(
                pos_edge_index,
                self.args.num_nodes,
                num_neg_samples=pos_edge_index.shape[1]
            )
        else:
            pos_edge_index = data['pedges'][t].long().to(self.args.device)
            neg_edge_index = data['nedges'][t].long().to(self.args.device)
        return pos_edge_index, neg_edge_index

    def cal_pred(self, z, edge_index_pos, edge_index_neg, decoder):
        pos_pred = decoder(z, edge_index_pos)
        neg_pred = decoder(z, edge_index_neg)
        return pos_pred, neg_pred

    def loss(self, pos_pred, neg_pred):
        pos_pred = pos_pred.clamp(1e-8, 1 - 1e-8)
        neg_pred = neg_pred.clamp(1e-8, 1 - 1e-8)
        pos_loss = -torch.log(pos_pred).mean()
        neg_loss = -torch.log(1 - neg_pred).mean()
        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index, decoder):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def js_divergence(self, p, q):
        eps = 1e-8
        p = p / (p.sum() + eps)
        q = q / (q.sum() + eps)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
        kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
        return 0.5 * (kl_pm + kl_qm)

    def sample_intervention_time(self, target_t, snapshot_routes, max_t):
        candidates = [i for i in range(max_t) if i != target_t]
        if len(candidates) == 0:
            return target_t

        target_route = snapshot_routes[target_t]
        divs = []
        for s in candidates:
            divs.append(self.js_divergence(target_route, snapshot_routes[s]))
        divs = torch.stack(divs)

        probs = F.softmax(divs / self.route_div_temp, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return candidates[idx]

    def distribution_aware_variant_replace(self, variant_t, variant_s, dom_t, dom_s):
        mask = (dom_t != dom_s).float().unsqueeze(-1)
        return mask * variant_s + (1.0 - mask) * variant_t

    def mix_predictions(self, causal_pred, conf_pred):
        eps = 1e-8
        causal_pred = causal_pred.clamp(eps, 1 - eps)
        conf_pred = conf_pred.clamp(eps, 1 - eps)

        causal_logit = torch.log(causal_pred) - torch.log(1 - causal_pred)
        conf_logit = torch.log(conf_pred) - torch.log(1 - conf_pred)
        mixed_logit = causal_logit + self.intervention_scale * conf_logit
        return torch.sigmoid(mixed_logit).clamp(eps, 1 - eps)

    def compose_aux_loss(self, aux_dict):
        # update EMA scales
        with torch.no_grad():
            for k, v in aux_dict.items():
                val = max(float(v.detach().item()), 1e-8)
                self.aux_stats[k] = self.aux_decay * self.aux_stats[k] + (1 - self.aux_decay) * val

        normalized = []
        for k, v in aux_dict.items():
            normalized.append(v / (self.aux_stats[k] + 1e-8))
        return torch.stack(normalized).mean()