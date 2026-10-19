import copy
import time
import torch
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NC_Motif_Trainer(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer

        self.len = len(data['edge_index_list'])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength

        self.node_mask = [torch.arange(data['x_list'][i].shape[0]) for i in range(self.len)]
        x = data['x_list'][-1].float().to(args.device)
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x

        self.route_div_temp = 0.7
        self.intervention_scale = 0.5
        self.aux_decay = 0.99
        self.aux_stats = {
            'variant': 1.0,
            'kl': 1.0,
            'proto': 1.0,
            'mask': 1.0,
            'orth': 1.0
        }

        setup_seed(args.seed)
        print('total length: {}, test length: {}'.format(self.len, args.testlength))

    def run(self):
        args = self.args
        max_acc, max_test_acc, max_train_acc = -1.0, -1.0, -1.0
        min_epoch = args.min_epoch
        max_patience = args.patience

        best_epoch = 0
        best_state = None
        best_test_acc_list = [0.0 for _ in range(max(1, self.len_test))]
        patience = 0

        average_train_acc, average_val_acc, average_test_acc = 0.0, 0.0, 0.0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        t_total0 = time.time()
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                average_epoch_loss, average_train_acc, average_val_acc, average_test_acc, test_acc_list = self.train(epoch, self.data)

                if average_val_acc > max_acc:
                    max_acc = average_val_acc
                    max_test_acc = average_test_acc
                    max_train_acc = average_train_acc
                    best_test_acc_list = test_acc_list
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break

                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                    print(f"Current: Epoch:{epoch}, Train Acc :{average_train_acc:.4f}, Val Acc: {average_val_acc:.4f}, Test Acc: {average_test_acc:.4f}")
                    print(f"Best_metric: Epoch:{best_epoch}, Train Acc:{max_train_acc:.4f}, Val Acc: {max_acc:.4f}, Test Acc: {max_test_acc:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        denom_epoch = best_epoch if best_epoch > 0 else 1
        epoch_time = (time.time() - t_total0) / denom_epoch
        return (
            epoch, average_train_acc, average_val_acc, average_test_acc,
            best_epoch, max_train_acc, max_acc, max_test_acc,
            epoch_time, best_test_acc_list
        )

    def train(self, epoch, data):
        self.model.train()
        optimizer = self.optimizer

        out = self.model(data['edge_index_list'], self.x, seq_len=self.len_train)
        causal_embeddings = out['causal_embeddings']
        variant_embeddings = out['variant_embeddings']
        route_probs = out['route_probs']

        snapshot_routes = route_probs.mean(dim=0)      # [T, K]
        dominant_expert = route_probs.argmax(dim=-1)   # [N, T]

        criterion = torch.nn.CrossEntropyLoss()

        causal_losses = []
        variant_losses = []
        invariance_losses = []

        for t in range(self.len_train):
            node_mask = self.node_mask[t].to(self.args.device)
            labels = data['label_list'][t].squeeze().long().to(self.args.device)
            y_t = labels if labels.size(0) == node_mask.size(0) else labels[node_mask]

            causal_logits = self.model.cs_decoder(causal_embeddings[:, t, :])[node_mask]
            variant_logits = self.model.ss_decoder(variant_embeddings[:, t, :])[node_mask]

            causal_losses.append(criterion(causal_logits, y_t))
            variant_losses.append(criterion(variant_logits, y_t))

            inter_losses = []
            for _ in range(self.args.intervention_times):
                src_t = self.sample_intervention_time(t, snapshot_routes, self.len_train)

                intervened_variant = self.distribution_aware_variant_replace(
                    variant_embeddings[:, t, :],
                    variant_embeddings[:, src_t, :],
                    dominant_expert[:, t],
                    dominant_expert[:, src_t]
                )

                conf_logits = self.model.ss_decoder(intervened_variant)[node_mask]
                mixed_logits = self.mix_logits(causal_logits, conf_logits.detach())
                inter_losses.append(criterion(mixed_logits, y_t))

            if len(inter_losses):
                inter_losses = torch.stack(inter_losses)
                invariance_losses.append(inter_losses.mean() + inter_losses.var(unbiased=False))

        causal_loss = torch.stack(causal_losses).mean() if len(causal_losses) else torch.tensor(0.0, device=self.args.device)
        variant_loss = torch.stack(variant_losses).mean() if len(variant_losses) else torch.tensor(0.0, device=self.args.device)
        inv_loss = torch.stack(invariance_losses).mean() if len(invariance_losses) else torch.tensor(0.0, device=self.args.device)

        aux_dict = {
            'variant': variant_loss,
            'kl': out['loss_kl'],
            'proto': out['loss_proto'],
            'mask': out['loss_mask'],
            'orth': out['loss_orth']
        }
        aux_reg = self.compose_aux_loss(aux_dict)

        la = self.args.weight1 if epoch >= self.args.warm_epoch else 0.0
        loss = causal_loss + la * (inv_loss + aux_reg)

        optimizer.zero_grad()
        loss.backward()
        if getattr(self.args, 'grad_clip', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        optimizer.step()

        average_epoch_loss = loss.item()
        average_train_acc, average_val_acc, average_test_acc, test_acc_list = self.eval_iid(data)
        return average_epoch_loss, average_train_acc, average_val_acc, average_test_acc, test_acc_list

    @torch.no_grad()
    def eval_iid(self, data):
        self.model.eval()
        out = self.model(data['edge_index_list'], self.x, seq_len=self.len)
        causal_embeddings = out['causal_embeddings']

        train_acc_list, val_acc_list, test_acc_list = [], [], []

        for t in range(self.len):
            node_mask = self.node_mask[t].to(self.args.device)
            labels = data['label_list'][t].squeeze().long().to(self.args.device)
            y_t = labels if labels.size(0) == node_mask.size(0) else labels[node_mask]

            logits = self.model.cs_decoder(causal_embeddings[:, t, :])[node_mask]
            pred = logits.argmax(dim=-1)
            acc = float((pred == y_t).float().mean().item())

            if t < self.len_train:
                train_acc_list.append(acc)
            elif t < self.len_train + self.len_val:
                val_acc_list.append(acc)
            else:
                test_acc_list.append(acc)

        return (
            np.mean(train_acc_list) if len(train_acc_list) else 0.0,
            np.mean(val_acc_list) if len(val_acc_list) else 0.0,
            np.mean(test_acc_list) if len(test_acc_list) else 0.0,
            test_acc_list
        )

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

    def mix_logits(self, causal_logits, conf_logits):
        return causal_logits + self.intervention_scale * conf_logits

    def compose_aux_loss(self, aux_dict):
        with torch.no_grad():
            for k, v in aux_dict.items():
                val = max(float(v.detach().item()), 1e-8)
                self.aux_stats[k] = self.aux_decay * self.aux_stats[k] + (1 - self.aux_decay) * val

        normalized = []
        for k, v in aux_dict.items():
            normalized.append(v / (self.aux_stats[k] + 1e-8))
        return torch.stack(normalized).mean()