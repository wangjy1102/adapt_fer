import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    nets.adapt_yolo_all import ADAPT_YoloBody
from    copy import deepcopy
from collections import defaultdict



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config=None):
        """

        :param args:
        :param config: 保留此参数以保持向后兼容，但不再使用
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        # 使用YoloBody替代Learner，支持预训练权重和主干网络冻结
        pretrained_path = getattr(args, 'pretrained_path', None)
        freeze_backbone = getattr(args, 'freeze_backbone', True)

        self.net = ADAPT_YoloBody(
            args.imgc,
            args.imgsz,
            args.n_way,
            phi='s',
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone
        )
        self.fcone = nn.Linear(224, 1).to('cuda')
        meta_params = list(self.net.parameters()) + list(self.fcone.parameters())
        self.meta_optim = optim.Adam(meta_params, lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def _create_fast_weights(self, params, grad, update_lr):
        """根据梯度和原始参数创建fast_weights，正确处理冻结参数"""
        fast_weights = []
        grad_idx = 0
        for param in params:
            if param.requires_grad:
                fast_weights.append(param - update_lr * grad[grad_idx])
                grad_idx += 1
            else:
                fast_weights.append(param)
        return fast_weights

    def _get_trainable_params(self, params):
        """从参数列表中筛选出可训练参数"""
        return [p for p in params if p.requires_grad]

    def forward(self, x_spt, y_spt, x_qry, y_qry, N_CLASS):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        # print(f'[DEBUG] y_qry range: {y_qry.min().item()} ~ {y_qry.max().item()}',
        #       f'model out_features: {self.net.classifier.out_features}')
        # assert y_qry.max() < self.net.classifier.out_features, 'label 越界!'
        # print('[DEBUG] forward 收到的 y_qry 最大值:', y_qry.max().item(),
        #       'n_way:', self.n_way)
        # assert y_qry.max() < self.n_way, f"标签越界！y_qry.max={y_qry.max()} n_way={self.n_way}"

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        criterion_inner = SmoothCE(n_cls=N_CLASS, eps=0.1)


        for i in range(task_num):
            self.net.classifier.weight = torch.nn.Parameter(
                self.fcone.weight.repeat(N_CLASS, 1)  # 去掉 .data，保留梯度跟踪
            )
            self.net.classifier.bias = torch.nn.Parameter(
                self.fcone.bias.repeat(N_CLASS, 1).reshape(-1, )  # 去掉 .data
            )

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].long())
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # trainable_params = self._get_trainable_params(list(self.net.parameters()))
            # grad = torch.autograd.grad(loss, trainable_params, create_graph=True)
            # fast_weights = self._create_fast_weights(list(self.net.parameters()), grad, self.update_lr)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].long()).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].long())
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # trainable_params = self._get_trainable_params(list(fast_weights))
                # grad = torch.autograd.grad(loss, trainable_params, create_graph=True)
                # fast_weights = self._create_fast_weights(list(fast_weights), grad, self.update_lr)

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i].long())
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()#每个可调节参数的梯度，设置为0
        loss_q.backward()
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs, loss_q


    def finetunning(self, x_spt, y_spt, x_qry, y_qry, idx2emotion):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        criterion_inner = SmoothCE(n_cls=self.n_way, eps=0.1)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        self.net.classifier.weight = torch.nn.Parameter(
            self.fcone.weight.repeat(self.n_way, 1)  # 去掉 .data，保留梯度跟踪
        )
        self.net.classifier.bias = torch.nn.Parameter(
            self.fcone.bias.repeat(self.n_way, 1).reshape(-1, )  # 去掉 .data
        )

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt.long())
        # grad = torch.autograd.grad(loss, net.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        trainable_params = self._get_trainable_params(list(net.parameters()))
        grad = torch.autograd.grad(loss, trainable_params)
        fast_weights = self._create_fast_weights(list(net.parameters()), grad, self.update_lr)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            # # 2. compute grad on theta_pi
            # grad = torch.autograd.grad(loss, fast_weights)
            # # 3. theta_pi = theta_pi - train_lr * grad
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            trainable_params = self._get_trainable_params(list(fast_weights))
            grad = torch.autograd.grad(loss, trainable_params)
            fast_weights = self._create_fast_weights(list(fast_weights), grad, self.update_lr)

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry.long())

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry.long()).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

            if k == self.update_step_test - 1:  # 最后一次更新
                with torch.no_grad():
                    # logits_q = net(x_qry, fast_weights, bn_training=True)
                    # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

                    emotion_correct = defaultdict(int)
                    emotion_total = defaultdict(int)

                    for p, q in zip(pred_q, y_qry):
                        emo = idx2emotion[q.item()]
                        emotion_total[emo] += 1
                        if p.item() == q.item():
                            emotion_correct[emo] += 1


        del net

        accs = np.array(corrects) / querysz

        return accs, emotion_correct, emotion_total

class SmoothCE(nn.Module):
    def __init__(self, n_cls, eps=0.1):
        super().__init__()
        self.n_cls, self.eps = n_cls, eps
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, logits, y):
        # y : LongTensor [N]
        one_hot = F.one_hot(y, self.n_cls).float()
        soft = (1 - self.eps) * one_hot + self.eps / self.n_cls
        loss = - (soft * self.logsoft(logits)).sum(dim=1).mean()
        return loss


def main():
    pass


if __name__ == '__main__':
    main()
