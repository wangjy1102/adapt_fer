import torch
import numpy as np
from    FERDB import FER_train
from    FERDB import FER_test
import argparse
from meta_yolo import Meta
from    torch.utils.data import DataLoader
from collections import defaultdict

import random
import torch
from functools import partial

def collate_4_diff_tasks(batch, dataset):
    """
    每次随机统一 n_way，再从 dataset 里重新采 n  个 *不同* 任务
    """
    n_way = random.choice([3, 4, 5])          # 统一当前 step 的 n_way
    dataset.set_n_way(n_way)                  

    sx, sy, qx, qy = [], [], [], []
    for _ in range(4): 
        sxi, syi, qxi, qyi, _ = dataset[0]
        sx.append(sxi)
        sy.append(syi)
        qx.append(qxi)
        qy.append(qyi)

    return torch.stack(sx), torch.stack(sy), torch.stack(qx), torch.stack(qy), [n_way] * 4

def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print("使用预训练权重进行ADAPT训练")
    print(args)

    device = torch.device('cuda')
    adapt = Meta(args, config=None).to(device)

    # 获取可训练参数数量
    trainable_params = sum(p.numel() for p in adapt.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in adapt.parameters())
    
    print(f'总参数量: {total_params}')
    print(f'可训练参数量: {trainable_params}')
    print(f'冻结参数量: {total_params - trainable_params}')
    print(adapt)

    # 加载数据集
    mini = FER_train('FER-2013/', mode='train', n_way=None, k_shot=args.k_spt,
                        k_query=args.k_qry, batchsz=1, resize=args.imgsz)
    mini_test = FER_test('JAFFE_JPG/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=300, resize=args.imgsz)
    db = DataLoader(mini, batch_size=1,
                              shuffle=False,
                              num_workers=0,        # 多线程需额外处理，先设 0
                              pin_memory=True,
                              collate_fn=partial(collate_4_diff_tasks, dataset=mini))

    for epoch in range(args.epoch//3000):
        # fetch meta_batchsz num of episode each time

        for step, (x_spt, y_spt, x_qry, y_qry, n_way_list) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            # print('>>> N_CLASS:', type(n_way_list), n_way_list)
            n_way = n_way_list[0]

            accs, loss_q = adapt(x_spt, y_spt, x_qry, y_qry, n_way)

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs, '\ttraining loss:', loss_q)

            if step % 180 == 0: # evaluation
            # if step % 180 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                # 1. 总计数器
                total_correct = defaultdict(int)
                total_count = defaultdict(int)
                n_tasks = 0

                for x_spt, y_spt, x_qry, y_qry , idx2emotion in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    # print('idx2emotion:', idx2emotion)

                    new_dict = {k: v[0] for k, v in idx2emotion.items()}
                    # print('Fixed mapping:', new_dict)

                    accs, task_correct, task_total = adapt.finetunning(x_spt, y_spt, x_qry, y_qry, new_dict)
                    accs_all_test.append(accs)


                    for emo, c in task_correct.items():
                        total_correct[emo] += c
                        total_count[emo] += task_total[emo]
                    n_tasks += 1

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30000)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=128)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0005)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.005)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    
    # 预训练权重相关参数
    argparser.add_argument('--pretrained_path', type=str, help='预训练权重路径', default=None)
    argparser.add_argument('--freeze_backbone', action='store_true', help='是否冻结主干网络', default=True)
    argparser.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false', help='不冻结主干网络')

    args = argparser.parse_args()
    main()