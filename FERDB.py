import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random


class FER_train(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        self.batchsz   = batchsz          # 仅决定 __len__，不再预生成列表
        self.n_way     = n_way if n_way is not None else None   # 外部可统一设置
        self.k_shot    = k_shot
        self.k_query   = k_query
        self.resize    = resize
        self.startidx  = startidx
        self.path      = os.path.join(root, mode)
        self.cls_names = sorted([d for d in os.listdir(self.path)
                                 if os.path.isdir(os.path.join(self.path, d))])
        self.cls_num   = len(self.cls_names)

        # 标签映射：文件名 → 全局 label
        self.img2label = {}
        self.data      = []                 # 按类别存放相对路径
        for glb_l, cls_name in enumerate(self.cls_names):
            cls_path = os.path.join(self.path, cls_name)
            imgs = [os.path.join(cls_name, f)
                    for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.png'))]
            self.data.append(imgs)
            for img in imgs:
                self.img2label[os.path.splitext(os.path.basename(img))[0]] = glb_l + startidx

        # transform
        if mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((resize, resize)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        print(f'shuffle DB :{mode}, 实时采样, {k_shot}-shot, {k_query}-query, resize:{resize}')

    # -------------------- 外部统一 n_way --------------------
    def set_n_way(self, n_way):
        self.n_way   = n_way
        self.setsz   = n_way * self.k_shot
        self.querysz = n_way * self.k_query

    # -------------------- 实时采样一条任务 --------------------
    def __getitem__(self, idx):
        max_retry = 10
        for retry in range(max_retry):
            try:
                if self.n_way is None:                      # 保险
                    self.set_n_way(random.choice([3, 4, 5]))

                # 1. 选 n_way 个类
                cls_idx = np.random.choice(self.cls_num, self.n_way, replace=False)

                # 2. 每类抽 (k_shot+k_query) 张图
                support_paths, query_paths = [], []
                support_lbl, query_lbl     = [], []
                for rel_idx, cls in enumerate(cls_idx):
                    imgs = self.data[cls]
                    if len(imgs) < self.k_shot + self.k_query:
                        idx_s = np.random.choice(len(imgs), self.k_shot + self.k_query, replace=True)
                    else:
                        idx_s = np.random.choice(len(imgs), self.k_shot + self.k_query, replace=False)
                    np.random.shuffle(idx_s)
                    for i in idx_s[:self.k_shot]:
                        support_paths.append(os.path.join(self.path, imgs[i]))
                        support_lbl.append(self.img2label[os.path.splitext(os.path.basename(imgs[i]))[0]])
                    for i in idx_s[self.k_shot:]:
                        query_paths.append(os.path.join(self.path, imgs[i]))
                        query_lbl.append(self.img2label[os.path.splitext(os.path.basename(imgs[i]))[0]])

                # 3. 相对标签 0~n_way-1
                unique = np.unique(support_lbl)
                if not set(query_lbl).issubset(unique):
                    raise ValueError('查询集含支撑集未出现类')
                random.shuffle(unique)
                rel_map = {old: new for new, old in enumerate(unique)}
                support_y = np.array([rel_map[l] for l in support_lbl], dtype=np.int32)
                query_y = np.array([rel_map[l] for l in query_lbl], dtype=np.int32)

                # 4. 加载图像
                support_x = torch.stack([self.transform(p) for p in support_paths])
                query_x   = torch.stack([self.transform(p) for p in query_paths])

                # 5. 校验（可选）
                if support_y.max() >= self.n_way or query_y.max() >= self.n_way:
                    raise ValueError('映射后标签越界')
                if not set(query_y).issubset(set(support_y)):
                    raise ValueError('查询集含新类')

                # 0~n_way-1 映射已完成
                # ----- 合法性检查 -----
                if not set(query_y).issubset(set(support_y)):
                    raise ValueError('查询集含支撑集未出现类')
                if support_y.max() >= self.n_way or support_y.min() < 0:
                    raise ValueError(f'支撑集标签越界: max={support_y.max()} min={support_y.min()} n_way={self.n_way}')
                if query_y.max() >= self.n_way or query_y.min() < 0:
                    raise ValueError(f'查询集标签越界: max={query_y.max()} min={query_y.min()} n_way={self.n_way}')

                return support_x, torch.LongTensor(support_y), \
                       query_x,   torch.LongTensor(query_y), \
                       self.n_way

            except ValueError:
                if retry == max_retry - 1:
                    return self[(idx + 1) % len(self)]   # 跳下个索引
                continue

    def __len__(self):
        return 100000          # 仅决定 epoch 长度，不再对应预生成列表
    # def __getitem__(self, index):
    #     support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
    #     support_y = np.zeros((self.setsz), dtype=np.int32)
    #     query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
    #     query_y = np.zeros((self.querysz), dtype=np.int32)
    #
    #     # Flatten support and query file paths
    #     flatten_support_x = [os.path.join(self.path, item)
    #                          for sublist in self.support_x_batch[index] for item in sublist]
    #
    #     # Extract labels from filenames (remove extension for img2label mapping)
    #     support_y = np.array([self.img2label[os.path.splitext(os.path.basename(item))[0]]
    #                           for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
    #
    #     flatten_query_x = [os.path.join(self.path, item)
    #                        for sublist in self.query_x_batch[index] for item in sublist]
    #     query_y = np.array([self.img2label[os.path.splitext(os.path.basename(item))[0]]
    #                         for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
    #
    #     # Convert to relative labels (0 to n_way-1)
    #     unique = np.unique(support_y)
    #     random.shuffle(unique)
    #     support_y_relative = np.zeros(self.setsz)
    #     query_y_relative = np.zeros(self.querysz)
    #
    #     for idx, l in enumerate(unique):
    #         support_y_relative[support_y == l] = idx
    #         query_y_relative[query_y == l] = idx
    #
    #     # Load and transform images
    #     for i, path in enumerate(flatten_support_x):
    #         support_x[i] = self.transform(path)
    #
    #     for i, path in enumerate(flatten_query_x):
    #         query_x[i] = self.transform(path)
    #
    #     if index >= 30:
    #         print('[TASK]', index, ' support 全局标签:', np.unique(support_y))
    #         print('[TASK]', index, ' query  全局标签:', np.unique(query_y))
    #         print('[TASK]', index, ' 映射后 query_max:', query_y_relative.max())
    #
    #     return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)


class FER_test(Dataset):
    """
    RAF Facial Expression Dataset structure:
    root/
        |- train/
            |- class1/
                |- img1.jpg
                |- img2.jpg
                |- ...
            |- class2/
            |- ...
        |- val/
            |- class1/
            |- class2/
            |- ...
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query
        self.resize = resize
        self.startidx = startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize))

        # Define transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                self._open_and_convert,
                transforms.Resize((self.resize, self.resize)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                # transforms.RandomHorizontalFlip(p=0.3),
                # transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1,hue=0.02),
                # transforms.RandomRotation(degrees=5),
                # transforms.RandomAffine(
                #     degrees=0,
                #     translate=(0.05, 0.05),  # 轻微平移
                #     scale=(0.95, 1.05)  # 轻微缩放
                # ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                self._open_and_convert,
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        # Load data from directory structure
        self.path = os.path.join(root, mode)
        self.data = []
        self.img2label = {}
        self.cls_names = sorted(os.listdir(self.path))

        for label_idx, cls_name in enumerate(self.cls_names):
            cls_path = os.path.join(self.path, cls_name)
            if os.path.isdir(cls_path):
                # Get all image files in this class directory
                img_files = [f for f in os.listdir(cls_path)
                             if f.lower().endswith(( '.jpg', '.png'))]
                img_paths = [os.path.join(cls_name, f) for f in img_files]
                self.data.append(img_paths)
                # Store mapping from image filename (without extension) to label
                for img_file in img_files:
                    img_name = os.path.splitext(img_file)[0]
                    self.img2label[img_name] = label_idx + self.startidx

        self.cls_num = len(self.data)
        self.create_batch(self.batchsz)

    def _open_and_convert(self, x):
        return Image.open(x).convert('RGB')

    def create_batch(self, batchsz):
        """
        Create batch for meta-learning
        """
        self.support_x_batch = []
        self.query_x_batch = []

        for b in range(batchsz):
            # 1. Select n_way classes randomly
            # remaining = self.n_way - self.cls_num
            # if remaining > 0:
            #     # 从所有类别中随机补充剩余的
            #     selected_cls = list(range(self.cls_num))  # 先包含所有类别
            #     additional = np.random.choice(self.cls_num, remaining, True)
            #     selected_cls.extend(additional)
            # else:
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)

            np.random.shuffle(selected_cls)

            support_x = []
            query_x = []

            for cls in selected_cls:
                # 2. Select k_shot + k_query images from this class
                if len(self.data[cls]) < self.k_shot + self.k_query:
                    # If not enough images, use all available with replacement
                    selected_imgs_idx = np.random.choice(len(self.data[cls]),
                                                         self.k_shot + self.k_query,
                                                         True)
                else:
                    selected_imgs_idx = np.random.choice(len(self.data[cls]),
                                                         self.k_shot + self.k_query,
                                                         False)

                np.random.shuffle(selected_imgs_idx)
                indexDtrain = selected_imgs_idx[:self.k_shot]
                indexDtest = selected_imgs_idx[self.k_shot:]

                support_x.append([self.data[cls][i] for i in indexDtrain])
                query_x.append([self.data[cls][i] for i in indexDtest])

            # Shuffle the corresponding relation between support and query sets
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)

    def __getitem__(self, index):
        # 最多重试次数，避免无限循环
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
                support_y = np.zeros((self.setsz), dtype=np.int32)
                query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
                query_y = np.zeros((self.querysz), dtype=np.int32)

                # Flatten support and query file paths
                flatten_support_x = [os.path.join(self.path, item)
                                     for sublist in self.support_x_batch[index] for item in sublist]

                # Extract labels from filenames (remove extension for img2label mapping)
                support_y = np.array([self.img2label[os.path.splitext(os.path.basename(item))[0]]
                                      for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

                flatten_query_x = [os.path.join(self.path, item)
                                   for sublist in self.query_x_batch[index] for item in sublist]
                query_y = np.array([self.img2label[os.path.splitext(os.path.basename(item))[0]]
                                    for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

                # === 添加严格的类别数量检查 ===
                support_unique = np.unique(support_y)
                query_unique = np.unique(query_y)
                all_unique = np.unique(np.concatenate([support_y, query_y]))

                # 检查1: 确保类别数量正确
                if len(support_unique) != self.n_way:
                    print(f'[ERROR] 任务 {index}: 支持集类别数量错误! 期望 {self.n_way}, 实际 {len(support_unique)}')
                    print(f'  支持集类别: {support_unique}')
                    raise ValueError("支持集类别数量不正确")

                # 检查2: 确保查询集类别是支持集类别的子集
                if not set(query_unique).issubset(set(support_unique)):
                    extra_classes = set(query_unique) - set(support_unique)
                    print(f'[ERROR] 任务 {index}: 查询集包含支持集中没有的类别! {extra_classes}')
                    print(f'  支持集类别: {support_unique}')
                    print(f'  查询集类别: {query_unique}')
                    raise ValueError("查询集包含额外类别")

                # 检查3: 确保总类别数正确
                if len(all_unique) != self.n_way:
                    print(f'[ERROR] 任务 {index}: 总类别数量错误! 期望 {self.n_way}, 实际 {len(all_unique)}')
                    print(f'  支持集类别: {support_unique}')
                    print(f'  查询集类别: {query_unique}')
                    print(f'  总类别: {all_unique}')
                    raise ValueError("总类别数量不正确")

                # Convert to relative labels (0 to n_way-1)
                unique = np.unique(support_y)
                random.shuffle(unique)
                support_y_relative = np.zeros(self.setsz)
                query_y_relative = np.zeros(self.querysz)

                for idx, l in enumerate(unique):
                    support_y_relative[support_y == l] = idx
                    query_y_relative[query_y == l] = idx

                idx2emotion = {int(idx): self.cls_names[int(l)] for idx, l in enumerate(unique)}


                # === 最终验证标签范围 ===
                if support_y_relative.max() >= self.n_way or query_y_relative.max() >= self.n_way:
                    print(f'[ERROR] 任务 {index}: 映射后标签越界!')
                    print(f'  支持集标签范围: {support_y_relative.min()} - {support_y_relative.max()}')
                    print(f'  查询集标签范围: {query_y_relative.min()} - {query_y_relative.max()}')
                    print(f'  允许范围: 0 - {self.n_way - 1}')
                    raise ValueError("映射后标签越界")

                # Load and transform images
                for i, path in enumerate(flatten_support_x):
                    support_x[i] = self.transform(path)

                for i, path in enumerate(flatten_query_x):
                    query_x[i] = self.transform(path)

                # if index >= 100:  # 限制输出范围，避免太多日志
                #     print('[TASK]', index, ' support 全局标签:', np.unique(support_y))
                #     print('[TASK]', index, ' query  全局标签:', np.unique(query_y))
                #     print('[TASK]', index, ' 映射后 query_max:', query_y_relative.max())


                return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative), idx2emotion

            except (ValueError, IndexError) as e:
                retry_count += 1
                print(f'[RETRY] 任务 {index} 第 {retry_count} 次重试，原因: {e}')

                # 如果重试次数用完，返回下一个有效的任务
                if retry_count >= max_retries:
                    print(f'[WARNING] 任务 {index} 重试 {max_retries} 次后仍然失败，跳过该任务')
                    next_index = (index + 1) % len(self)
                    return self.__getitem__(next_index)

                # 更新索引重试
                index = (index + 1) % len(self)

        # 理论上不会执行到这里
        return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return self.batchsz
# class MiniImagenet(Dataset):
#     """
#     put mini-imagenet files as :
#     root :
#         |- images/*.jpg includes all imgeas
#         |- train.csv
#         |- test.csv
#         |- val.csv
#     NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
#     batch: contains several sets
#     sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
#     """
#
#     def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
#         """
#
#         :param root: root path of mini-imagenet
#         :param mode: train, val or test
#         :param batchsz: batch size of sets, not batch of imgs
#         :param n_way:
#         :param k_shot:
#         :param k_query: num of qeruy imgs per class
#         :param resize: resize to
#         :param startidx: start to index label from startidx
#         """
#
#         self.batchsz = batchsz  # batch of set, not batch of imgs
#         self.n_way = n_way  # n-way
#         self.k_shot = k_shot  # k-shot
#         self.k_query = k_query  # for evaluation
#         self.setsz = self.n_way * self.k_shot  # num of samples per set
#         self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
#         self.resize = resize  # resize to
#         self.startidx = startidx  # index label not from 0, but from startidx
#         print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
#         mode, batchsz, n_way, k_shot, k_query, resize))
#
#         if mode == 'train':
#             self.transform = transforms.Compose([self._open_and_convert,
#                                                  transforms.Resize((self.resize, self.resize)),
#                                                  # transforms.RandomHorizontalFlip(),
#                                                  # transforms.RandomRotation(5),
#                                                  transforms.ToTensor(),
#                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                                                  ])
#         else:
#             self.transform = transforms.Compose([self._open_and_convert,
#                                                  transforms.Resize((self.resize, self.resize)),
#                                                  transforms.ToTensor(),
#                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                                                  ])
#
#         self.path = os.path.join(root, 'images')  # image path
#         csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path
#         self.data = []
#         self.img2label = {}
#         for i, (k, v) in enumerate(csvdata.items()):
#             self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
#             self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
#         self.cls_num = len(self.data)
#
#         self.create_batch(self.batchsz)
#
#     def _open_and_convert(self, x):
#         return Image.open(x).convert('RGB')
#
#     def loadCSV(self, csvf):
#         """
#         return a dict saving the information of csv
#         :param splitFile: csv file name
#         :return: {label:[file1, file2 ...]}
#         """
#         dictLabels = {}
#         with open(csvf) as csvfile:
#             csvreader = csv.reader(csvfile, delimiter=',')
#             next(csvreader, None)  # skip (filename, label)
#             for i, row in enumerate(csvreader):
#                 filename = row[0]
#                 label = row[1]
#                 # append filename to current label
#                 if label in dictLabels.keys():
#                     dictLabels[label].append(filename)
#                 else:
#                     dictLabels[label] = [filename]
#         return dictLabels
#
#     def create_batch(self, batchsz):
#         """
#         create batch for meta-learning.
#         ×episode× here means batch, and it means how many sets we want to retain.
#         :param episodes: batch size
#         :return:
#         """
#         self.support_x_batch = []  # support set batch
#         self.query_x_batch = []  # query set batch
#         for b in range(batchsz):  # for each batch
#             # 1.select n_way classes randomly
#             selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
#             np.random.shuffle(selected_cls)
#             support_x = []
#             query_x = []
#             for cls in selected_cls:
#                 # 2. select k_shot + k_query for each class
#                 selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
#                 np.random.shuffle(selected_imgs_idx)
#                 indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
#                 indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
#                 support_x.append(
#                     np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
#                 query_x.append(np.array(self.data[cls])[indexDtest].tolist())
#
#             # shuffle the correponding relation between support set and query set
#             random.shuffle(support_x)
#             random.shuffle(query_x)
#
#             self.support_x_batch.append(support_x)  # append set to current sets
#             self.query_x_batch.append(query_x)  # append sets to current sets
#
#     def __getitem__(self, index):
#         """
#         index means index of sets, 0<= index <= batchsz-1
#         :param index:
#         :return:
#         """
#         # [setsz, 3, resize, resize]
#         support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
#         # [setsz]
#         support_y = np.zeros((self.setsz), dtype=np.int32)
#         # [querysz, 3, resize, resize]
#         query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
#         # [querysz]
#         query_y = np.zeros((self.querysz), dtype=np.int32)
#
#         flatten_support_x = [os.path.join(self.path, item)
#                              for sublist in self.support_x_batch[index] for item in sublist]
#         support_y = np.array(
#             [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
#              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
#
#         flatten_query_x = [os.path.join(self.path, item)
#                            for sublist in self.query_x_batch[index] for item in sublist]
#         query_y = np.array([self.img2label[item[:9]]
#                             for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
#
#         # print('global:', support_y, query_y)
#         # support_y: [setsz]
#         # query_y: [querysz]
#         # unique: [n-way], sorted
#         unique = np.unique(support_y)
#         random.shuffle(unique)
#         # relative means the label ranges from 0 to n-way
#         support_y_relative = np.zeros(self.setsz)
#         query_y_relative = np.zeros(self.querysz)
#         for idx, l in enumerate(unique):
#             support_y_relative[support_y == l] = idx
#             query_y_relative[query_y == l] = idx
#
#         # print('relative:', support_y_relative, query_y_relative)
#
#         for i, path in enumerate(flatten_support_x):
#             support_x[i] = self.transform(path)
#
#         for i, path in enumerate(flatten_query_x):
#             query_x[i] = self.transform(path)
#         # print(support_set_y)
#         # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
#
#         return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
#
#     def __len__(self):
#         # as we have built up to batchsz of sets, you can sample some small batch size of sets.
#         return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()
