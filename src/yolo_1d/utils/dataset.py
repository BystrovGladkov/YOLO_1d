from typing import Tuple, Literal
import math
import os
import random

import numpy
import torch
from torch.utils import data

import yaml

class Dataset(data.Dataset):
    
    def __init__(self, filename, transform, type: Literal["train", "val", "test"],
                 mosaic_prob=0,
                 mix_up_prob=0,
                 mix_up_alpha=20):
        super().__init__()
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.mix_up_prob = mix_up_prob
        self.mix_up_alpha = mix_up_alpha
        self.filename = filename
        self.type = type
        # Загружаем конфигурацию датасета
        self._parse_yaml()
        # Загружаем пути к данным
        self._load_data()
        # Загружаем отметки
        self._load_labels()
        
        self.n = len(self.filenames)    # Количество данных
        self.indices = range(self.n)    # Индексы данных
        
    def _parse_yaml(self):
        """ 
            Считывает их файла настроек:
            - путь к данным и разметке
            - Информацию о названиях классов и их количестве
        """
        data = yaml.safe_load(open(self.filename, encoding="utf-8")) # Парсим yaml - файл
        
        # Загружаем названия классов
        self.class_names = [i for i in data["names"].values()]
        # Считаем количество классов
        self.num_classes = len(self.class_names)
        # Считываем путь к данным
        self.data_path = data[f"{self.type}_data"]
        # Считываем путь к разметке
        self.labels_path = data[f"{self.type}_labels"]
        
    def _load_data(self):
        """
            Saves filenames of files with data.
        """
        self.filenames = []
        for file in os.listdir(self.data_path):
            self.filenames.append(os.path.join(self.data_path, file))
        
    def _load_labels(self):
        """
            Loads labels from cache and creates cache if it does not exist.
        """
        # loading cache
        path = os.path.join(self.labels_path, "labels.cache")
        if os.path.exists(path):
            self.labels = torch.load(path, weights_only=False)
            return 
        # cache not found
        dict = {}
        for filename in self.filenames:
            try:
                # get filenames
                name = os.path.basename(filename).rsplit(".", 1)[0] + ".txt"
                label_file = os.path.join(self.labels_path, name)
                # read labels
                if os.path.isfile(label_file):
                    with open(label_file) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        # simple asserts,
                        # TODO: better no to raise error but fix problem and show message
                        assert (label >= 0).all()   # all parts of label must be positive
                        assert label.shape[1] == 3  # 3 parts: class, center, width
                        assert (label[:, 1:] <= 1).all() # check labels not exceed corners
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # check for duplicates
                            label = label[i]  # delete duplicates
                            # TODO: show amount of duplicates deleted
                    else:# no labels
                        label = numpy.zeros((0, 3),dtype=numpy.float32)
                else: # labels file not found (not a file)
                    label = numpy.zeros((0, 3),dtype=numpy.float32)
            except FileNotFoundError: # labels file not found
                label = numpy.zeros((0, 3), dtype=numpy.float32)
            except AssertionError:
                continue
            # save labels
            dict[filename] = label
        # save cache
        torch.save(dict, path)
        self.labels = dict
        return
    
    def load_signal(self, i) -> torch.Tensor:
        """
            Reads signal with index i from dataset.

            Returns
            ---
            signal
                Signal from dataset with index i
        """
        with open(self.filenames[i], "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=False)
        return data.float()
    
    def load_mosaic(self, index):
        # resulting labels
        labels = []
        # resulting signal
        signals = []
        # select second signal
        indices = [index] + random.choices(self.indices, k=1)
        # shuffle signals
        random.shuffle(indices)
        
        length = []
        for i, index in enumerate(indices):
            signal = self.load_signal(index)
            # save signal
            signals.append(signal)
            # save length of signal
            length.append(signal.size(1))
            # save labels
            labels.append(self.labels[self.filenames[index]].copy())
        
        length = numpy.array(length)
        full_length = length.sum()
        offset = numpy.concatenate([[0], length.cumsum()[:-1]])
        
        for index in range(0, length.shape[0]):
            # calculate a coefficient to convert labels
            scale = length[index] / full_length
            # calculate center of labels
            labels[index][:, 1] = offset[index] / full_length + labels[index][:, 1] * scale
            # calculate width of labels
            labels[index][:, 2] = labels[index][:, 2] * scale
            
        labels = numpy.concatenate(labels, axis=0)
        signals = torch.concat(signals, dim=1)
        
        return signals, labels
    
    def load_mix_up(self, data1, target_box1, target_cls1):
        # load another signal
        index = random.choices(self.indices, k=1)[0]
        # load signal
        data2 = self.load_signal(index)
            # load labels
        label = self.labels[self.filenames[index]].copy()

        nl = len(label)  # number of labels
        cls = label[:, 0:1]
        box = label[:, 1:3]
        # convert labels to torch tensors
        target_cls2 = torch.zeros((nl, 1))
        target_box2 = torch.zeros((nl, 2))
        if nl:
            target_cls2 = torch.from_numpy(cls)
            target_box2 = torch.from_numpy(box)
        # perform transform
        data2, target_box2, target_cls2 = self.transform.forward(data2, target_box2, target_cls2)
        
        lamb = float(numpy.random.beta(self.mix_up_alpha, self.mix_up_alpha, 1))
        # mix the signals
        data1 = data1 * lamb + data2 * (1 - lamb)
        target_box1 = torch.concat([target_box1, target_box2], axis=0)
        target_cls1 = torch.concat([target_cls1, target_cls2], axis=0)
        return data1, target_box1, target_cls1
        
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Extracts signal and labels from dataset.
        """
        index = self.indices[index]
        
        # load mosaic signal
        if self.mosaic_prob and random.random() < self.mosaic_prob:
            data, label = self.load_mosaic(index)
        else:
            # load signal
            data = self.load_signal(index)
            # load labels
            label = self.labels[self.filenames[index]].copy()

        nl = len(label)  # number of labels
        cls = label[:, 0:1]
        box = label[:, 1:3]
        # convert labels to torch tensors
        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 2))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)
        # perform transform
        data, target_box, target_cls = self.transform.forward(data, target_box, target_cls)
        
        if self.mix_up_prob and random.random() < self.mix_up_prob:
            data, target_box, target_cls = self.load_mix_up(data, target_box, target_cls)
        
        nl = target_box.size(0)

        return data, target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return self.n

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)

        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'cls': cls,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets



def wh2xy(x, w=49152):
    """ from [x, w] normalized to [x1, x2]"""
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 1] / 2) # left
    y[:, 1] = w * (x[:, 0] + x[:, 1] / 2) # right
    return y

def xy2wh(x, w):
    """from [x1, x2] to [x, w]"""
    # warning: inplace clip
    x[:, [0, 1]] = x[:, [0, 1]].clip(0, w - 1E-3)  # x1, x2
    
    # Convert nx4 boxes
    # from [x1, x2] to [x, w]
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 1]) / 2) / w  # center
    y[:, 1] = (x[:, 1] - x[:, 0]) / w  # width
    return y
