from typing import Tuple
import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

import yaml

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

class Dataset(data.Dataset):
    
    def __init__(self, filename, input_size, params, augment, is_test=False):
        super().__init__()
        self.is_test = is_test
        self.input_size = input_size
        self.params = params
        self.augment = augment
        self.filename = filename
        # Загружаем конфигурацию датасета
        self._parse_yaml()
        # Загружаем пути к данным
        self._load_data()
        # Загружаем отметки
        self._load_labels()
        
        self.n = len(self.filenames)    # Количество дынных
        self.indices = range(self.n)    # Индексы данных
        
    def _parse_yaml(self):
        """ 
            Считывает их файла настроек:
            - путь к данным и разметке
            - Информацию о названиях классов и их количестве
        """
        data = yaml.safe_load(open(self.filename,encoding="utf-8")) # Парсим yaml - файл
        
        # Загружаем названия классов
        self.class_names = [i for i in data["names"].values()]
        # Считаем количество классов
        self.num_classes = len(self.class_names)
        # Считываем путь к данным
        self.data_path = data["val_data"] if self.is_test else data["train_data"]
        # Считываем путь к разметке
        self.labels_path = data["val_labels"] if self.is_test else data["train_labels"]
        
    def _load_data(self):
        """
            Записывает все имена файлов данных в список.
        """
        self.filenames = []
        for file in os.listdir(self.data_path):
            self.filenames.append(os.path.join(self.data_path, file))
        
    def _load_labels(self):
        """
        Загружает разметку из кэша или создаёт кэш, если его нет.
        """
        # Загружаем кэш
        path = os.path.join(self.labels_path, "labels.cache")
        if os.path.exists(path):
            self.labels = torch.load(path, weights_only=False)
            return 
        # Не нашли кэш
        dict = {}
        for filename in self.filenames:
            try:
                #  Получаем имя файла
                name = os.path.basename(filename).rsplit(".", 1)[0] + ".txt"
                label_file = os.path.join(self.labels_path, name)
                # Читаем отметки
                if os.path.isfile(label_file):
                    with open(label_file) as f:
                        # Разбиваем строки
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()   # Проврка неотрицательности
                        assert label.shape[1] == 3  # 3 части: класс, центр, ширина
                        assert (label[:, 1:] <= 1).all() # обьект не выходит за границы
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # Проверка дубликатов
                            label = label[i]  # Удаление дубликатов
                    else:
                        label = numpy.zeros((0, 3),dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 3),dtype=numpy.float32)
            except FileNotFoundError:
                label = numpy.zeros((0, 3), dtype=numpy.float32)
            except AssertionError:
                continue
            # Добавляем отметку 
            dict[filename] = label
        # Кэшируем разметку
        torch.save(dict, path)
        self.labels = dict
        return
    
    def load_image(self, i):
        with open(self.filenames[i], "rb") as f:
            loaded_binary = f.read()
        data = numpy.frombuffer(loaded_binary, dtype=numpy.float32).reshape(15, -1)
        w = data.shape[-1]
        return data, (15, w)
    
    def __getitem__(self, index):
        """
            Реализация метода для получения объекта из датасета
            Важно:
            Наши данные - 50000
            Сеть принимает 49152=(2^16)*3
            Можно просто взять интерполяцию, а можно взять случайный сдвиг и обрезку!
            
        """
        index = self.indices[index]
        
        data, shape = self.load_image(index)
        w = data.shape[-1]
        
        # Resize
        data, ratio = resize(data, self.input_size, self.augment)

        label = self.labels[self.filenames[index]].copy()
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio[1])
        # if self.augment:
        #     image, label = random_perspective(image, label, self.params)

        nl = len(label)  # number of labels
        w = data.shape[-1]
        cls = label[:, 0:1]
        box = label[:, 1:3]
        box = xy2wh(box, w)

        # if self.augment:
        #     # Albumentations
        #     data, box, cls = self.albumentations(data, box, cls)
        #     nl = len(box)  # update after albumentations
        #     # HSV color-space
        #     augment_hsv(data, self.params)
        #     # Flip up-down
        #     if random.random() < self.params['flip_ud']:
        #         data = numpy.flipud(data)
        #         if nl:
        #             box[:, 1] = 1 - box[:, 1]
        #     # Flip left-right
        #     if random.random() < self.params['flip_lr']:
        #         data = numpy.fliplr(data)
        #         if nl:
        #             box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 2))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        data = numpy.ascontiguousarray(data)

        return torch.from_numpy(data), target_cls, target_box, torch.zeros(nl)

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

class DatasetTorch(Dataset):
    
    def load_image(self, i) -> Tuple[torch.Tensor, Tuple]:
        with open(self.filenames[i], "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=False)
        return data.float(), data.size()
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Реализация метода для получения объекта из датасета
            Важно:
            Наши данные - 50000
            Сеть принимает 49152=(2^16)*3
            Можно просто взять интерполяцию, а можно взять случайный сдвиг и обрезку!
            
        """
        index = self.indices[index]
        
        data, shape = self.load_image(index)
        
        # Resize
        data, ratio = resize_tensor(data, self.input_size, self.augment)

        label = self.labels[self.filenames[index]].copy()
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio[1])
        # if self.augment:
        #     image, label = random_perspective(image, label, self.params)

        nl = len(label)  # number of labels
        w = shape[-1]
        cls = label[:, 0:1]
        box = label[:, 1:3]
        box = xy2wh(box, w)

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 2))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        return data, target_cls, target_box, torch.zeros(nl)
    

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

def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)

def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed

def interpolate(data: numpy.ndarray, input_size):
    """Интерполяция данных до указанного размера по количеству столбцов"""
    original_indices = numpy.linspace(0, 1, data.shape[-1])
    new_indices = numpy.linspace(0, 1, input_size)

    # Линейная интерполяция
    return numpy.array([numpy.interp(new_indices, original_indices, d) for d in data])

def interpolate_tensor(data: torch.Tensor, input_size) -> torch.Tensor:
    """Интерполяция данных до указанного размера по количеству столбцов"""
    # Линейная интерполяция
    return torch.nn.functional.interpolate(data.unsqueeze(0), input_size, mode='linear').squeeze(0)

def zero_pad(data: numpy.ndarray, input_size):
    """Дополнение матрицы нулями до указанного размера по количеству столбцов"""
    if data.shape[1] >= input_size:
        return data
    result = numpy.zeros((data.shape[0],input_size),dtype=data.dtype)
    result[:,:data.shape[1]] = data
    return result

def zero_pad_tensor(data: torch.Tensor, input_size):
    """Дополнение матрицы нулями до указанного размера по количеству столбцов"""
    if data.size(1) >= input_size:
        return data
    return torch.nn.functional.pad(
        data, 
        pad=(0,  input_size - data.size(1)),  # (padding_left, padding_right)
        mode='constant', 
        value=0
    )




def resize(data: torch.Tensor, input_size, augment) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[float, float]]:
    """ Изменение размера матрицы до размера input_size.
        Результат тензор и две пары:\\
        (begin1, end1) - отрезок данных, который отображёно в следующий интервал\\
        (begin2, end2) - отрезок данных, в который были отображены данные
    """
    const_size = 50000
    # length of signal
    length = data.size(1) 
    if augment:    
        # length difference for random resize
        a = 0.1
        # random resize
        augment_length = length * numpy.random.uniform(1 - a, 1 + a)
        data = interpolate_tensor(data, augment_length)
        if augment_length <= input_size:
            # if data fully fit
            pos = numpy.random.randint(0, input_size - augment_length)
            result = torch.tensor((data.size(0), input_size))
            result[pos:augment_length] = data
            return result, (0, length), (pos, pos + augment_length)
        else:
            # data is larger, need to crop
            pos = numpy.random.randint(0, augment_length - input_size)
            result = torch.tensor((data.size(0), input_size))
            result[:] = data[pos: pos + input_size]
            scale = length / augment_length
            return result, (pos*scale, (pos+input_size) * scale), (0, input_size)
    else:
        
    
    # Дополняем нулями до 50 метров
    if shape < const_size:
        data = zero_pad(data, const_size)
    # коэффициент сжатия данных
    r = input_size / const_size
    if not augment:
        # Интерполяция до желаемого размера
        return interpolate(data, input_size), (,), (0, shape * r)
    # Можно сделать случайную обрезку для аугментации
    diff = shape - input_size
    begin = numpy.random.randint(0, diff)
    end = numpy.random.randint(begin + input_size, shape)
    return interpolate(data[:,begin:end], input_size), (begin, end - begin)

def resize_tensor(data, input_size, augment):
    """Изменение размера матрицы до размера input_size"""
    const_size = 50000
    
    shape = data.size(1)  # длина данных
    # Дополняем нулями до 50 метров
    if shape < const_size:
        data = zero_pad_tensor(data, const_size)
    # коэффициент сжатия данных
    r = input_size / const_size
    # Интерполяция до желаемого размера
    return interpolate_tensor(data, input_size), (0, shape * r)

def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(label)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        box = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
        # filter candidates
        indices = candidates(box1=label[:, 1:5].T * s, box2=box.T)

        label = label[indices]
        label[:, 1:5] = box[indices]

    return image, label


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    alpha = numpy.random.beta(a=32.0, b=32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations

            transforms = [albumentations.Blur(p=0.01),
                          albumentations.CLAHE(p=0.01),
                          albumentations.ToGray(p=0.01),
                          albumentations.MedianBlur(p=0.01)]
            self.transform = albumentations.Compose(transforms,
                                                    albumentations.BboxParams('yolo', ['class_labels']))

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box,
                               class_labels=cls)
            image = x['image']
            box = numpy.array(x['bboxes'])
            cls = numpy.array(x['class_labels'])
        return image, box, cls
