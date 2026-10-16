from typing import Dict

from .filesystem import *
from .preprare import prepare_signal

import os
import json
import pandas
import numpy as np

import torch
from tqdm import tqdm

SHORT_CONSTRICTIVES = [1, 2, 3]

def read_data(filename, dt:np.dtype=np.int32) -> np.ndarray:
    """Читает данные из файла в указанном формате и возвращает их в форме (channels,length)"""
    with open(filename, "r", encoding="utf-8") as f:
        data = np.loadtxt(
            f,
            dtype=dt,
            usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        )
        (x, y) = data.shape
        true_data = np.ndarray((y, x), dtype=dt)
        true_data[:, :] = data.T
        return true_data

def delete_empty(labels_dir: str, images_dir: str, max_num: int, prefix: str, extension: str):
    """
        Удаляет картинки на которых нет отметок
    """
    dir_exists(labels_dir)
    dir_exists(images_dir)
    
    for number in range(0,max_num + 1):
        for left in [True,False]:
            name = "%s_%s_%d." % (prefix, "left"if left else "right", number)
            label = os.path.join(labels_dir, f"{name}txt")
            image = os.path.join(images_dir, f"{name}{extension}")
            if not os.path.exists(label):
                if os.path.exists(image):
                    os.remove(image)

class D1LabelsConverter:
    
    def __init__(self):
        pass

    def back_convert(self, box:Tuple[float,float], data_size: int) -> Tuple[int,int]:
        """ 
            Преобразует отометку YOLO в координату отметки в блоке анализа. (begin, end) 
            (centr_x, width) -> (x1,x2) 
        """
        x1 = (box[0] - box[1]) / 2 / data_size
        x2 = (box[0] + box[1]) / 2 / data_size
        return x1,x2
        
    def forward_convert(self, box:Tuple[int,int], data_size: int)->Tuple[float,float]:
        """ Преобразует отметку в блоке анализа в отметку для 1d сети
            (x1,x2) -> (centr_x, width)
        """
        x = (box[0] + box[1]) / 2 / data_size
        w = (box[1] - box[0]) / data_size
        return x, w
    


def convert_labels(labels_df, out_dir: str, reg_name:str, 
                   left_size: Dict[int,int], right_size: Dict[int,int], converter):
    """
        Преоброазует метки в формат YOLO для соответствующих картинок.
        
        converter
            Конвертер отметок.
        left_size
            Словарь с размерами 
        
        only_types 
            Типы отметок, которые переносятся в разметку. (Остальные отбрасываются.) 
            По умолчанию установлены Алюмотермитные и Электроконтактные сварки и Рельсовые стыки
    """
    
    def get_size(block_number,is_left):
        return left_size.get(block_number, 50000) if is_left else right_size.get(block_number, 50000)
    
    for index, row in labels_df.iterrows():
        fname = os.path.join(out_dir, reg_name % ("left" if row["is_left"] else "right", row["block_number"]))
        with open(fname, "w") as f:
            for cls, begin, end in zip(row["type"], row["begin"], row["end"]):
                x,w = converter.forward_convert((begin, end), get_size(row["block_number"], row["is_left"]))
                if x > 1 or w < 0:
                    print(f"{row['block_number']} {row['is_left']} wrong mark")
                    continue
                f.write("%d %.6f %.6f\n" %(cls, x, w))

def prepare_labels(labels_file:str, only_types:List[int], only_containing_type:List[int]):

    # Конвертируем метки
    with open(labels_file, "r") as f:
        labels = json.load(f)
    df = pandas.DataFrame(labels)
    # фильтруем 
    df = df[df['type'].isin(only_types)]
    # фиксированная рамка 200 mm
    short = df["type"].isin(SHORT_CONSTRICTIVES)
    center = (df[short]["begin"] + df[short]["end"]) / 2
    df[short]["begin"] = center - 100
    df[short]["end"]   = center + 100
    # фильтруем те блоки анализа, где есть указанные объекты
    if only_containing_type is not None:
        # получили Блоки, где есть желаемые объекты
        filter_indexes = df[df["type"].isin(only_containing_type)].loc[:,["block_number","is_left"]]
        # Удалили дубликаты
        filter_indexes = filter_indexes.drop_duplicates()
        # выбрали подвыборку
        df = pandas.merge(filter_indexes, df, how="left", left_on=["block_number", "is_left"],
                                                    right_on=["block_number", "is_left"]) 
    # Меняем номера классов
    type_mapping = {old: new for new, old in enumerate(only_types)}
    df['type'] = df['type'].map(type_mapping)
    # Сгруппировали объекты по блокам анализа
    df = df.groupby(["block_number", "is_left"]).agg({
        "type": list,
        "begin": list,
        "end": list}).reset_index()
    return df

def get_1d_data(data_dir: str, labels_file: str, output_dir: str, only_types: List[int]=[1,2,3],
                name_prefix: str = "", save_empty:bool=False, 
                use_furier:bool=True, only_containing_type:List[int]=None):
    
    # Проверки файловой системы
    dir_exists(data_dir)
    file_exists(labels_file)
    create_dir(output_dir)
    
    labels_dir = os.path.join(output_dir,"labels")
    images_dir = os.path.join(output_dir,"data")
    create_dir(labels_dir)
    create_dir(images_dir)
    
    left_sizes = dict()
    right_sizes = dict()
     
    labels_df = prepare_labels(labels_file, only_types, only_containing_type)
    
    indexes = set(zip(labels_df['block_number'], labels_df['is_left']))
    
    max_number = 0
    print(f"Processing {name_prefix}")
    for file in tqdm(os.listdir(data_dir)):
        if not is_data_file(file):
            continue
        full_file_path = os.path.join(data_dir,file)
        # Получаем информацию о файле
        left, number = get_file_info(file)
        if (number, left) not in indexes and not save_empty: 
            continue
        max_number = max(max_number,number)
        # Читаем и предобрабатываем данные
        raw_data = read_data(full_file_path)
        # Запоминаем размеры файлов
        if left:
            left_sizes[number] = raw_data.shape[1]
        else:
            right_sizes[number] = raw_data.shape[1]
        # ВЫравнивание и нахождение пороговых уровней
        signal = prepare_signal(raw_data, furier=use_furier, SW=True)
        # Масштабирование данных по пороговому уровню
        data = np.divide(signal.data, signal.sigma[:, None], dtype=np.float32)
        raw_data = np.divide(raw_data, signal.sigma[:, None], dtype=np.float32)
        data = np.concatenate([raw_data, data], axis=0)
        # data = np.astype(data, np.float16)
        data = data.astype(np.float16)
        # сохраняем файлы
        data_name = "%s_%s_%d.pt" % (name_prefix, ("left" if left else "right"), number)
        data_full_name = os.path.join(images_dir, data_name)
        with open(data_full_name, "wb") as f:
            tensor = torch.from_numpy(data)
            torch.save(tensor, f)

    
    # шаблон имени, куда можно подставить сторону и номер файла
    reg_name = f"{name_prefix}_%s_%d.txt"
    
    convert_labels(labels_df, labels_dir, reg_name, left_sizes, right_sizes, D1LabelsConverter())
    if not save_empty:
        delete_empty(labels_dir, images_dir, max_number, name_prefix, extension="pt")

    