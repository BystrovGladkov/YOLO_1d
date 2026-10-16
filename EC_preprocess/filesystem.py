from typing import Tuple, List
import os
from re import match,compile

def dir_exists(filename: str) -> None:
    if not os.path.exists(filename) or not os.path.isdir(filename):
        raise Exception(f"Directory not exist: {filename}.")
    
def file_exists(filename: str) -> None:
    if not os.path.exists(filename) or not os.path.isfile(filename):
        raise Exception(f"File not exist: {filename}.")
    
def create_dir(filename: str) -> None:
    if os.path.exists(filename):
        if not os.path.isdir(filename):
            raise Exception(f"Path already exists: {filename}.")
        else:
            pass
    else:
        os.makedirs(filename,exist_ok=True)
            
def is_data_file(filename: str) -> bool:
    correct = r"(left|right)_[0-9]*.txt"
    return (match(correct,filename) is not None)

def get_file_info(filename: str) -> Tuple[bool,int]:
    strs = filename.split(".")[0].split("_")
    is_left = strs[0] == "left"
    number = int(strs[1])
    return (is_left, number)


class FileInfo:
    def __init__(self, filename: str, full_path: str, is_left:bool, block_number:int, number:int):
        self.filename = filename
        self.full_path = full_path
        self.is_left = is_left
        self.block_number = block_number
        self.number = number
        
def extract_data_from_file(name:str) -> Tuple[str, bool, int, int]:
    """
        Извлекает данные из названия файла
        Формат: `{prefix}_{side}_{block_number}_{number}.*`
    """
    # 1. Определяем регулярное выражение с именованными группами захвата.
    #    Именованные группы делают код более читаемым и удобным в обслуживании.
    pattern = compile(
        r"^(?P<prefix>[^_]+)_(?P<side>[^_]+)_(?P<block_number>[^_]+)_(?P<number>[^_]+)\.(?P<extension>[a-zA-Z0-9]+)$"
    )

    # 2. Пытаемся сопоставить шаблон с входной строкой.
    match = pattern.match(name)

    # 3. Проверяем, было ли найдено соответствие.
    if match:
        # 4. Извлекаем параметры по именам групп. Это более читаемо, чем по индексам.
        params = match.groupdict()  # Возвращает словарь с именами групп и их значениями.

        prefix = params['prefix']
        is_left = True if params['side'] == "left" else False
        block_number = params['block_number']
        number = params['number']

        # 5. Дальнейшая обработка извлеченных параметров (например, преобразование типов).
        try:
            block_number = int(block_number)  # Преобразуем block_number в целое число.
            number = int(number) # Преобразуем number в целое число.
        except ValueError:
            return None

        # 6. Выводим извлеченные и обработанные параметры.
        return (prefix, is_left, block_number, number)
    else:
        return None

def extract_data_from_file_no_number(name:str) -> Tuple[str, bool, int]:
    """
        Извлекает данные из названия файла
        Формат: `{prefix}_{side}_{block_number}.*`
    """
    # 1. Определяем регулярное выражение с именованными группами захвата.
    #    Именованные группы делают код более читаемым и удобным в обслуживании.
    pattern = compile(
        r"^(?P<prefix>[^_]+)_(?P<side>[^_]+)_(?P<block_number>[^_]+)\.(?P<extension>[a-zA-Z0-9]+)$"
    )

    # 2. Пытаемся сопоставить шаблон с входной строкой.
    match = pattern.match(name)

    # 3. Проверяем, было ли найдено соответствие.
    if match:
        # 4. Извлекаем параметры по именам групп. Это более читаемо, чем по индексам.
        params = match.groupdict()  # Возвращает словарь с именами групп и их значениями.

        prefix = params['prefix']
        is_left = True if params['side'] == "left" else False
        block_number = params['block_number']

        # 5. Дальнейшая обработка извлеченных параметров (например, преобразование типов).
        try:
            block_number = int(block_number)  # Преобразуем block_number в целое число.
        except ValueError:
            return None

        # 6. Выводим извлеченные и обработанные параметры.
        return (prefix, is_left, block_number)
    else:
        return None

def enumerate_txts(path:str) -> List[FileInfo]:
    """Считывает файлы из указанного каталога и извлекает данные из названия файла """
    files: List[FileInfo] = []
    for item in os.listdir(path):
        if not is_data_file(item):
            continue
        info = get_file_info(item)
        if info is not None:
            files.append(FileInfo(item, os.path.join(path, item), info[0], info[1], 0))
    files.sort(key = lambda x: (x.block_number, x.is_left))
    return files

def enumerate_images(path:str) -> List[FileInfo]:
    """Считывает файлы из указанного каталога и извлекает данные из названия файла """
    files: List[FileInfo] = []
    for item in os.listdir(path):
        info = extract_data_from_file(item)
        if info is not None:
            files.append(FileInfo(item, os.path.join(path, item), info[1], info[2], info[3]))
    files.sort(key = lambda x: (x.block_number, x.is_left, x.number))
    return files

def erm_name_prefix(name: str):
    """ Преобразует имя erm в префикс. Берёт имя дороги (до первой ~) и odo начала"""
    parts = name.split("~")
    rail_name = parts[0].lower()
    odo = parts[5].split(".")[0]
    date = parts[3]
    return "~".join([rail_name, odo, date])