import logging
import os
import random
from PIL import ImageFile
from PIL import Image
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_logger(save_path, task):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(task)
    handler = logging.FileHandler(save_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def image_random_crop(image, prob=0.7):
    if random.random() < prob:
        w, h = image.size
        left = random.randint(0, w // 2)
        right = random.randint(min(left + 224, w), w)
        top = random.randint(0, h // 2)
        bottom = random.randint(min(top + 224, h), h)
        try:
            return image.crop((left, top, right, bottom))
        except Exception as e:
            print(e)
            return image
    else:
        return image


# 获取文件夹下的所有图片路径
def get_image_paths(img_dir_path):
    img_paths = []
    for root, dirs, files in os.walk(img_dir_path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('png'):
                img_paths.append(os.path.join(root, file))
    return img_paths


# 获取文件夹下的PDF路径
def get_pdf_paths(pdf_dir_path, pdf_ext='.pdf'):
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_dir_path):
        for file in files:
            if file.lower().endswith(pdf_ext):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def clean_img(img_dir, logger):
    if logger is None:
        print(f'Cleaning images in {img_dir}')
    else:
        logger.info(f'Cleaning images in {img_dir}')
    del_count = 0
    convert_count = 0
    img_paths = get_image_paths(img_dir)
    for img_path in tqdm(img_paths):
        try:
            img = Image.open(img_path)
            # if img.mode == 'RGBA':
            #     print(f'{img_path} is RGBA')
            #     img = img.convert('RGB')
            #     img.save(img_path)
            #     convert_count += 1
        except Exception as e:
            print(f'{img_path} open error')
            os.remove(img_path)
            del_count += 1
    if logger is None:
        print(f'{del_count} images deleted, {convert_count} images converted')
    else:
        logger.info(f'{del_count} images deleted, {convert_count} images converted')
