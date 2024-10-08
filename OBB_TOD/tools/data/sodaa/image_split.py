import os
import shutil
import json

# 当前原始数据集目录
base_dir = '/data/zhr/SODA/SODA-A/divData/test/Annotations/'
anno_files_name = []
anno_files = []
anno_files_absolute = []
anno_dest_path = '/data/zhr/SODA/SODA-A/divData/test/Annotations_filter/'
for file in os.listdir(base_dir):
    absolute_path = os.path.join(base_dir, file)
    if len(json.load(open(absolute_path, 'r'))['annotations']) == 0:
        continue

    anno_files_name.append(file.split('.')[0])
    anno_files.append(file)
    anno_files_absolute.append(absolute_path)
    dst_path = os.path.join(anno_dest_path, file)
    shutil.copy(absolute_path, dst_path)

img_base_path = '/data/zhr/SODA/SODA-A/divData/test/Images/'
img_dest_path = '/data/zhr/SODA/SODA-A/divData/test/Images_filter/'
for file in os.listdir(img_base_path):
    absolute_path = os.path.join(img_base_path, file)
    name = file.split('.')[0]
    if name in anno_files_name:
        dst_path = os.path.join(img_dest_path, file)
        shutil.copy(absolute_path, dst_path)








# dest_dir = '/data/zhr/SODA/SODA-A/images/test'
# raw_image_dir = '/data/zhr/SODA/SODA-A/Images'

# # 获取当前目录下的所有文件
# anno_files = [file.split('.')[0] for file in os.listdir(base_dir)]

# images_files = []
# images_name = []
# images_raw_name = []
# for file in os.listdir(raw_image_dir):
#     images_files.append(os.path.join(raw_image_dir, file))
#     images_name.append(file.split('.')[0])
#     images_raw_name.append(file)

# for i in range(len(images_name)):
#     img_name = images_name[i]
#     img_path = images_files[i]
#     img_raw_name = images_raw_name[i]
#     if img_name in anno_files:
#         ori_path = img_path
#         dst_path = os.path.join(dest_dir, img_raw_name)
#         print('ori path: ', ori_path)
#         print('dst path: ', dst_path)
#         shutil.copy(ori_path, dst_path)
#         print('----------------------')