from PIL import Image
import os

path = 'C:/Users/hp/Desktop/1'  # 文件目录
# path这个目录处理完之后需要手动更改
path_list = os.listdir(path)
print(path_list)

for i in path_list:  # 截左半张图片
    a = open(os.path.join(path, i), 'rb')
    img = Image.open(a)
    w = img.width  # 图片的宽
    h = img.height  # 图片的高
    print('正在处理图片', i, '宽', w, '长', h)

    box = (0, 0, w * 0.5, h)  # box元组内分别是 所处理图片中想要截取的部分的 左上角和右下角的坐标
    img = img.crop(box)
    print('正在截取左半张图...')
    img.save('L' + i)  # 这里需要对截出的图加一个字母进行标识，防止名称相同导致覆盖
    print('L-', i, '保存成功')

for i in path_list:  # 截取右半张图片
    a = open(os.path.join(path, i), 'rb')
    img = Image.open(a)
    w = img.width  # 图片的宽
    h = img.height  # 图片的高
    print('正在处理图片', i, '宽', w, '长', h)

    box = (w * 0.5, 0, w, h)
    img = img.crop(box)
    print('正在截取右半张图...')
    img.save('R' + i)
    print('R-', i, '保存成功')

print("'{}'目录下所有图片已经保存到本文件目录下。".format(path))


