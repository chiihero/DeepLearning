import multiprocessing
import os
from PIL import Image
from PIL import ImageFile
import tqdm
from PIL.Image import EXTENT
ImageFile.LOAD_TRUNCATED_IMAGES = True
input = "D:\\photo\\temp\\"
# outputcolor ="D:\\photo\\temp\\color\\"
# outputbw ="D:\\photo\\temp\\black&white\\"
outputcolor = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\color\\"
outputbw = "E:\\Chii\\百度云\\GitHub\\DeepLearning\\PictureColoring\\photo\\black&white\\"


def c2bw(inputfile):
    try:
        img = Image.open(input + inputfile)
    except BaseException:
        print(input + inputfile + "  error")
        os.remove(input + inputfile)
        return
    w = img.size[0]
    h = img.size[1]
    #切割成正方形
    if h > w:
        offset = int(h - w) / 2
        outcolor = img.crop((0,offset,w,offset+w))
    else:
        offset = int(w - h) / 2
        outcolor = img.crop((offset,0,offset+h,h))
    # 图片缩小
    outcolor = outcolor.resize((128, 128), Image.ANTIALIAS).convert('RGB')
    outcolor.save(outputcolor + inputfile, 'JPEG')
    # 图片黑白
    outbw = outcolor.convert('L')
    outbw.save(outputbw + inputfile, 'JPEG')


def start_run(start):
    for i in tqdm.trange(start, start + 100, desc=str(start) + 'Task', ncols=100):
        inputfile = str(i).zfill(6) + ".jpg"
        # print(inputfile)
        if os.path.exists(input + inputfile):
            c2bw(inputfile)


if __name__ == '__main__':
    # start_run(1)
    testFL = []
    pool_size = multiprocessing.cpu_count()  # 进程数量
    for i in range(1, 50000, 100):
        testFL.append(i)
    pool = multiprocessing.Pool(processes=pool_size)
    for i in testFL:
        pool.apply_async(start_run, args=(i,))
    pool.close()
    pool.join()

    print("====================代码执行完毕==========================")
