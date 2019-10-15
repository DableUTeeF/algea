import os
import cv2
# shape: y, x, c
# bbox: x1, y1, x2, y2

if __name__ == '__main__':
    ann_root = '/home/palm/PycharmProjects/efficient-retinanet/datastuff/algea/'
    anns = ['annotations', 'val_annotations']
    img_dir = os.path.join('images', str(len(os.listdir('images'))))
    os.mkdir(img_dir)
    os.mkdir(os.path.join(img_dir, 'ov'))
    os.mkdir(os.path.join(img_dir, 'mif'))
    for ann in anns:
        csv = open(os.path.join(ann_root, ann)).read().split('\n')
        for line in csv:
            line = line.split(',')
            x1, y1, x2, y2, cls = line[1:]
            img = cv2.imread(line[0])
            cropped = img[int(y1):int(y2), int(x1):int(x2), :]
            x = os.path.join(img_dir, cls)
            cv2.imwrite(os.path.join(img_dir, cls, str(len(os.listdir(x)))+'.jpg'), cropped)
