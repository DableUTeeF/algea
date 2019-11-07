import os
from xml.etree import cElementTree as ET
from PIL import Image
from yolo.utils import create_csv_training_instances


if __name__ == '__main__':
    train_path = '/home/palm/PycharmProjects/algea/dataset/train_annotations'
    test_path = '/home/palm/PycharmProjects/algea/dataset/test_annotations'
    classes_path = '/home/palm/PycharmProjects/algea/dataset/classes'
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(train_path, test_path, classes_path, False)
    for instance in train_ints:
        imname = os.path.split(instance['filename'])[1]
        impath = instance['filename']
        try:
            image = Image.open(impath)
        except FileNotFoundError:
            continue
        width, height = image.size
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = imname
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)

        objs = instance['object']
        for obj_ in objs:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = 'obj'

            x1 = obj_['xmin']
            x2 = obj_['xmax']
            y1 = obj_['ymin']
            y2 = obj_['ymax']

            bndbx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbx, 'xmin').text = str(x1)
            ET.SubElement(bndbx, 'xmax').text = str(x2)
            ET.SubElement(bndbx, 'ymin').text = str(y1)
            ET.SubElement(bndbx, 'ymax').text = str(y2)
        tree = ET.ElementTree(root)
        tree.write(os.path.join('/media/palm/data/MicroAlgae/16_8_62/voc_anns/train', imname[:-4]+'.xml'))

    for instance in valid_ints:
        imname = os.path.split(instance['filename'])[1]
        impath = instance['filename']
        try:
            image = Image.open(impath)
        except FileNotFoundError:
            continue
        width, height = image.size
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = imname
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)

        objs = instance['object']
        for obj_ in objs:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = 'obj'

            x1 = obj_['xmin']
            x2 = obj_['xmax']
            y1 = obj_['ymin']
            y2 = obj_['ymax']

            bndbx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbx, 'xmin').text = str(x1)
            ET.SubElement(bndbx, 'xmax').text = str(x2)
            ET.SubElement(bndbx, 'ymin').text = str(y1)
            ET.SubElement(bndbx, 'ymax').text = str(y2)
        tree = ET.ElementTree(root)
        tree.write(os.path.join('/media/palm/data/MicroAlgae/16_8_62/voc_anns/test/', imname[:-4]+'.xml'))
