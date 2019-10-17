import cv2
import csv
from retinanet.preprocessing.csv_generator import _read_annotations, _open_for_csv, _read_classes
from utils import add_bbox

if __name__ == '__main__':

    with _open_for_csv('/home/palm/PycharmProjects/algea/dataset/classes') as file:
        classes = _read_classes(csv.reader(file, delimiter=','))
    with _open_for_csv('/home/palm/PycharmProjects/algea/dataset/train_annotations') as file:
        train_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    with _open_for_csv('/home/palm/PycharmProjects/algea/dataset/test_annotations') as file:
        test_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)

    for k in train_image_data:
        anns = train_image_data[k]
        image = cv2.imread(k)
        for ann in anns:
            bbox = (ann['x1'], ann['y1'], ann['x2'], ann['y2'])
            image = add_bbox(image, bbox, classes[ann['class']], ['mif', 'ov'])
        cv2.imwrite(f'/home/palm/PycharmProjects/algea/dataset/plotted/{k.split("/")[-1]}', image)
