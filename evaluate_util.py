import numpy as np
from yolo.utils import compute_ap, compute_overlap


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def all_annotation_from_instance(instance):
    all_annotation = [[], []]
    for obj in instance['object']:
        if obj['name'] == 'ov':
            all_annotation[0].append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
        else:
            all_annotation[1].append(np.array([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]))
    return np.array(all_annotation)


def evaluate(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    # all_detections = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]  # [[bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)], [bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)]]
    # all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]
    assert len(all_annotations) == len(all_detections)
    average_precisions = {}
    total_instances = []
    for label in range(num_classes):
        print()
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            print(i, end='\r')
            detections = all_detections[i][label]
            annotations = np.array(all_annotations[i][label])
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
        total_instances.append(num_annotations)

    return average_precisions, total_instances
