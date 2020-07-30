import cv2


def draw_bboxes(image, bboxes, color, box_format='coco', yxyx=False):
    if yxyx:
        bboxes = bboxes[:, [1, 0, 3, 2]]

    for box in bboxes:
        pt1 = tuple(box[:2])
        if box_format == 'coco':
            pt2 = tuple(box[:2] + box[2:])
        elif box_format == 'pascal_voc':
            pt2 = tuple(box[2:])
        else:
            raise AttributeError("Not supported: {}".format(box_format))
        cv2.rectangle(image, pt1, pt2, color, 1)


def save_image(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
