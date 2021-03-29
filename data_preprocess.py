import numpy as np
import albumentations as al
import cv2
from box_processing import preprocess_true_boxes

def parse_annotation (annotation, train_input_size, annotation_type):
    line = annotation.split()
    image_path = line[0]
    image = np.array(cv2.imread(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    exist_boxes = True

    augmentation = al.Compose([
        # al.VerticalFlip(p=0.5),
        # al.HorizontalFlip(p=0.5),
        # al.RandomRotate90(p=0.5),
        al.OneOf([al.GaussNoise(0.002, p=0.5),
                  al.IAAAffine(p=0.5), ], p=0.2),
        al.OneOf([al.Blur(blur_limit=(3, 10), p=0.4),
                  al.MedianBlur(blur_limit=3, p=0.3),
                  al.MotionBlur(p=0.3)], p=0.3),
        al.OneOf([al.RandomBrightness(p=0.3),
                  al.RandomContrast(p=0.4),
                  al.RandomGamma(p=0.3)], p=0.5),
        al.Cutout(num_holes=20, max_h_size=20, max_w_size=20, p=0.5),
    ])

    bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    if annotation_type == 'train':
        # image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
        # image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
        # image, bboxes = random_translate(np.copy(image), np.copy(bboxes))

        augm = augmentation(image=image)
        image = augm['image'].astype(np.float32) / 255.

    # image, bboxes = image_preprocess(np.copy(image), [train_input_size, train_input_size], np.copy(bboxes))
    return image, bboxes, exist_boxes


def data_generator(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
    n = len(annotation_lines)
    i = 0

    strides = np.array([8, 16, 32])

    while True:
        train_input_size = 512
        train_output_sizes = train_input_size // strides

        batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3))

        batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0], 3, 5 + num_classes))
        batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1], 3, 5 + num_classes))
        batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2], 3, 5 + num_classes))

        batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))

        for num in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)

            image, bboxes, exist_boxes = parse_annotation(annotation_lines[i], train_input_size, annotation_type)
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors)

            batch_image[num, :, :, :] = image
            if exist_boxes:
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
            i = (i + 1) % n
        yield [batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type)




