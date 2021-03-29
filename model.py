from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from yolo_body import yolo4_body
from loss import yolo_loss


def create_model(input_shape, anchors_stride_base, num_classes, load_pretrained=True, freeze_body=2, weights_path='./yolov4.h5'):
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors_stride_base)

    max_bbox_per_scale = 150
    iou_loss_thresh = 0.5

    model_body = yolo4_body(image_input, num_anchors, num_classes)
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors*3, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            num = (250, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    y_true = [Input(name='input_2', shape=(None, None, 3, (num_classes + 5))),
              Input(name='input_3', shape=(None, None, 3, (num_classes + 5))),
              Input(name='input_4', shape=(None, None, 3, (num_classes + 5))),
              Input(name='input_5', shape=(max_bbox_per_scale, 4)),
              Input(name='input_6', shape=(max_bbox_per_scale, 4)),
              Input(name='input_7', shape=(max_bbox_per_scale, 4))]

    loss_list = Lambda(yolo_loss, name='yolo_loss',
                           arguments={'num_classes': num_classes, 'iou_loss_thresh': iou_loss_thresh,
                                      'anchors': anchors_stride_base})([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], loss_list)
    model.summary()
    return model, model_body

