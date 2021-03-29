import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_model
from tensorflow.keras.optimizers import Adam
from data_preprocess import data_generator_wrapper


def main():
    annotation_train_path = 'annotation.txt'
    log_dir = 'logs/000/'
    num_classes = 14

    max_bbox_per_scale = 100

    anchors_stride_base = np.array([[[ 18, 148], [ 40,  92], [ 52,  36]],
                                    [[ 80, 122], [116,  66], [120,  50]],
                                    [[134,  30], [158,  96], [184, 150]]])

    anchors_stride_base = anchors_stride_base.astype(np.float32)
    anchors_stride_base[0] /= 8
    anchors_stride_base[1] /= 16
    anchors_stride_base[2] /= 32

    input_shape = (512, 512)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5', monitor='loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

    with open(annotation_train_path) as f:
        lines_train = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines_train)
    np.random.seed(None)
    num_train = len(lines_train)

    model, model_body = create_model(input_shape, anchors_stride_base, num_classes, load_pretrained=False, freeze_body=2, weights_path=None)

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-5), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Unfreeze all of the layers.')

    batch_size = 32
    print('Train on {} samples, with batch size {}.'.format(num_train, batch_size))
    model.fit(data_generator_wrapper(lines_train, batch_size, anchors_stride_base, num_classes, max_bbox_per_scale, 'train'),
                steps_per_epoch=max(1, num_train//batch_size),
                epochs=20,
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])


if __name__ == '__main__':
    main()
