import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(args):
    print('Start generate image ...')
    img_generator = ImageDataGenerator(validation_split=args.test_size)

    train_ds = img_generator.flow_from_directory(directory=args.image_dir, target_size=args.image_size,
                                                 class_mode='categorical', subset='training')

    validation_ds = img_generator.flow_from_directory(directory=args.image_dir, target_size=args.image_size,
                                                      class_mode='categorical', subset='validation')

    dict_train, dict_test = {}, {}
    for key, value in train_ds.class_indices.items():
        dict_train[key] = list(train_ds.labels).count(value)
    for key, value in validation_ds.class_indices.items():
        dict_test[key] = list(validation_ds.labels).count(value)

    print(f'train dataset: {dict_train}')
    print(f'val dataset: {dict_test}')

    num_train = len(train_ds.filepaths)
    num_val = len(validation_ds.filepaths)

    train_datasets = np.array([list(train_ds.filepaths)[:num_train], list(train_ds.labels)[:num_train]]).transpose(
        (1, 0))
    validation_datasets = np.array(
        [list(validation_ds.filepaths)[:num_val], list(validation_ds.labels)[:num_val]]).transpose((1, 0))

    list_train_datasets = []
    list_val_datasets = []
    for traindata in train_datasets:
        list_train_datasets.append([traindata[0], int(traindata[1])])
    for valdata in validation_datasets:
        list_val_datasets.append([valdata[0], int(valdata[1])])

    print(f'Train => {train_datasets.shape[0]} '
          f'Test => {validation_datasets.shape[0]}')
    print(f'Numbers of class => {args.num_class}')
    return list_train_datasets, list_val_datasets, list(train_ds.class_indices.keys())
