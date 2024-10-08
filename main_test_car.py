import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = './data/CamVid/'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


# Helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# ****************************** Dataloader ******************************

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations."""

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
               'bicyclist', 'unlabelled']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# Visualize data
dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])
image, mask = dataset[4]
visualize(image=image, cars_mask=mask.squeeze())


# ****************************** Augmentations ******************************

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0, value=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf([albu.CLAHE(p=1), albu.RandomBrightnessContrast(p=1), albu.RandomGamma(p=1)], p=0.9),
        albu.OneOf([albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1), albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1)], p=0.9),
        albu.OneOf([albu.RandomBrightnessContrast(p=1), albu.HueSaturationValue(p=1)], p=0.9),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [albu.PadIfNeeded(384, 480)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# Visualize augmented images
augmented_dataset = Dataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation(), classes=['car'])
for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(image=image, mask=mask.squeeze(-1))

# ****************************** Create model and train ******************************

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

# Create segmentation model with pretrained encoder
model = smp.FPN(encoder_name=ENCODER, encoder_weights=None, classes=len(CLASSES), activation=ACTIVATION)
state_dict = torch.load("./pretrained/se_resnext50_32x4d-a260b3a4.pth")
model.encoder.load_state_dict(state_dict, strict=False)
model.to(DEVICE)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = Dataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
valid_dataset = Dataset(x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation(),
                        preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

loss = utils.losses.DiceLoss()
metrics = [utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

train_epoch = utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=DEVICE,
                                     verbose=True)
valid_epoch = utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)

if __name__ == '__main__':
    max_score = 0

    for i in range(0, 40):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    # ****************************** Test best saved model ******************************

    best_model = torch.load('./best_model.pth')

    test_dataset = Dataset(x_test_dir, y_test_dir, augmentation=get_validation_augmentation(),
                           preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
    test_dataloader = DataLoader(test_dataset)

    test_epoch = utils.train.ValidEpoch(model=best_model, loss=loss, metrics=metrics, device=DEVICE)
    logs = test_epoch.run(test_dataloader)

    # ****************************** Visualize predictions ******************************

    test_dataset_vis = Dataset(x_test_dir, y_test_dir, classes=CLASSES)
    for i in range(5):
        n = np.random.choice(len(test_dataset))
        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
        gt_mask = gt_mask.squeeze()
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(image=image_vis, ground_truth_mask=gt_mask, predicted_mask=pr_mask)
