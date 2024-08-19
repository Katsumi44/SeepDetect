"""
Program Name: main_SeepDetection.py
Author: Yutong Zhang
Date: 2024-06-11
Description:
1)  This program is written for the CGG Oil Seep Detection Exercise.
2)  This program is based on a popular third-party library for image segmentation. The GitHub link is as below:
    https://github.com/qubvel/segmentation_models.pytorch.git
3)  Steps to run this program under the above-mentioned third-party library: (it is recommended to send .txt files to your email, so please contact me if you want a packaged ZIP version)
    a)  Clone the GitHub repository.
    b)  Put this program under the root path of the repository.
    c)  Put the training images under: ./dataset_SeepDetection/train_images_256.
        Put the training masks under: ./dataset_SeepDetection/train_masks_256.
    d)  Install the dependencies following instructions on the GitHub repository.
    e)  Adjust some parameters if needed:
        seg_mode:
            "binary": segment regions that contain seeps
            "multiclass": classify the seeps
        images_type:
            set to "gray" since the seep detection images are gray
        images_dir, masks_dir:
            change the paths correspondingly if putting the images and masks somewhere else
        transform:
            any customized transformations for the images and masks
        batch_size
        architecture:
            options and their corresponding main features are listed in the below annotations
        encoder:
            see the GitHub repository for more information
        learning_rate
        num_classes:
            only used when seg_mode = "multiclass", set to 8 since we have 0~7 mask values (seep classes)
        epochs
        gpus
    f)  Currently, dice loss is used for training, and IoU (Intersection over Union, a common metric used for image segmentation) is reported. The library also provides other losses and metrics, but further detailed modifications for the SeepDetectionModel class would be needed to realize each of them.
    g)  Run the program.
4)  Clarification about my contributions:
    a)  Went through the whole repository and got the baseline working (some contents are out-dated, so need to modify them to get everything work).
    b)  Modified to make it work for the seep detection exercise. Main changes are in:
        SeepDetectionDataset
        format_SeepDetectionDataset
        SeepDetectionModel
    c)  Added some codes and functions to make further codings or investigations easier. For example:
        Summarized the adjustable parameters.
        Made the switching between binary and multiclass segmentation more fluent by using a global setting and pre-writing the respective essential codes for each.
        Added mask value check before training.
        Added last model saving.
    d)  Rewrote and added some annotations to make everything clearer and easier to understand.
    e)  Did some experiments with different settings (was not able to run for large epoch numbers or try every possible setting due to computation and time limits)
        seg_mode="binary" + architecture="FNP" + epochs=8
        seg_mode="multiclass" + architecture="FNP" + epochs=8
"""

import glob
import os
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split

import segmentation_models_pytorch as smp

seg_mode = "binary"  # options: "binary", "multiclass"
images_type = "gray"  # options: "gray", "rgb"
images_dir = "./dataset_SeepDetection/train_images_256"
masks_dir = "./dataset_SeepDetection/train_masks_256"


class SeepDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # corresponding images and masks have the same file names
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.masks[index])

        image = np.array(Image.open(image_path))  # 16 bit
        image = image.astype(np.float32)  # ensure compatibility (uint16 is not supported for later computation)

        mask = np.array(Image.open(mask_path))  # 8 bit
        mask = mask.astype(np.float32)
        mask = self._preprocess_mask(mask)

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        if seg_mode == "binary":
            mask[mask == 0.0] = 0.0
            mask[mask > 0.0] = 1.0
        return mask


class format_SeepDetectionDataset(SeepDetectionDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))

        # format conversion for later computations: HWC -> CHW (H: Height, W: Width, C: Channel)

        # print(f"image shape (original) = {sample['image'].shape}")
        if images_type == "gray":
            sample["image"] = np.expand_dims(image, 0)  # add C = 1
        elif images_type == "rgb":
            sample["image"] = np.moveaxis(image, -1, 0)
        # print(f"image shape (after CHW conversion) = {sample['image'].shape}")

        # print(f"mask shape (original) = {sample['mask'].shape}")
        sample["mask"] = np.expand_dims(mask, 0)  # add C = 1
        # print(f"mask shape (after CHW conversion) = {sample['mask'].shape}")

        return sample


def get_unique_values(file_path):
    img = Image.open(file_path)
    img_array = np.array(img)
    unique_values = np.unique(img_array)
    return unique_values


def check_masks(directory):
    tif_files = glob.glob(os.path.join(directory, "*.tif"))  # get all .tif files in the directory
    temp_min = float('inf')
    temp_max = float('-inf')
    for file_path in tif_files:
        unique_values = get_unique_values(file_path)
        if temp_max < max(unique_values):
            temp_max = max(unique_values)
        if temp_min > min(unique_values):
            temp_min = min(unique_values)
        # print(f"Unique values in {os.path.basename(file_path)}: {unique_values}")
    print(f"Max unique value in {os.path.basename(directory)}: {temp_max}")
    print(f"Min unique value in {os.path.basename(directory)}: {temp_min}")


if __name__ == '__main__':

    # ****************************** Parameters ******************************

    # define transforms for images and masks
    transform = None

    batch_size = 16

    # choose an architecture, options:
    # "Unet": symmetric U-shaped architecture with encoder-decoder paths and skip connections
    # "UnetPlusPlus": U-Net with nested and dense skip connections -> better capturing fine-grained details
    # "MAnet": integrate attention mechanisms to focus on important regions of the image -> better for complex scenes
    # "Linknet": lightweight -> better for real-time applications
    # "FPN": multi-scale feature pyramid -> better detecting objects at different scales
    # "PSPNet": pyramid pooling module to capture global context information
    # "DeepLabV3": atrous convolution and spatial pyramid pooling to capture multiscale context information
    # "DeepLabV3Plus": encoder-decoder structure to refine the segmentation results, especially along object boundaries
    # "PAN": pyramid pooling + attention mechanisms -> focus on relevant features at multiple scales
    architecture = "Unet"

    # choose an encoder, options see GitHub repository
    encoder = "resnet34"

    learning_rate = 0.0001  # for Adam optimizer
    num_classes = 8  # only used for multiclass, correspond to mask values 0~7
    epochs = 20
    gpus = 1
    log_every_n_steps = 5  # total step number related to both training dataset size and batch size

    # ****************************** Dataset Preparation ******************************

    # check and print the mask values
    check_masks(masks_dir)

    dataset = format_SeepDetectionDataset(images_dir, masks_dir, transform=transform)

    # split the dataset into training and validation
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    # since no separate test dataset is provided, use the validation dataset as the test dataset
    test_size = valid_size
    test_dataset = valid_dataset
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    # plot some samples
    for i in range(5):
        sample = dataset[i]
        plt.figure()
        plt.subplot(1, 2, 1)
        if images_type == "gray":
            plt.imshow(sample["image"].squeeze(), cmap='gray')  # remove the C = 1 dimension
        elif images_type == "rgb":
            plt.imshow(sample["image"].transpose(1, 2, 0))  # convert back from CHW to HWC
        plt.subplot(1, 2, 2)
        plt.imshow(sample["mask"].squeeze(), cmap='gray')  # remove the C = 1 dimension
        plt.show()

    input("Press Enter to start training!")

    # ****************************** Training ******************************

    class SeepDetectionModel(pl.LightningModule):
        def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
            super().__init__()
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )

            # preprocessing parameters for images
            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

            # for image segmentation dice loss could be the best first choice
            if seg_mode == "binary":
                self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
            elif seg_mode == "multiclass":
                self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        def forward(self, image):
            image = (image - self.mean) / self.std  # normalization
            mask = self.model(image)
            return mask

        def shared_step(self, batch, stage):
            image = batch["image"]
            mask = batch["mask"]

            # check image shape: [batch_size, num_channels, height, width]
            # (if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width])
            assert image.ndim == 4

            # check that image dimensions are divisible by 32, reasons:
            # encoder and decoder connected by "skip connections" and usually encoder have 5 stages of down-sampling by
            # factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have the following shapes of
            # features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80; and we will get an error trying
            # to concat these features
            h, w = image.shape[2:]
            assert h % 32 == 0 and w % 32 == 0

            # check mask shape: should be [batch_size, num_classes, height, width]
            # (for binary segmentation num_classes = 1)
            assert mask.ndim == 4

            # check mask values
            if seg_mode == "binary":
                assert mask.max() <= 1.0 and mask.min() >= 0
            elif seg_mode == "multiclass":
                assert mask.max() <= 7.0 and mask.min() >= 0

            logits_mask = self.forward(image)  # shape: [batch_size, num_classes, 256, 256]
            # print(f"\nlogits_mask shape = {logits_mask.shape}")
            # print(f"mask shape = {mask.shape}")

            loss = self.loss_fn(logits_mask, mask)

            if seg_mode == "binary":
                # probabilities for predicting 0 or 1
                prob_mask = logits_mask.sigmoid()  # shape: [batch_size, 1, 256, 256]
                # print(f"prob_mask shape (binary) = {prob_mask.shape}")

                # predict 1 if the probability is larger than predicting 0
                pred_mask = (prob_mask > 0.5).float()  # shape: [batch_size, 1, 256, 256]
                # print(f"pred_mask shape (binary) = {pred_mask.shape}")

            elif seg_mode == "multiclass":
                # probabilities for predicting every class
                prob_mask = logits_mask.softmax(dim=1)  # shape: [batch_size, num_classes, 256, 256]
                # print(f"prob_mask shape (multiclass) = {prob_mask.shape}")

                # set the one with the highest probability as the prediction
                pred_mask = torch.argmax(prob_mask, dim=1, keepdim=True)  # shape: [batch_size, num_classes, 256, 256]
                # print(f"pred_mask shape (multiclass) = {pred_mask.shape}")

            # compute IoU metric by two ways: dataset-wise, image-wise
            # (but for now we just compute true positive, false positive, false negative and true negative "pixels" for
            # each image and class; these values will be aggregated in the end of an epoch)
            if seg_mode == "binary":
                tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(),
                                                       mode="binary")
            elif seg_mode == "multiclass":
                tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(),
                                                       mode="multiclass", num_classes=num_classes)

            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        def shared_epoch_end(self, outputs, stage):
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            # per image IoU means that we first calculate IoU score for each image
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset
            # with "empty" images (images without target class) a large gap could be observed.
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            metrics = {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
            }

            self.log_dict(metrics, prog_bar=True)

        def training_step(self, batch, batch_idx):
            return self.shared_step(batch, "train")

        def training_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_step(batch, "valid")

        def validation_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "valid")

        def test_step(self, batch, batch_idx):
            return self.shared_step(batch, "test")

        def test_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "test")

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=learning_rate)


    if seg_mode == "binary":
        model = SeepDetectionModel(arch=architecture, encoder_name=encoder, in_channels=3, out_classes=1)
    elif seg_mode == "multiclass":
        model = SeepDetectionModel(arch=architecture, encoder_name=encoder, in_channels=3, out_classes=num_classes)

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        log_every_n_steps=log_every_n_steps
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # save the last model
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name = f"model_{current_time}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"\nLast model saved to: {model_name}\n")

    # ****************************** Test and Visualization of Results ******************************

    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)

    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])

    if seg_mode == "binary":
        pr_masks = logits.sigmoid()
    elif seg_mode == "multiclass":
        pr_masks = logits.softmax(dim=1)
        pr_masks = torch.argmax(pr_masks, dim=1, keepdim=True)

    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure()

        plt.subplot(1, 3, 1)
        if images_type == "gray":
            plt.imshow(image.numpy().squeeze(), cmap='gray')
        elif images_type == "rgb":
            plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze(), cmap='gray')
        plt.title("Prediction")
        plt.axis("off")

        plt.show()
