"""
Program Name: main_SeepDetection.py
Author: Yutong Zhang
Date: 2024-06-12
Description:
1)  This program is written for the CGG Oil Seep Detection Exercise.
2)  This program is based on a popular third-party library for image segmentation. The GitHub link is as follows:
    https://github.com/qubvel/segmentation_models.pytorch.git
3)  Clarification about my contributions:
    a)  Went through the entire repository and got the baseline working (some contents are outdated, so I needed to
        modify them to ensure everything works).
    b)  Modified it to work for the seep detection exercise. The main changes are in "SeepDetectionDataset",
        "format_SeepDetectionDataset" and "SeepDetectionModel".
    c)  Added some code and functions to facilitate further coding or investigations. For example:
        1.  Summarized adjustable parameters.
        2.  Enhanced the switch between binary and multiclass segmentation using a global setting and pre-written
            essential codes for each.
        3.  Added mask value checks before training.
        4.  Implemented last model saving.
    d)  Rewrote and added annotations to enhance clarity and ease of understanding.
    e)  Experimented with different tasks and architectures. Below are some comparisons and analyses:
        (Results sorted from better to worse overall performance)
        Binary segmentation (seg_mode="binary"):
            Training and test results:
                UnetPlusPlus    +   epoch=20:   loss=0.171  valid_per_image_iou=0.383   valid_dataset_iou=0.455
                                                            train_per_image_iou=0.655   train_dataset_iou=0.758
                Unet            +   epoch=20:   loss=0.221  valid_per_image_iou=0.397   valid_dataset_iou=0.448
                                                            train_per_image_iou=0.569   train_dataset_iou=0.696
                PAN             +   epoch=20:   loss=0.258  valid_per_image_iou=0.339   valid_dataset_iou=0.437
                                                            train_per_image_iou=0.457   train_dataset_iou=0.612
                UnetPlusPlus    +   epoch=8:    loss=0.600  valid_per_image_iou=0.344   valid_dataset_iou=0.408
                                                            train_per_image_iou=0.379   train_dataset_iou=0.457
                FPN             +   epoch=8:    loss=0.358  valid_per_image_iou=0.246   valid_dataset_iou=0.315
                                                            train_per_image_iou=0.336   train_dataset_iou=0.468
            Analysis:
                1.  UnetPlusPlus is superior to Unet: the additional skip connections and nested structures provide
                    benefits.
                2.  UnetPlusPlus outperforms PAN and FPN: for binary segmentation, given that seeps typically exhibit
                    distinct image features (e.g., darker continuous areas), capturing detailed information appears
                    more efficient than focusing on multiscale features.
                3.  Upon reviewing predicted mask plots, UnetPlusPlus proves to be a good architecture for binary
                    segmentation in seep detection. Performance could further improve with additional data,
                    incorporation of better pre-trained models, and fine-tuning of parameters.
        Multiclass segmentation (seg_mode="multiclass"):
            Training and test results:
                UnetPlusPlus    +   epoch=20:   loss=0.512  valid_per_image_iou=0.979   valid_dataset_iou=0.979
                                                            train_per_image_iou=0.978   train_dataset_iou=0.977
                PAN             +   epoch=20:   loss=0.412  valid_per_image_iou=0.968   valid_dataset_iou=0.966
                                                            train_per_image_iou=0.976   train_dataset_iou=0.975
                MAnet           +   epoch=20:   loss=0.592  valid_per_image_iou=0.969   valid_dataset_iou=0.968
                                                            train_per_image_iou=0.971   train_dataset_iou=0.970
                FPN             +   epoch=8:    loss=0.603  valid_per_image_iou=0.956   valid_dataset_iou=0.954
                                                            train_per_image_iou=0.960   train_dataset_iou=0.959
                Unet            +   epoch=8:    loss=0.695  valid_per_image_iou=0.825   valid_dataset_iou=0.786
                                                            train_per_image_iou=0.841   train_dataset_iou=0.814
                DeepLabV3Plus   +   epoch=20:   loss=0.599  valid_per_image_iou=0.818   valid_dataset_iou=0.754
                                                            train_per_image_iou=0.817   train_dataset_iou=0.760
                DeepLabV3       +   epoch=20:   loss=0.532  valid_per_image_iou=0.829   valid_dataset_iou=0.759
                                                            train_per_image_iou=0.728   train_dataset_iou=0.640
                Unet            +   epoch=20:   loss=0.598  valid_per_image_iou=0.753   valid_dataset_iou=0.670
                                                            train_per_image_iou=0.644   train_dataset_iou=0.543
                DeepLabV3       +   epoch=8:    loss=0.669  valid_per_image_iou=0.623   valid_dataset_iou=0.535
                                                            train_per_image_iou=0.654   train_dataset_iou=0.586
                UnetPlusPlus    +   epoch=8:    loss=0.684  valid_per_image_iou=0.588   valid_dataset_iou=0.491
                                                            train_per_image_iou=0.593   train_dataset_iou=0.511
            Analysis:
                1.  It is easier to classify a pixel as 0 (non-seep) in multiclass segmentation, which inflates the IoU
                    scores compared to binary segmentation, since IoUs in multiclass include intersections with non-seep
                    areas. These areas can be easily excluded for evaluation, but the value of this depends on whether
                    the focus is on correctly classifying seep areas or equally significant classification of non-seep
                    areas.
                2.  UnetPlusPlus, PAN, MAnet, and FPN exhibit similar performances in multiclass segmentation. This
                    suggests that it might be both important to focus on fine-grained details and features at different
                    scales, since the seep class is not something that could be detected straightforwardly like the seep
                    or non-seep classification. However, the dataset's small size (only 790 images in total, averaging
                    about 100 images per class across 8 classes) may limit the ability to differentiate architecture
                    features, potentially reflecting dataset limitations rather than architectural differences.
                3.  DeepLabV3Plus outperforms DeepLabV3: the additional decoder structure is beneficial.
                4.  UnetPlusPlus epoch = 20 better than Unet epoch = 8 or 20, but UnetPlusPlus epoch = 8 worse: the
                    additional skip connections and nested structures are beneficial for performance, but they also
                    require more training.
                5.  Amazing to see Unet epoch = 8 better than Unet epoch = 20. This variability might stem from the
                    inherent randomness in learning processes, and more trials would be helpful for further analysis.
                6.  As mentioned before, increasing the dataset size, integrating better pre-trained models, and
                    fine-tuning parameters further are likely to enhance performance
4)  It is recommended to send ".txt" files to your email, so please contact me if you want a packaged ZIP version or
    have any other questions: katsumi.zyt@gmail.com
5)  Steps to run this program using the aforementioned third-party library:
    a)  Clone the GitHub repository.
    b)  Place this program in the root path of the repository.
    c)  Place the training images in: ./dataset_SeepDetection/train_images_256
        Place the training masks in: ./dataset_SeepDetection/train_masks_256
    d)  Install the dependencies as instructed in the GitHub repository.
    e)  Adjust some parameters if needed:
        seg_mode:
            "binary": segment regions containing seeps.
            "multiclass": classify the seeps.
        images_type:
            Set to "gray" as the seep detection images are grayscale.
        images_dir, masks_dir:
            Modify paths accordingly if images and masks are stored elsewhere.
        transform:
            Apply customized transformations to images and masks.
        batch_size:
            Default setting is 16.
        architecture:
            Refer to annotations below for options and their main features.
        encoder:
            See GitHub repository for detailed information.
        learning_rate:
            Learning rate used for Adam optimizer.
        num_classes:
            Only applicable when seg_mode = "multiclass". Set to 8 for mask values ranging from 0 to 7 (seep classes).
        epochs:
            Number of training epochs.
        gpus:
            Number of GPUs to train on (int) or which GPUs to train on (list or str).
    f)  Currently, dice loss is used for training, and IoU (Intersection over Union, a common metric for image
        segmentation) is reported. The library offers additional losses and metrics, but implementing them in the
        "SeepDetectionModel" class would require specific modifications.
    g)  Proceed to run the program.
    h)  Plots will be displayed, and the model will be saved to the root path with a timestamped name.
6)  Seep detection is crucial for various applications such as exploration (e.g., assisting teams in locating oil or gas
    accumulations) and pollution control (e.g., identifying potential oil spills for early detection and response).
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
    architecture = "UnetPlusPlus"

    # choose an encoder, options see GitHub repository
    encoder = "resnet34"

    learning_rate = 0.0001  # for Adam optimizer
    num_classes = 8  # only used for multiclass, correspond to mask values 0~7
    epochs = 8
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
