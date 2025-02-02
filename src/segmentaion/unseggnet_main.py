from .unseggnet import Segmentation
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from ..dataset.isic import ISICDataset 
from torchvision import transforms
import numpy as np


parser = ArgumentParser()
parser.add_argument("--bs", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--dataset", type=str, default='./data')  
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument("--conv_type", type=str, default='ARMA')

args = parser.parse_args()

if __name__ == '__main__':
    seg = Segmentation(
        process=args.process, 
        batch_size=args.bs, 
        epochs=args.epochs, 
        resolution=tuple(args.resolution), 
        activation=args.activation, 
        loss_type=args.loss_type, 
        threshold=args.threshold, 
        conv_type=args.conv_type
    )



    # Initialize the dataset and DataLoader
    dataset = ISICDataset(
        root=args.dataset, 
        return_mask=True
    )
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    # Training loop
    total_iou = 0
    total_samples = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for batch in dataloader:
            images = batch['image']
            masks = batch['mask']

            try:
                # Convert images and masks to numpy arrays
                batch_iou = []
                for img, mask in zip(images, masks):
                    img_np = np.array(img)  
                    mask_np = np.array(mask) 

                    # Perform segmentation and calculate IoU
                    iou, _, _ = seg.segment(img_np, mask_np)
                    batch_iou.append(iou)

                total_iou += sum(batch_iou)
                total_samples += len(batch_iou)
                print(f"Batch IoU: {sum(batch_iou) / len(batch_iou):.2f}  mIoU so far: {(total_iou / total_samples):.2f}")

            except Exception as e:
                print(f"Error during training: {e}")
                continue

    print(f'Final mIoU: {(total_iou / total_samples):.4f}')
