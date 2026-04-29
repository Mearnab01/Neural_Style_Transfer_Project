import argparse
import sys
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

from utils.utils import *
from utils.models import *

def parse_arguments():
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument('--content_dir', type=str, default=r'/content/drive/MyDrive/ai_nst/big_datasets/content_dataset')
    
    p.add_argument('--style_dir',   type=str, default=r'/content/drive/MyDrive/ai_nst/big_datasets/style_dataset')
    
    p.add_argument('--vgg', type=str, default=r'/content/drive/MyDrive/ai_nst/Neural_Style_Transfer_Project/vgg_normalised.pth')
    
    p.add_argument('--experiment',    type=str, default='experiment1')
    p.add_argument('--decoder_path',  type=str, default=None)
    p.add_argument('--optimizer_path',type=str, default=None)
    
    # Image
    p.add_argument('--final_size',   type=int,   default=256)
    p.add_argument('--content_size', type=int,   default=512)
    p.add_argument('--style_size',   type=int,   default=512)
    p.add_argument('--crop',         action='store_true', default=True)

    # Training
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--epochs',          type=int,   default=2)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--lr_decay',        type=float, default=5e-5)
    p.add_argument('--content_weight',  type=float, default=1.0)
    p.add_argument('--style_weight',    type=float, default=5.0)
    p.add_argument('--log_interval',    type=int,   default=1)
    p.add_argument('--save_interval',   type=int,   default=2)
    p.add_argument('--resume',          action='store_true', default=False)
    
    return p.parse_args()
    
    
def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments for reproducibility
    with open(save_dir / 'args.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
            
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    styled_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageDataset(args.content_dir, content_transform)
    style_dataset = ImageDataset(args.style_dir, styled_transform)
    
    if device.type == 'cpu':
        print("Using CPU for training.")
        
        subset_size = 5
        content_dataset = Subset(content_dataset, list(range(subset_size)))
        style_dataset = Subset(style_dataset, list(range(subset_size)))

    else:
        print("Using GPU for training.")

    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    exp_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=1 - args.lr_decay
    )
    lamda_schedular = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: 1.0/ (1.0 - epoch * args.lr_decay)
    )
    
    if args.resume:
        if args.decoder_path and args.optimizer_path:
            decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
            optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
            print(f"Resumed from {args.decoder_path} and {args.optimizer_path}")
        else:
            print("Resume flag is set but decoder or optimizer path is missing. Starting fresh.")
    
    print("Starting training...")
    mse_loss = nn.MSELoss()
    
    encoder.eval()
    
    running_loss = None
    running_closs = None
    running_sloss = None
    
    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_loader, style_loader),
                            total=min(len(content_loader), len(style_loader)))

        running_loss = 0
        running_closs = 0
        running_sloss = 0
        
        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            # 🔹 Extract features
            content_features = encoder(content_batch)
            style_features = encoder(style_batch)

            # 🔹 Apply AdaIN (align content to style)
            target_features = adaptive_instance_normalization(
                content_features[-1], 
                style_features[-1]
            )

            # 🔹 Generate stylized image
            generated_images = decoder(target_features)

            # 🔹 Extract features from generated image
            generated_features = encoder(generated_images)

            # 🔹 Content Loss (structure preservation)
            content_loss = mse_loss(
                generated_features[-1], 
                target_features
            ) * args.content_weight

            # 🔹 Style Loss (texture + statistics)
            style_loss = 0
            for gen_feat, style_feat in zip(generated_features, style_features):
                gen_mean, gen_std = calc_mean_std(gen_feat)
                style_mean, style_std = calc_mean_std(style_feat)

                style_loss += (
                    mse_loss(gen_mean, style_mean) +
                    mse_loss(gen_std, style_std)
                )

            style_loss *= args.style_weight

            # 🔹 Total Loss
            total_loss = content_loss + style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            progress_bar.set_description(f'Loss:{total_loss.item():4f}, Content Loss: {content_loss.item():4f}, Style Loss: {style_loss.item():4f}')
            
            running_loss += total_loss.item()
            running_closs += content_loss.item()
            running_sloss += style_loss.item()
            
        lamda_schedular.step()
        
        running_loss /= len(content_loader)
        running_closs /= len(content_loader)
        running_sloss /= len(content_loader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, generated_images], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)

    
    

if __name__ == "__main__":
    main()