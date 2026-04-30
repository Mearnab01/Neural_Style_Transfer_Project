import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm

# Fix truncated / oversized images before anything else
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from utils.utils import get_transform, ImageDataset, adaptive_instance_normalization, calc_mean_std
from utils.models import VGGEncoder, Decoder


# ── Arguments ─────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser(description='AdaIN Style Transfer — Training')

    # Paths
    p.add_argument('--content_dir',    type=str, default='datasets/content')
    p.add_argument('--style_dir',      type=str, default='datasets/style')
    p.add_argument('--vgg',            type=str, default='vgg_normalised.pth')
    p.add_argument('--experiment',     type=str, default='experiment1')
    p.add_argument('--decoder_path',   type=str, default=None)
    p.add_argument('--optimizer_path', type=str, default=None)

    # Image
    p.add_argument('--final_size',   type=int,  default=256)
    p.add_argument('--content_size', type=int,  default=512)
    p.add_argument('--style_size',   type=int,  default=512)
    p.add_argument('--crop',         action='store_true', default=True)

    # Training
    p.add_argument('--batch_size',     type=int,   default=4)
    p.add_argument('--epochs',         type=int,   default=8)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--lr_decay',       type=float, default=5e-5)
    p.add_argument('--content_weight', type=float, default=1.0)
    p.add_argument('--style_weight',   type=float, default=5.0)
    p.add_argument('--save_interval',  type=int,   default=1)
    p.add_argument('--resume',         action='store_true', default=False)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save directory
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    # Log args for reproducibility
    with open(save_dir / 'args.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    content_tf = get_transform(args.content_size, args.crop, args.final_size)
    style_tf   = get_transform(args.style_size,   args.crop, args.final_size)

    content_dataset = ImageDataset(args.content_dir, content_tf)
    style_dataset   = ImageDataset(args.style_dir,   style_tf)

    # On CPU (local testing) use only 5 images so it runs instantly
    if device.type == 'cpu':
        print("CPU detected — using 5-image subset for testing.")
        content_dataset = Subset(content_dataset, list(range(5)))
        style_dataset   = Subset(style_dataset,   list(range(5)))
    else:
        print("GPU detected — using full dataset.")

    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        prefetch_factor=2
    )
    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,          
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        prefetch_factor=2
    )

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = VGGEncoder(args.vgg).to(device)
    encoder.eval()   # encoder is always frozen

    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    # Simple LR decay: lr = lr / (1 + lr_decay * iteration)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda iteration: 1.0 / (1.0 + args.lr_decay * iteration)
    )

    start_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        # Auto-locate checkpoint if no explicit path given
        decoder_ckpt   = args.decoder_path   or str(save_dir / 'checkpoint_decoder.pth')
        optimizer_ckpt = args.optimizer_path or str(save_dir / 'checkpoint_optimizer.pth')

        if Path(decoder_ckpt).exists():
            decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
            print(f"Loaded decoder: {decoder_ckpt}")
        else:
            print(f"Decoder checkpoint not found at {decoder_ckpt} — starting fresh.")

        if Path(optimizer_ckpt).exists():
            optimizer.load_state_dict(torch.load(optimizer_ckpt, map_location=device))
            print(f"Loaded optimizer: {optimizer_ckpt}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Starting training...")
    mse = nn.MSELoss()
    iters_per_epoch = min(len(content_loader), len(style_loader))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        decoder.train()

        total_loss_sum = 0.0
        c_loss_sum     = 0.0
        s_loss_sum     = 0.0

        pbar = tqdm(
            zip(content_loader, style_loader),
            total=iters_per_epoch,
            desc=f'Epoch {epoch + 1}',
        )

        for content_batch, style_batch in pbar:
            content_batch = content_batch.to(device, non_blocking=True)
            style_batch   = style_batch.to(device,   non_blocking=True)

            with torch.inference_mode():
                content_feats = encoder(content_batch)
                style_feats   = encoder(style_batch)

            # AdaIN: align content stats to style stats
            target_feats = adaptive_instance_normalization(
                content_feats[-1], style_feats[-1]
            )

            # Decode to image
            generated = decoder(target_feats)

            # Re-encode generated image (gradients needed here)
            generated_feats = encoder(generated)

            # Content loss — generated deep features vs AdaIN target
            c_loss = mse(generated_feats[-1], target_feats) * args.content_weight

            # Style loss — match mean & std at every VGG layer
            s_loss = sum(
                mse(*calc_mean_std(gf)) + mse(*calc_mean_std(sf))   # mean + std
                if False else
                mse(calc_mean_std(gf)[0], calc_mean_std(sf)[0]) +
                mse(calc_mean_std(gf)[1], calc_mean_std(sf)[1])
                for gf, sf in zip(generated_feats, style_feats)
            ) * args.style_weight

            loss = c_loss + s_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss_sum += loss.item()
            c_loss_sum     += c_loss.item()
            s_loss_sum     += s_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'c':    f'{c_loss.item():.4f}',
                's':    f'{s_loss.item():.4f}',
            })

        # ── End of epoch stats ────────────────────────────────────────────────
        avg_loss = total_loss_sum / iters_per_epoch
        avg_c    = c_loss_sum     / iters_per_epoch
        avg_s    = s_loss_sum     / iters_per_epoch
        tqdm.write(
            f'[Epoch {epoch + 1}]  '
            f'Loss: {avg_loss:.4f}  '
            f'Content: {avg_c:.4f}  '
            f'Style: {avg_s:.4f}'
        )

        # ── Save checkpoint ───────────────────────────────────────────────────
        if (epoch + 1) % args.save_interval == 0:
            # Rolling checkpoint — always overwrites, safe to resume from
            torch.save(decoder.state_dict(),   save_dir / 'checkpoint_decoder.pth')
            torch.save(optimizer.state_dict(), save_dir / 'checkpoint_optimizer.pth')

            # Also keep a numbered snapshot every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(decoder.state_dict(), save_dir / f'decoder_epoch{epoch + 1}.pth')

            # Save a sample grid: content | style | output
            decoder.eval()
            with torch.inference_mode():
                sample = torch.cat([content_batch, style_batch, generated], dim=0)
                save_image(sample.clamp(0, 1), save_dir / f'sample_epoch{epoch + 1}.png',
                           nrow=args.batch_size)
            decoder.train()

            print(f"Checkpoint saved at epoch {epoch + 1}")

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save(decoder.state_dict(), save_dir / 'decoder_final.pth')
    print(f"Training complete. Final model saved to {save_dir / 'decoder_final.pth'}")


if __name__ == '__main__':
    main()