import torch
from utils.models import VGGEncoder, Decoder

device = torch.device('cpu')

# Create dummy decoder and save it
decoder = Decoder().to(device)
torch.save(decoder.state_dict(), 'weights/decoder_final.pth')

print("Dummy decoder saved to weights/decoder_final.pth")