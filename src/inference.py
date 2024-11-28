import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Resize, ToPILImage
from models import ResnetGenerator

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = ResnetGenerator(3, 3).to(device)
generator.load_state_dict(torch.load("/path/to/netG_A2B_epoch_200.pth"))
generator.eval()

# Image preprocessing
transform = transforms.Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# Inference function
def style_transfer(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)
    output_image = ToPILImage()((output_tensor.squeeze(0).cpu() + 1) / 2)  # Denormalize
    output_image.save(output_path)

# Example usage
style_transfer("/path/to/input_image.png", "/path/to/output_image.png")
