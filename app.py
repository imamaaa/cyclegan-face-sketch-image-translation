from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from PIL import Image
import torch
from io import BytesIO
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)

# Load CycleGAN models (update paths with actual saved model paths)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
netG_A2B = torch.load("netG_A2B_epoch_39.pth").to(device)
netG_B2A = torch.load("netG_B2A_epoch_39.pth").to(device)
netG_A2B.eval()
netG_B2A.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Convert Tensor back to Image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) * 0.5
    return transforms.ToPILImage()(tensor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files or 'conversion_type' not in request.form:
        return redirect(request.url)
    
    file = request.files['image']
    conversion_type = request.form['conversion_type']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        # Choose which generator to use based on the user's selection
        with torch.no_grad():
            if conversion_type == 'face_to_sketch':  # Convert face to sketch
                output_tensor = netG_A2B(img)
            elif conversion_type == 'sketch_to_face':  # Convert sketch to face
                img = img.convert('L')  # Ensure sketch is grayscale
                output_tensor = netG_B2A(img)

        # Convert the tensor back to image
        output_image = tensor_to_image(output_tensor)

        # Save image to BytesIO to send back to the client
        img_io = BytesIO()
        output_image.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
