import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define the path to the saved model
MODEL_PATH = 'cifar10_cnn.pth'

# Define the allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)

# Define the transform to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to match the input size of the model
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

# Define the CNN model class
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the saved model
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            # Predict the class of the uploaded image
            predicted_class = predict_image(file_path, model, transform)
            return render_template('result.html', predicted_class=predicted_class, image_path=file_path)
    return render_template('upload.html')

# Function to predict the class of an input image
def predict_image(image_path, model, transform):
    image = Image.open(image_path)  # Open the image file
    image = transform(image).unsqueeze(0)  # Apply the transformation and add a batch dimension
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image)  # Pass the image through the model
    _, predicted = torch.max(output, 1)  # Get the index of the class with the highest probability
    return predicted.item()  # Return the predicted class index

# Run the app
if __name__ == '__main__':
    app.secret_key = 'super_secret_key'  # Secret key for Flask session
    app.run(debug=True)
