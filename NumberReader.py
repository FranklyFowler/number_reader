#For this file to work you need to have mspaint as a default opener for jpg photos 
#If you understand how it works you may know how to change things in the code to have it trained better
#im not totally sure how to have the file found anywhere on your pc so they all have to have the same directory
#Hopefully this help but i hope you enjoy my first AI project



import os
import time
import shutil
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale

# Define the CNN model structure for image classification
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),         # First convolutional layer with 32 filters of size 3x3
            nn.ReLU(),                       # call ReLU
            nn.Conv2d(32, 64, (3,3)),        # Second convolutional layer with 64 filters
            nn.ReLU(),                       # call ReLU
            nn.Conv2d(64, 64, (3,3)),        # Third convolutional layer
            nn.ReLU(),                       # call ReLU
            nn.Flatten(),                    # Flatten the output for the linear layer
            nn.Linear(64*(28-6)*(28-6), 10)  # Output layer with 10 units for 10 classes
        )

    def forward(self, x):
        return self.model(x)


def train_model():
    # Load the MNIST dataset, apply transformations
    full_dataset = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))  # 80% training, 20% validation
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoader for both training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create the network, optimizer, and loss function
    classifier = ImageClassifier().to('cpu')
    optimizer = Adam(classifier.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')  # Move data to CPU
            optimizer.zero_grad()  # Clear gradients
            yhat = classifier(X)  # Forward pass
            loss = loss_func(yhat, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Save the trained model state
    with open('model_state.pt', 'wb') as f:
        save(classifier.state_dict(), f)


def evaluate_model():
    # Load pre-trained model
    try:
        with open('model_state.pt', 'rb') as f:
            classifier = ImageClassifier()
            classifier.load_state_dict(load(f))
            classifier.eval()

            # Load validation dataset
            val_dataset = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())
            val_loader = DataLoader(val_dataset, batch_size=32)

            # Evaluate on the validation set
            total_correct = 0
            total_samples = 0
            with torch.no_grad():  # Disable gradient calculation for inference
                for images, labels in val_loader:
                    images, labels = images.to('cpu'), labels.to('cpu')  # Move data to CPU
                    outputs = classifier(images)  # Forward pass
                    _, predicted = torch.max(outputs, 1)  # Get predicted class
                    total_samples += labels.size(0)  # Update total number of samples
                    total_correct += (predicted == labels).sum().item()  # Count correct predictions

            # Calculate accuracy
            accuracy = total_correct / total_samples
            print(f'Accuracy on validation set: {accuracy:.2%}')
    except FileNotFoundError:
        print("No saved model to evaluate.")


def main():
    while True:
        # Copy the file so it can start with a black screen
        source_file = 'C:\\Users\\lfowl824\\Desktop\\AI project\\CopyPhoto.jpg' #Put Your file adress here for the Black photo
        destination_file = 'C:\\Users\\lfowl824\\Desktop\\AI project\\YourNumber.jpg' #  It should copy to the same file location as the orginal pic
        shutil.copy(source_file, destination_file)

        # Open the file with the default application
        os.startfile(destination_file)

        # Wait for the file to be saved
        wait_for_file_save(destination_file)

        # Load pre-trained model
        try:
            with open('model_state.pt', 'rb') as f:
                classifier = ImageClassifier()
                classifier.load_state_dict(load(f))
        except FileNotFoundError:
            print("No saved model to load.")

        # Open an image, convert to grayscale, resize, convert to tensor, and add batch dimension
        img = Image.open('YourNumber.jpg')
        img_gray = Grayscale()(img)  # Convert to grayscale
        img_gray_resized = img_gray.resize((28, 28))  # Resize to 28x28
        img_tensor = ToTensor()(img_gray_resized).unsqueeze(0)  # Convert to tensor

        # Model inference
        output = classifier(img_tensor)
        predicted_class = torch.argmax(output).item()  # Get the class with the highest probability
        print("Predicted Number:", predicted_class)  # Output the predicted class

        # Remove the temporary image file
        os.remove(destination_file)
        break





def wait_for_file_save(file_path):
    initial_mtime = os.path.getmtime(file_path)
    print("Waiting for file to be saved...")
    timeout = 300  # Set a timeout limit (e.g., 300 seconds or 5 minutes)
    start_time = time.time()

    while True:
        current_mtime = os.path.getmtime(file_path)

        if current_mtime != initial_mtime:
            print("File has been saved.")
            os.system("taskkill /f /im mspaint.exe")
            os.system('cls')
            break

        time.sleep(1)  # Check every second

def question():
    while True:
        # User chooses what to do: Train, Evaluate, or Quit
        action = input("Would you like to Train (T), Evaluate (E), Run Model (R), or Quit (Q): ").upper()
        if action == "T":
            train_model()
        elif action == "E":
            evaluate_model()
        elif action == "R":
            main()  # Includes the workflow for using mspaint to edit an image and then predict
        elif action == "Q":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter T to train, E to evaluate, R to run the model, or Q to quit.")
        

if __name__ == "__main__":
    question()