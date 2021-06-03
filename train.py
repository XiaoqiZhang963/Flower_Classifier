# ------------ Imports ----------------
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time


# ------------- Get inputs ---------------
parser = argparse.ArgumentParser(description='Get inputs')
# Basic usage
parser.add_argument('data_dir', action="store")
# Options
parser.add_argument('--save_dir', action="store", default='./checkpoint.pth')
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--hidden_units', action="store", type=int, default=512)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--epochs', action="store", type=int, default=3)
parser.add_argument('--gpu', action="store_true")
args = parser.parse_args()


# ------------ Load the Data ----------------
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)


# ---------------- Build and Train the Classifier -------------------
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

# Load pretrained model and freeze parameters
model = getattr(models, args.arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Define a new Classifier
feature_num = model.classifier[0].in_features
model.classifier = nn.Sequential(nn.Linear(feature_num, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(args.dropout),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train the Classifier
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 10

start = time.time()
print("Start training at ", time.strftime("%H:%M:%S", time.localtime()))

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1

        inputs,labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

time_elapsed = time.time() - start
print("Finsih training at ", time.strftime("%H:%M:%S", time.localtime()))
print("\nTime spent training: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))


# ------------ Save the Checkpoint ---------------
model.class_to_idx = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'pretrained_model': args.arch,
              'classifier': model.classifier,
              'learning_rate':args.learning_rate,
              'epochs':args.epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict':model.state_dict(),
              'optimizer': optimizer.state_dict()}

torch.save(checkpoint, args.save_dir)
