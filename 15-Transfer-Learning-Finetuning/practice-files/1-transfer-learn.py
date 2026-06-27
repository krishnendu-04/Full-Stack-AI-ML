# TENSORFLOW
print("--------------------TENSORFLOW--------------------")
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

# Load a pre-trained ResNet50 model
model = ResNet50(weights="imagenet")

# Display the models architecture
model.summary()

# Access specific layers in tensorflow
for i,layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

# Freezing all layers except the top layer
for layer in model.layers[:-10]:
    layer.trainable = False
    




# PYTORCH
print("--------------------PYTORCH--------------------")
import torch
import torchvision.models as models

# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Print model architecture
print(model)

# Freeze the model parameter
for param in model.parameters():
    param.requires_grad = False
    
# Modify the final layer for a new task
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features,10)
print("Modified Model: \n",model)

for name,param in model.named_parameters():
    print(f"Layer: {name}, Required Grad: {param.requires_grad}")
    
    
# To unfreeze specific layers
for name,param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True
        