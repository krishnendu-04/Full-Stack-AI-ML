# TENSORFLOW
print("\n--------------------TENSORFLOW--------------------\n")
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input , outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Data preparation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "PATH_TO_DATASET",
    target_size = (224,224),
    batch_size = 32,
    class_mode = "categorical",
    subset = "training"
)

val_data = datagen.flow_from_directory(
    "PATH_TO_DATASET",
    target_size = (224,224),
    batch_size = 32,
    class_mode = "categorical",
    subset = "validation"
)


for layer in base_model.layers[-5:]:
    layer.trainable = True
    
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data)
)    
val_loss, val_accuracy = model.evaluate(val_data)






# PYTORCH
print("\n--------------------PYTORCH--------------------\n")
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Freeze all the layers
for param in model.parameters():
    param.requires_grad = False
    
# Replace the last layer for a new task
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5),
    nn.Softmax(dim=1)
)

print(model)

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("PATH_TO_FOLDER_TRAIN", transform=transform)
val_data = datasets.ImageFolder("PATH_TO_FOLDER_VAL", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
    

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted ==labels ).sum().item()
        
        
print(f"Validation Accuracy: {100 * correct/total:.2f}")