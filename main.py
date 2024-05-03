import torch
import torch.optim as optim
from model3 import Net
from torchvision import transforms
from utils import Trainer, build_mnist, evaluate_model, plot_history, plot_sampledata, modelsummary
import time

# Train data transformations
train_transforms = transforms.Compose(
    [
           
        #transforms.Resize((28, 28)),
                       transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9, 1.1)),
                        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#                        transforms.RandomCrop(28 , padding = 1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

# Test data transformations
test_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

batch_size = 256

kwargs = {
    "batch_size": batch_size,
    "shuffle": True,
    "num_workers": 2,
    "pin_memory": True,
}

train_data, train_loader = build_mnist(set="train", transforms=train_transforms, **kwargs)
test_data, test_loader = build_mnist(set="test", transforms=test_transforms, **kwargs)




Net()(torch.rand(1, 1, 28, 28))
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
modelsummary(Net(), device)

num_epochs = 15
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1, verbose=True)
# scheduler = None

trainer = Trainer(model, device, optimizer)
t0 = time.time()
for epoch in range(1, num_epochs + 1):
    
    print(f"Epoch {epoch}")
    trainer.train(train_loader)
    trainer.test(test_loader)
    if scheduler:
        scheduler.step()
print("Training time:", time.time()-t0)