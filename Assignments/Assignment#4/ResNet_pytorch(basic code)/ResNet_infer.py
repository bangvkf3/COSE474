import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch_basic.ResNet_model import ResNet32_model, Config


if __name__ == "__main__":
    cfg = Config()

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # GPU 사용시
    if torch.cuda.is_available():
        torch.cuda.device(0)

    # 모델 생성
    model = ResNet32_model()


    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()

    save_path = "./saved_model/setting_1/epoch_99.pth"
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transforms_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    imgs = ImageFolder('./example', transform=transforms_test)
    print("imgs:", imgs)
    test_loader = DataLoader(imgs, batch_size=1)

    # pp.imshow(inputimg.permute([2, 1, 0]))
    # pp.show()
    print("test_loader:", test_loader)
    print(test_loader.dataset)

    for thisimg, label in test_loader:
        pred = model.forward(thisimg.to(device))
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)
        print("--------------------------------------")
        print("truth:", classes[label])
        print("model prediction:", classes[top_pred])
