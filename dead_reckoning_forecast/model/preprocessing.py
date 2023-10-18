import torchvision.transforms as transforms

def create_transformer(img_size, mean_std=None, aug=False):
    augmentations = []
    if aug:
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
        ]
    normalizations = []
    if mean_std is not None:
        normalizations = [transforms.Normalize(*mean_std)]
    return transforms.Compose([
        transforms.Resize(img_size),
        *augmentations,
        transforms.ToTensor(),
        *normalizations
    ])
