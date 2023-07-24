from utils import *

class BreastScanDataset(Dataset):
    def __init__(self, data, root_dir, transform=None, augmentations=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.augmentations = augmentations
        self.label_map = {i: label for i, label in enumerate(data.columns[3:])}  # Create a mapping from class index to label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.root_dir+self.data.iloc[idx, 2][1:]
        
        # Open the image file
        image = Image.open(img_path)

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Scale the pixel values to the range 0-1
        image_array = image_array / np.max(image_array)

        # Convert to 'RGB' by duplicating the scaled array across three channels
        rgb_image_array = np.stack([image_array]*3, axis=-1)

        # Scale the pixel values to the range 0-255 and convert to integers
        rgb_image_array = (rgb_image_array * 255).astype(np.uint8)

        # Convert the numpy array back to a PIL Image
        image = Image.fromarray(rgb_image_array)

        label_vector = self.data.iloc[idx, 3:]
        
        label_index = torch.argmax(torch.tensor(label_vector))  # Convert one-hot vector to class index

        if self.transform:
            image = self.transform(image)

        if self.augmentations and label_index in [1, 2, 3]:  # Check if augmentations should be applied
            # Randomly select one augmentation and apply it to the image
            aug = random.choice(self.augmentations)
            image = aug(image)
            # ensure the output of the augmentation is also a tensor
            if not isinstance(image, torch.Tensor):
                tensor_transform = torchvision.transforms.ToTensor()
                image = tensor_transform(image)

        return image, label_index

    def get_label(self, index):
        return self.label_map[index]