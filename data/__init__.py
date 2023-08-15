from utils import *
from data.dataset import BreastScanDataset

class RandomZoom:
    def __init__(self, min_scale=1.0, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        # Generate a random scale factor
        scale = random.uniform(self.min_scale, self.max_scale)

        # Get the size of the image tensor
        _, height, width = img.shape

        # Compute the new height and width
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Rescale the image
        img = nn.functional.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

        # If the image is smaller than the original, pad it to the original size
        if scale < 1.0:
            padding = (0, 0, 0, width - new_width, 0, height - new_height)
            img = F.pad(img, padding)

        # If the image is larger than the original, crop it to the original size
        elif scale > 1.0:
            start_x = random.randint(0, new_width - width)
            start_y = random.randint(0, new_height - height)
            img = img[:, start_y:start_y+height, start_x:start_x+width]

        return img

class RandomNoise:
    def __init__(self, mean=0., std=0.1):  # use a smaller std
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.  # normalize to [0, 1]
        noise = np.random.normal(self.mean, self.std, img_array.shape)  
        noisy_img_array = np.clip(img_array + noise, 0., 1.)  # make sure we're still in [0, 1]
        noisy_img_array = (noisy_img_array * 255).astype(np.uint8)  # rescale to original range

        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        noisy_img_array = np.array(img) + np.random.normal(loc=0, scale=self.std, size=np.array(img).shape)
        return Image.fromarray(noisy_img_array.astype('uint8'), img.mode)

def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path+'data_csv/train-v2_table_list_slice.csv')
    return df

def plot_augmentations(image_path, augment_list, transform, save_path):
    # Open the image file
    image = Image.open(image_path)

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

    image = transform(image)

    fig = plt.figure(figsize=(2+(2*len(augment_list)), 3), dpi=250)

    # Add the original image to the figure
    ax = fig.add_subplot(1, len(augment_list) + 1, 1)
    ax.imshow(image.permute(1, 2, 0).numpy(), cmap='gray')

    ax.set_title('Original')
    ax.axis('off')

    # Apply each transformation and add the result to the figure
    for i, transform in enumerate(augment_list, start=2):
        transformed_image = transform(image)

        # transformed_image = transformed_image.convert('I')

        ax = fig.add_subplot(1, len(augment_list) + 1, i)
        if isinstance(transformed_image, Image.Image):
            # Convert PIL Image to numpy array
            image_numpy = np.array(transformed_image)
        elif torch.is_tensor(transformed_image):
            # If it's a PyTorch tensor, it's probably in CxHxW format
            image_numpy = transformed_image.permute(1, 2, 0).numpy()

        # If the image is grayscale but still has a redundant channel dimension, remove it
        if image_numpy.shape[2] == 1:
            image_numpy = image_numpy.squeeze()

        ax.imshow(image_numpy, cmap='gray')


        ax.set_title(type(transform).__name__)
        ax.axis('off')

    # Display the figure
    plt.tight_layout()
    plt.savefig(save_path+'Transformations.png')

def preprocess_dataset(df, dataset_path, n_augment, batch_size, img_size, savefig=None):

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True), 
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.0824, 0.0824, 0.0824], std=[0.1355, 0.1355, 0.1355])  # Normalize to DBT dataset mean and std
    ])

    # Define the list of possible augmentations
    augmentations_list = [
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        RandomZoom(min_scale=1.2, max_scale=1.4),  # Random zoom
        transforms.ColorJitter(contrast=0.2),  # Random contrast
        RandomNoise(mean=0., std=0.05),  # Random noise
    ]

    if savefig is not None:
        img_path = dataset_path+df['img_path'][0]
        plot_augmentations(img_path, augmentations_list, transform, savefig)

    # Split the data into train, validation and test sets
    train_data, val_data, test_data = split_data(df)

    # Create datasets for training, validation and testing
    train_dataset, val_dataset, test_dataset = create_datasets(train_data, val_data, test_data, dataset_path, transform, augmentations_list, n_augment)

    # Create data loaders for training, validation and testing
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    # Uncomment to calculate mean and std of the dataset (takes long)
    # mean, std, min_value, max_value = calculate_statistics(train_loader)
    # print(f"Mean: {mean}, Std: {std}, Min: {min_value}, Max: {max_value}")
    
    # Output: 
    # Mean: tensor([-0.0004, -0.0004, -0.0004]), Std: tensor([1.0003, 1.0003, 1.0003]), Min: -0.6081181168556213, Max: 6.771955490112305

    # Print the class distribution 
    class_counts = print_split_distribution(train_data, 'Train', n_augment)
    _ = print_split_distribution(val_data, 'Validation', n_augment)
    _ = print_split_distribution(test_data, 'Test', n_augment)


    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts

def calculate_statistics(dataloader):
    mean = 0.
    std = 0.
    min_value = float('inf')
    max_value = float('-inf')
    nb_samples = 0.
    for data, _ in tqdm(dataloader):
        batch_samples = data.size(0)  # batch size (the last batch can have smaller size!)
        data = data.view(batch_samples, data.size(1), -1)  # flatten height and width into one dimension
        mean += data.mean(2).sum(0)  # sum over the batch dimension
        std += data.std(2).sum(0)  # sum over the batch dimension
        min_value = min(min_value, data.min())
        max_value = max(max_value, data.max())
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std, min_value, max_value

def mean_std(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

def split_data(df, test_size=0.2, val_size=0.2):
    """
    Split the data into train, validation and test sets.
    """
    # Create a new column indicating whether each study includes at least one image where 'Cancer' is 1
    df['StudyHasCancer'] = df.groupby('StudyUID')['Cancer'].transform('max')

    # Get the unique studies and their cancer status
    unique_studies = df[['StudyUID', 'StudyHasCancer']].drop_duplicates()

    # Split the studies into a temporary train set and the test set
    temp_train_studies, test_studies = train_test_split(unique_studies, test_size=test_size, stratify=unique_studies['StudyHasCancer'], random_state=5)

    # Split the temporary train set into the final train set and the validation set
    train_studies, val_studies = train_test_split(temp_train_studies, test_size=val_size/(1-test_size), stratify=temp_train_studies['StudyHasCancer'], random_state=5)

    df.drop('StudyHasCancer', axis=1, inplace=True)

    # Get the training, validation and test data
    train_data = df[df['StudyUID'].isin(train_studies['StudyUID'])]
    val_data = df[df['StudyUID'].isin(val_studies['StudyUID'])]
    test_data = df[df['StudyUID'].isin(test_studies['StudyUID'])]

    return train_data, val_data, test_data

def create_datasets(train_data, val_data, test_data, dataset_path, transform, augmentations_list, n_augment):
    """
    Create datasets for training, validation and testing.
    """
    # Generate n_augment random combinations of augmentations
    random_combinations = []
    for _ in range(n_augment):
        n_elements = random.randint(1, len(augmentations_list))  # Randomly choose the number of elements in the combination
        combination = random.sample(augmentations_list, n_elements)  # Randomly select elements
        random_combinations.append(combination)

    # Create datasets
    normal_train_data = train_data[(train_data['Normal'] == 1)]
    augmented_train_data = train_data[(train_data['Actionable'] == 1) | (train_data['Benign'] == 1) | (train_data['Cancer'] == 1)]

    # Create dataset for normal class
    normal_train_dataset = BreastScanDataset(normal_train_data, dataset_path, transform)

    # Create datasets for underrepresented without augmentation
    original_train_dataset = BreastScanDataset(augmented_train_data, dataset_path, transform)

    # Now, for each combination, create an augmented dataset and add it to the list of datasets
    datasets = [normal_train_dataset, original_train_dataset]

    for combination in random_combinations:
        augmented_dataset = BreastScanDataset(augmented_train_data, dataset_path, transform, combination)
        datasets.append(augmented_dataset)

    # Concatenate all datasets
    train_dataset = torch.utils.data.ConcatDataset(datasets)

    val_dataset = BreastScanDataset(val_data, dataset_path, transform)
    test_dataset = BreastScanDataset(test_data, dataset_path, transform)

    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Create data loaders for training, validation and testing.
    """
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def print_class_distribution(train_data, n_augment):
    """
    Print the class distribution before and after augmentation.
    """
    # Calculate the number of samples in each class before augmentation
    original_class_counts = train_data.iloc[:, 3:].sum()
    print("Class distribution before augmentation:")
    print(original_class_counts)

    print("\nClass distribution after augmentation:")
    # Calculate the number of augmented samples
    augmented_train_data = train_data[(train_data['Actionable'] == 1) | (train_data['Benign'] == 1) | (train_data['Cancer'] == 1)]
    augmented_class_counts = augmented_train_data.iloc[:, 3:].sum() * n_augment

    # Calculate the total number of samples in each class after augmentation
    total_class_counts = original_class_counts + augmented_class_counts

    print(total_class_counts)

def print_split_distribution(data, split_name, n_augment):
    """
    This function prints the class distribution of a given split.

    Parameters:
    data (DataFrame): The data of the split.
    split_name (str): The name of the split (e.g., 'Train', 'Validation', 'Test').
    """
    # Calculate the number of samples in each class
    class_counts = data.iloc[:, 3:].sum()

    if split_name=="Train":
        # Calculate the number of augmented samples
        augmented_train_data = data[(data['Actionable'] == 1) | (data['Benign'] == 1) | (data['Cancer'] == 1)]
        augmented_class_counts = augmented_train_data.iloc[:, 3:].sum() * n_augment

        # Calculate the total number of samples in each class after augmentation
        class_counts += augmented_class_counts


    print(f"\n{split_name} class distribution:")
    print(class_counts)

    return class_counts

