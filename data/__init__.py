from utils import *
from data.dataset import BreastScanDataset

def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path+'data_csv/train-v2_table_list_slice.csv')
    df.drop(df.tail(16500).index, axis=0, inplace=True)
    return df

def mean_std(loader):
    images, labels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    return mean, std

def preprocess_dataset(df, dataset_path, n_augment, batch_size):

    # Define transformations
    transform = transforms.Compose([
        # transforms.Resize((224, 224), antialias=True),  # Resize images to fit ResNet input size
        transforms.Resize((518, 518), antialias=True), 
        # transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet mean and std
        transforms.Normalize(mean=[0.3283, 0.3283, 0.3283], std=[0.4667, 0.4667, 0.4667])  # Normalize to dataset mean and std

    ])

    # Define the list of possible augmentations
    augmentations_list = [
        transforms.RandomRotation(10),
        # transforms.RandomResizedCrop(518, scale=(0.8, 1.0), antialias=True),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0))

        # NO
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
    ]

    # Split the data into train, validation and test sets
    train_data, val_data, test_data = split_data(df)

    # Create datasets for training, validation and testing
    train_dataset, val_dataset, test_dataset = create_datasets(train_data, val_data, test_data, dataset_path, transform, augmentations_list, n_augment)

    # Create data loaders for training, validation and testing
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    mean, std = mean_std(train_loader)
    print("mean and std: \n", mean, std)

    # Print the class distribution before and after augmentation
    print_class_distribution(train_data, n_augment)


    class_counts = print_split_distribution(train_data, 'Train', n_augment)
    _ = print_split_distribution(val_data, 'Validation', n_augment)
    _ = print_split_distribution(test_data, 'Test', n_augment)


    # # Create a new column indicating whether each study includes at least one image where 'Cancer' is 1
    # df['StudyHasCancer'] = df.groupby('StudyUID')['Cancer'].transform('max')

    # # Get the unique studies and their cancer status
    # unique_studies = df[['StudyUID', 'StudyHasCancer']].drop_duplicates()

    # # Split the studies using stratified sampling based on the 'StudyHasCancer' column
    # train_studies, test_studies = train_test_split(unique_studies, test_size=0.2, stratify=unique_studies['StudyHasCancer'], random_state=42)

    
    # df.drop('StudyHasCancer', axis=1, inplace=True)

    # # Get the training and test data
    # train_data = df[df['StudyUID'].isin(train_studies['StudyUID'])]
    # test_data = df[df['StudyUID'].isin(test_studies['StudyUID'])]

    # # Generate n_augment random combinations of augmentations
    # random_combinations = []
    # for _ in range(n_augment):
    #     n_elements = random.randint(1, len(augmentations_list))  # Randomly choose the number of elements in the combination
    #     combination = random.sample(augmentations_list, n_elements)  # Randomly select elements
    #     random_combinations.append(combination)

    # # Create datasets
    # normal_train_data = train_data[(train_data['Normal'] == 1)]
    # augmented_train_data = train_data[(train_data['Actionable'] == 1) | (train_data['Benign'] == 1) | (train_data['Cancer'] == 1)]


    # # Create dataset for normal class
    # normal_train_dataset = BreastScanDataset(normal_train_data, dataset_path, transform)

    # # Create datasets for underrepresesnted without augmentation
    # original_train_dataset = BreastScanDataset(augmented_train_data, dataset_path, transform)

    # # Now, for each combination, create an augmented dataset and add it to the list of datasets
    # datasets = [normal_train_dataset, original_train_dataset]

    # for combination in random_combinations:
    #     augmented_dataset = BreastScanDataset(augmented_train_data, dataset_path, transform, combination)
    #     datasets.append(augmented_dataset)

    # # Concatenate all datasets
    # train_dataset = torch.utils.data.ConcatDataset(datasets)

    # test_dataset = BreastScanDataset(test_data, dataset_path, transform)

    # # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # # Calculate the number of samples in each class before augmentation
    # original_class_counts = train_data.iloc[:, 3:].sum()
    # print("Class distribution before augmentation:")
    # print(original_class_counts)


    # print("\nClass distribution after augmentation:")
    # # Calculate the number of augmented samples
    # augmented_class_counts = augmented_train_data.iloc[:, 3:].sum() * n_augment

    # # Calculate the total number of samples in each class after augmentation
    # total_class_counts = original_class_counts + augmented_class_counts

    # print(total_class_counts)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts

def split_data(df, test_size=0.2, val_size=0.2):
    """
    Split the data into train, validation and test sets.
    """
    # Create a new column indicating whether each study includes at least one image where 'Cancer' is 1
    df['StudyHasCancer'] = df.groupby('StudyUID')['Cancer'].transform('max')

    # Get the unique studies and their cancer status
    unique_studies = df[['StudyUID', 'StudyHasCancer']].drop_duplicates()

    # Split the studies into a temporary train set and the test set
    temp_train_studies, test_studies = train_test_split(unique_studies, test_size=test_size, stratify=unique_studies['StudyHasCancer'], random_state=42)

    # Split the temporary train set into the final train set and the validation set
    train_studies, val_studies = train_test_split(temp_train_studies, test_size=val_size/(1-test_size), stratify=temp_train_studies['StudyHasCancer'], random_state=42)

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

