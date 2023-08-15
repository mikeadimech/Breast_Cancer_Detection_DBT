from utils import *
 
def fine_tuning(model_name, img_size, verbose, device):
    """
    Fine-tunes a pre-trained model for breast cancer image classification.

    Args:
        model_name (str): Name of the model architecture (e.g., 'MaxViT', 'ConvNeXt').
        img_size (int): Size of the input images (224, 384, or 512).
        verbose (bool): If True, print additional information during execution.
        device (torch.device): Device to use for training ('cuda' or 'cpu').

    Returns:
        None
    """
    
    dataset_path = '/data/md311/Breast_Cancer_Detection_DBT/data/'
    save_weights_path = '/data/md311/Breast_Cancer_Detection_DBT/models/'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'

    # Read dataset
    df = read_dataset(dataset_path)

    n_augment = 0
    num_classes = 4

    model, hyperparameters, num_epochs, batch_size, img_size, n_layers_to_freeze = load_model(model_name, num_classes, img_size=img_size)

    train_loader, val_loader, test_loader, train_dataset, _, _, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size, img_size, save_fig_path)
    
    unique_labels = df.columns.values[3:]
    
    if verbose:

        image, label = next(iter(train_dataset))

        # Print the shape of the image
        print("\nInput image shape:",image.shape,"\n")


        summary(model=model, 
            input_size=(batch_size, image.shape[0], image.shape[1], image.shape[2]),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )
    
    print(f"\nTRAINING\n")

    print(f"Learning Rate: {hyperparameters['learning_rate']}")
    print(f"Beta1: {hyperparameters['beta1']}")
    print(f"Beta2: {hyperparameters['beta2']}")
    print(f"Weight Decay: {hyperparameters['weight_decay']}")
    print(f"n_augment: {n_augment}")
    print(f"n_freeze: {n_layers_to_freeze}\n")

    criterion, optimizer = get_loss_optimizer(model, hyperparameters, class_counts, device)
    
    train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_layers_to_freeze, model_name, save_weights=save_weights_path, \
                    save_fig=save_fig_path, evaluate=True)
    
def main():

    model_name, verbose, _, img_size = parse_arguments()

    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == 'cuda':
        torch.cuda.empty_cache()

    print("Device",device)

    fine_tuning(model_name, img_size, verbose, device)

if __name__ == "__main__":
    main()