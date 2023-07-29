from utils import *
 
def fine_tuning(model_name, verbose, device):
    
    dataset_path = '/data/md311/Breast_Cancer_Detection_DBT/data/'
    save_weights_path = '/data/md311/Breast_Cancer_Detection_DBT/models/'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'

    # Read dataset
    df = read_dataset(dataset_path)

    n_augment = 16
    num_classes = 4

    model, hyperparameters, num_epochs, batch_size, img_size, n_layers_to_freeze = load_model(model_name, num_classes)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size, img_size, save_fig_path)
    
    unique_labels = df.columns.values[3:]
    
    if verbose:

        image, label = next(iter(train_dataset))

        # Print the shape of the image
        print("\nInput image shape:",image.shape,"\n")


        # summary(model=model, 
        #     input_size=(batch_size, image.shape[0], image.shape[1], image.shape[2]),
        #     col_names=["input_size", "output_size", "num_params", "trainable"],
        #     col_width=20,
        #     row_settings=["var_names"]
        # )

    criterion, optimizer = get_loss_optimizer(model, hyperparameters, class_counts, device)
    
    train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_layers_to_freeze, model_name, save_weights=save_weights_path, \
                    save_fig=save_fig_path, evaluate=True)
    
def training_from_scratch(model_name, verbose, device):
    
    dataset_path = '/data/md311/Breast_Cancer_Detection_DBT/data/'
    save_weights_path = '/data/md311/Breast_Cancer_Detection_DBT/models/'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'

    # Read dataset
    df = read_dataset(dataset_path)

    n_augment = 16
    num_classes = 4

    model, hyperparameters, num_epochs, batch_size, img_size, n_layers_to_freeze = load_model(model_name, num_classes, n_layers_to_freeze=0)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size, img_size, save_fig_path)
    
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

    criterion, optimizer = get_loss_optimizer(model, hyperparameters, class_counts, device)
    
    train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_layers_to_freeze, model_name, save_weights=save_weights_path, \
                    save_fig=save_fig_path, evaluate=True)
    
def main():

    model_name, verbose, _ = parse_arguments()

    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == 'cuda':
        torch.cuda.empty_cache()

    print("Device",device)

    # fine_tuning(model_name, verbose, device)

    training_from_scratch(model_name, verbose, device)


if __name__ == "__main__":
    main()