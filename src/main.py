from utils import *
 
def fine_tuning(model_name, verbose, device):
    # Read dataset
    dataset_path ='/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)

    n_augment = 2
    num_classes = 4

    model, hyperparameters, num_epochs, batch_size = load_model(model_name, num_classes)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size)
    
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
    save_weights_path = '/data/md311/Breast_Cancer_Detection_DBT/models/'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'
    
    # train_model_cv(model, criterion, optimizer, train_dataset, train_loader, test_loader, unique_labels, \
                #    device, num_epochs, batch_size, n_splits, n_augment, model_name, save_weights_path,  \
                #     save_fig_path, evaluate=True) 
    
    train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, model_name, save_weights=save_weights_path, \
                    save_fig=save_fig_path, evaluate=True)
    
def main():

    model_name, verbose, _ = parse_arguments()

    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fine_tuning(model_name, verbose, device)


if __name__ == "__main__":
    main()