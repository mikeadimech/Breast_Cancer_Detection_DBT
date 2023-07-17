from utils import *

    
def single_stage_transfer_learning(model_name, verbose, device):
    # Read dataset
    dataset_path ='/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)

    n_augment = 2
    batch_size = 32

    train_loader, test_loader, train_dataset, test_dataset = preprocess_dataset(df, dataset_path, n_augment, batch_size)
    

    unique_labels = df.columns.values[3:]
    num_classes = len(unique_labels)

    print("num classes ",num_classes)
    
    # load model
    model, hyperparameters, num_epochs, n_splits = load_model_single(model_name, num_classes)

    if verbose:
        summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )

    model_name += '_single'

    criterion, optimizer = get_loss_optimizer(model, hyperparameters)
    save_weights_path = '/data/md311/Breast_Cancer_Detection_DBT/models/'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'
    
    train_model_cv(model, criterion, optimizer, train_dataset, train_loader, test_loader, unique_labels, \
                   device, num_epochs, batch_size, n_splits, n_augment, model_name, save_weights_path,  \
                    save_fig_path, evaluate=True) 
    
def main():

    model_name, transfer_learning, verbose, _ = parse_arguments()

    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if transfer_learning=="single":
        single_stage_transfer_learning(model_name, verbose, device)
    elif transfer_learning=="multi":
        print("area under construction")
    else:
        raise Exception("Invalid input.")


if __name__ == "__main__":
    main()