from utils import *

def hyperparameter_tuning(model_name, verbose, device, num_trials):
    
    dataset_path ='/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)

    # Initialise Random Values
    learning_rates = [10**(-random.uniform(3, 5)) for _ in range(num_trials)]
    beta1_values = [random.uniform(0, 1) for _ in range(num_trials)]
    beta2_values = [random.uniform(0, 1) for _ in range(num_trials)]
    weight_decays = [10**(-random.uniform(3, 5)) for _ in range(num_trials)]
    n_augment_values = [random.randint(0, 14) for _ in range(num_trials)]
    if model_name=="ConvNeXt":
        n_freeze_values = [random.randint(0, 2) for _ in range(num_trials)]
    elif model_name=="ViT":
        n_freeze_values = [random.randint(0, 3) for _ in range(num_trials)]
    elif model_name=="Swin":
        n_freeze_values = [random.randint(0, 5) for _ in range(num_trials)]
    else:
        raise Exception("Error: Invalid model name.")

    num_epochs = 12

    best_auc = 0
    best_params = {}

    for trial in range(num_trials):
        
        learning_rate = learning_rates[trial]
        beta1 = beta1_values[trial]
        beta2 = beta2_values[trial]
        weight_decay = weight_decays[trial]
        n_augment = n_augment_values[trial]
        n_freeze = n_freeze_values[trial]

        print(f"\n________________________\nTRIAL {trial}/{num_trials}\n")

        print(f"Learning Rate: {learning_rate}")
        print(f"Beta1: {beta1}")
        print(f"Beta2: {beta2}")
        print(f"Weight Decay: {weight_decay}")
        print(f"n_augment: {n_augment}")
        print(f"n_freeze: {n_freeze}\n")

        hyperparameters = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay
        }

        num_classes = 4

        model, _, _, batch_size, img_size, _ = load_model(model_name, num_classes, n_layers_to_freeze=n_freeze)

        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size, img_size)
    
        unique_labels = df.columns.values[3:]

        # load model

        criterion, optimizer = get_loss_optimizer(model, hyperparameters, class_counts, device)

        save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/ht_fig/test_{date:%d-%m-%Y_%H:%M:%S}_'.format(date=datetime.now())

        metrics = train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_freeze, model_name, save_weights=None, \
                    save_fig=save_fig_path, evaluate=True)

        hyperparameters.update({
            "batch_size": batch_size,
            "n_augment": n_augment,
            "n_freeze": n_freeze
        })
        hyperparameters.update(metrics)
        
        save_csv_path = '/data/md311/Breast_Cancer_Detection_DBT/models/hyperparameters.csv'
        save_parameters_to_csv(save_csv_path, hyperparameters, model_name)

        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_params = hyperparameters

    return best_params, best_auc

def main():

    model_name, verbose, num_trials = parse_arguments()
    if num_trials == 0:
        raise Exception("Error: --trials must be specified.")

    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    best_params, best_auc = hyperparameter_tuning(model_name, verbose, device, num_trials)
    end_time = time.time()
    
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(f"______________________\nHyperparameter tuning complete ({num_trials} trials).\nTime taken: {end_time - start_time:.2f} seconds ({h:02.0f}:{m:02.0f}:{s:02.0f})")
    print(f"\nBest AUC: {best_auc:.2f}")
    print(f"\nBest parameters & Results:\n")
    for param, value in best_params.items():
        print(f"{param.capitalize()}: {value}")

if __name__ == "__main__":
    main()