from utils import *

def hyperparameter_tuning(model_name, verbose, device, num_trials):
    
    dataset_path ='/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)

    learning_rates = [10**(-random.uniform(3, 5)) for _ in range(num_trials)]
    beta1_values = [random.uniform(0, 1) for _ in range(num_trials)]
    beta2_values = [random.uniform(0, 1) for _ in range(num_trials)]
    weight_decays = [10**(-random.uniform(3, 5)) for _ in range(num_trials)]
    batch_sizes = [2**random.randint(5, 7) for _ in range(num_trials)]
    n_augment_values = [random.randint(0, 14) for _ in range(num_trials)]

    num_epochs = 5
    n_splits = 7

    best_auc = 0
    best_params = {}

    for trial in range(num_trials):
        
        learning_rate = learning_rates[trial]
        beta1 = beta1_values[trial]
        beta2 = beta2_values[trial]
        weight_decay = weight_decays[trial]
        batch_size = batch_sizes[trial]
        n_augment = n_augment_values[trial]

        print(f"\n________________________\nTRIAL {trial}/{num_trials}\n")

        print(f"Learning Rate: {learning_rate}")
        print(f"Beta1: {beta1}")
        print(f"Beta2: {beta2}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Batch Size: {batch_size}")
        print(f"n_augment: {n_augment}\n")

        train_loader, test_loader, train_dataset, _ = preprocess_dataset(df, dataset_path, n_augment, batch_size)
    
        unique_labels = df.columns.values[3:]
        num_classes = len(unique_labels)
    
        # load model
        model, _, _, _, _ = load_model_single(model_name, num_classes)

        hyperparameters = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay
        }

        criterion, optimizer = get_loss_optimizer(model, hyperparameters)

        save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/test_{date:%d-%m-%Y_%H:%M:%S}_'.format(date=datetime.datetime.now())
        
        metrics = train_model_cv(model, criterion, optimizer, train_dataset, train_loader, test_loader, unique_labels, \
                    device, num_epochs, batch_size, n_splits, n_augment, model_name, save_weights=None, save_fig=save_fig_path, evaluate=True)

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