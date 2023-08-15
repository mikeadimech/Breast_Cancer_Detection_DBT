from utils import *

def objective(trial, device, model_name):
    """
    Optuna objective function for hyperparameter tuning.

    Args:
        trial: Optuna Trial object.
        device: Device for training (cuda or cpu).
        model_name (str): Name of the model.

    Returns:
        float: The ROC AUC metric to be minimized.
    """
    dataset_path = '/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)
    
    # Define an objective function to be minimized.
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    beta1 = trial.suggest_float('beta1', 0.8, 1.0)
    beta2 = trial.suggest_float('beta2', 0.9, 1.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    n_augment = trial.suggest_int('n_augment', 0, 25)

    print("\nNEW TRIAL\n")
    print(f"Learning Rate: {learning_rate}")
    print(f"Beta1: {beta1}")
    print(f"Beta2: {beta2}")
    print(f"Weight Decay: {weight_decay}")
    print(f"n_augment: {n_augment}\n")

    hyperparameters = {
        'learning_rate': learning_rate,
        'beta1': beta1,
        'beta2': beta2,
        'weight_decay': weight_decay
    }

    num_classes = 4
    num_epochs = 5

    model, _, _, batch_size, img_size, n_freeze = load_model(model_name, num_classes)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size, img_size)
    
    unique_labels = df.columns.values[3:]

    criterion, optimizer = get_loss_optimizer(model, hyperparameters, class_counts, device)

    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/ht_fig/test_{date:%d-%m-%Y_%H:%M:%S}_'.format(date=datetime.now())

    try:
        metrics = train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_freeze, model_name, save_weights=None, \
                    save_fig=save_fig_path, evaluate=True, trial=trial)
        return metrics['roc_auc']
    except optuna.exceptions.TrialPruned:
        return float('inf')  # Return a large value to indicate pruning


def main():
    """
    Main function for hyperparameter tuning using Optuna.

    Returns:
        None
    """
    model_name, verbose, num_trials, _ = parse_arguments()
    if num_trials == 0:
        raise Exception("Error: --trials must be specified.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.login()

    # Use partial to provide additional arguments
    objective_with_args = partial(objective, device=device, model_name=model_name)

    study_path = '/data/md311/Breast_Cancer_Detection_DBT/models/'

    if os.path.exists(study_path+'optuna_study_'+model_name+'.pkl'):
        study = joblib.load(study_path+'optuna_study_'+model_name+'.pkl')
    else:
        study = optuna.create_study(study_name='study_'+model_name, direction='maximize', pruner=optuna.pruners.MedianPruner())
        joblib.dump(study, study_path+'optuna_study_'+model_name+'.pkl')
    
    start_time = time.time()

    study.optimize(objective_with_args, n_trials=num_trials)

    end_time = time.time()
    
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(f"______________________\nHyperparameter tuning complete ({num_trials} trials).\nTime taken: {end_time - start_time:.2f} seconds ({h:02.0f}:{m:02.0f}:{s:02.0f})")
    
    best_params = study.best_params
    best_auc = study.best_value
    
    print(f"\nBest AUC: {best_auc:.2f}")
    print(f"\nBest parameters & Results:\n")
    for param, value in best_params.items():
        print(f"{param.capitalize()}: {value}")

if __name__ == "__main__":
    main()