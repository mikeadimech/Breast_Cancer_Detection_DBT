from utils import *

def main():
    
    model_name, verbose, _ = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path ='/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)

    n_augment = 2
    num_classes = 4
    batch_size = 128

    model, hyperparameters, num_epochs, batch_size = load_model(model_name, num_classes)
    model.to(device)
    
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_counts = preprocess_dataset(df, dataset_path, n_augment, batch_size)

    unique_labels = df.columns.values[3:]

    model_path='/data/md311/Breast_Cancer_Detection_DBT/models/dbt_classification_'+model_name+'.pth'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'

    start_time = time.time()

    _ = evaluate_model(model, test_loader, device, unique_labels, model_name, save_fig_path)

    end_time = time.time()
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(f"\n______________________\nModel evaluation complete.\nTotal evaluation time: {seconds:.2f} seconds ({h:02.0f}:{m:02.0f}:{s:02.0f})")


if __name__ == "__main__":
    main()