from utils import *

def main():
    
    model_name, transfer_learning, verbose = parse_arguments()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    dataset_path ='/data/md311/Breast_Cancer_Detection_DBT/data/' 
    df = read_dataset(dataset_path)

    train_loader, test_loader, train_dataset, test_dataset = preprocess_dataset(df, dataset_path)
    

    unique_labels = df.columns.values[3:]
    num_classes = len(unique_labels)

    model_path='/data/md311/Breast_Cancer_Detection_DBT/models/dbt_classification_'+model_name+'_'+transfer_learning+'.pth'
    save_fig_path = '/data/md311/Breast_Cancer_Detection_DBT/fig/'
    
    # load model
    model, hyperparameters, num_epochs, n_splits = load_model_single(model_name, num_classes, model_path)

    model.to(device)
    model_name += '_'+transfer_learning

    print("--------------\n",test_dataset.labels)

    start_time = time.time()

    _ = evaluate_model(model, test_loader, device, unique_labels, model_name, save_fig_path)

    end_time = time.time()
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print(f"\n______________________\nModel evaluation complete.\nTotal evaluation time: {seconds:.2f} seconds ({h:02.0f}:{m:02.0f}:{s:02.0f})")


if __name__ == "__main__":
    main()