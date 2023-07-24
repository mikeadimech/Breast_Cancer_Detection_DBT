from utils import *

def load_model(model_name, num_classes, from_path=None, n_layers_to_freeze=1):
    if model_name=="ResNet":
        num_epochs = 2
        batch_size = 32
        img_size = 224
        n_layers_to_freeze = 1
        model = load_resnet50_model(num_classes, from_path)
        model = freeze_layers(model, n_layers_to_freeze)
        hyperparameters = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0001
        }
    elif model_name=="ConvNeXt":
        num_epochs = 2
        batch_size = 32
        img_size = 518
        n_layers_to_freeze = 1
        model = load_convnext_model(num_classes, from_path)
        model = freeze_layers(model, n_layers_to_freeze)
        hyperparameters = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0001
        }
    elif model_name=="ViT":
        num_epochs = 3
        batch_size = 16
        img_size = 384 #fixed
        n_layers_to_freeze = 1
        model = load_vit_model(num_classes, from_path)
        model = freeze_layers(model, n_layers_to_freeze)
        hyperparameters = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0001
        }
        
    elif model_name=="Swin":
        num_epochs = 4
        batch_size = 64
        img_size = 518
        n_layers_to_freeze = 4
        model = load_swin_model(num_classes, from_path)
        model = freeze_layers(model, n_layers_to_freeze)
        hyperparameters = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.0001
        }
    else:
        raise Exception("Invalid model name.")

    return model, hyperparameters, num_epochs, batch_size, img_size, n_layers_to_freeze

def load_resnet50_model(num_classes, saved_weights_path=None):
    # Load the pretrained model
    model = models.resnet50(weights='DEFAULT')
    
    # Get the input of the fc layer
    num_ftrs = model.fc.in_features
    
    # Re-define the fc layer / classifier
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the model weights from the saved file
    if saved_weights_path is not None:
        model.load_state_dict(torch.load(saved_weights_path))
    
    return model

def load_convnext_model(num_classes, saved_weights_path=None):
    # Load the pretrained model
    model = models.convnext_large(weights='DEFAULT')
    
    # Get the number of input features to the last layer
    num_ftrs = model.classifier[2].in_features

    # Modify the final layer for binary classification
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # Load the model weights from the saved file
    if saved_weights_path is not None:
        model.load_state_dict(torch.load(saved_weights_path))
    
    return model

def load_vit_model(num_classes, saved_weights_path=None):
    # Load the pretrained model
    model = models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)

    print(model)
    
    # Get the number of input features to the last layer
    num_features = model.heads.head.in_features

    # Replace the final layer to match the number of classes in your dataset
    model.heads.head = nn.Linear(num_features, num_classes)

    # Load the model weights from the saved file
    if saved_weights_path is not None:
        model.load_state_dict(torch.load(saved_weights_path))
    
    return model

def load_swin_model(num_classes, saved_weights_path=None):
    # Load the pretrained model
    model = models.swin_v2_b(weights='DEFAULT')

    print(model)
    
    # Get the number of input features to the last layer
    num_ftrs = model.head.in_features

    # Modify the final layer for binary classification
    model.head = nn.Linear(num_ftrs, num_classes)

    # Load the model weights from the saved file
    if saved_weights_path is not None:
        model.load_state_dict(torch.load(saved_weights_path))
    
    return model

def freeze_layers(model, n_layers_to_freeze):
    layers = list(model.children())[:n_layers_to_freeze]
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False
    return model

def get_loss_optimizer(model, hyperparameters, class_counts, device):
    # Calculate class weights
    class_weights = 1. / class_counts
    class_weights = class_weights / np.sum(class_weights)  # Normalize to make the weights sum to 1
    class_weights = torch.tensor(class_weights.values).float().to(device)  # Convert to a PyTorch tensor and move to device

    print("class weights:",class_weights)

    # Define the loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define the optimizer
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=hyperparameters['learning_rate'], 
                           betas=(hyperparameters['beta1'], hyperparameters['beta2']),
                           weight_decay=hyperparameters['weight_decay'])
    
    return criterion, optimizer


def train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_freeze, model_name, save_weights=None, \
                    save_fig=None, evaluate=True):
    
    # log training process
    run = wandb.init(
        project="breast-cancer-dbt-"+model_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": optimizer.param_groups[-1]['lr'],
            "epochs": num_epochs,
            "batch_size:": batch_size,
            "n_augment": n_augment,
            "n_augment": n_freeze
        }
    )
    
    model.to(device)

    train_losses = []
    val_losses = []
    epochs = []
    train_roc_auc_scores = []
    val_roc_auc_scores = []
    train_balanced_accuracy_scores = []
    val_balanced_accuracy_scores = []

    # set timer
    start_time = time.time()

    # Train the model
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Compute predictions, probabilities, and collect labels for ROC AUC and balanced accuracy
            _, train_preds = torch.max(outputs, 1)
            train_probs = torch.nn.functional.softmax(outputs, dim=1)
            all_train_preds.extend(train_preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            all_train_probs.append(train_probs.detach().cpu().numpy())

        all_train_probs = np.vstack(all_train_probs)  # Stack the probabilities

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_epoch_loss)
        epochs.append(epoch)

        # Compute ROC AUC and balanced accuracy for the training set
        train_roc_auc = roc_auc_score(all_train_labels, all_train_probs, average='macro', multi_class='ovr')
        train_roc_auc_scores.append(train_roc_auc)
        
        train_bal_acc = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_balanced_accuracy_scores.append(train_bal_acc)

        print(f'Train Loss: {train_epoch_loss:.4f}  |  Train ROC AUC: {train_roc_auc:.4f}  |  Train Balanced Accuracy: {train_bal_acc:.4f}')

        # Validation
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_probs = np.vstack(all_probs)  # Stack the probabilities

        val_epoch_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        val_roc_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        val_roc_auc_scores.append(val_roc_auc)

        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        val_balanced_accuracy_scores.append(val_bal_acc)

        print(f'Validation Loss: {val_epoch_loss:.4f}  |  Validation ROC AUC: {val_roc_auc:.4f}  |  Validation Balanced Accuracy: {val_bal_acc:.4f}')

        wandb.log({"train_loss":train_epoch_loss, "val_loss":val_epoch_loss, "roc_auc_train":train_roc_auc, "roc_auc_val":val_roc_auc, "balanced_accuracy_train":train_bal_acc, "balanced_accuracy_val":val_bal_acc})
    
    end_time = time.time()
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print(f"\n______________________\nModel training complete.\nTotal training time: {seconds:.2f} seconds ({h:02.0f}:{m:02.0f}:{s:02.0f})")

    # save model weights
    if save_weights is not None:
        torch.save(model.state_dict(), save_weights+'dbt_classification_'+model_name+'.pth')
        print("Weights saved to",save_weights+'dbt_classification_'+model_name+'.pth')
    
    # plot loss
    if save_fig is not None:
        plot_loss(epochs, train_losses, val_losses, save_fig, model_name)
        plot_roc_auc(epochs, train_roc_auc_scores, val_roc_auc_scores, save_fig, model_name)
        plot_bal_acc(epochs, train_balanced_accuracy_scores, val_balanced_accuracy_scores, save_fig, model_name)

    if evaluate==True:
        metrics = evaluate_model(model, test_loader, device, unique_labels, model_name, save_fig)
        wandb.log(metrics)
    else:
        metrics = {}

    wandb.finish()

    metrics['training_time'] = f"{h:02.0f}:{m:02.0f}:{s:02.0f}"

    return metrics

def train_model_cv(model, criterion, optimizer, train_dataset, train_loader, test_loader, unique_labels, \
                   device, num_epochs, batch_size, n_splits, n_augment, model_name, save_weights=None, save_fig=None, evaluate=True):
    
    # log training process
    run = wandb.init(
        project="breast-cancer-dbt-"+model_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": optimizer.param_groups[-1]['lr'],
            "epochs": num_epochs,
            "batch_size:": batch_size,
            "n_splits": n_splits,
            "n_augment": n_augment
        }
    )
    
    model.to(device)
    
    # Initialize the KFold class
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Loop over all datasets in the ConcatDataset
    labels = []

    # Iterate over each Dataset in the ConcatDataset
    for dataset in train_dataset.datasets:
        # Convert one-hot encoded labels to categorical labels and add them to the list
        labels.extend(np.argmax(dataset.data.iloc[:, 3:].values, axis=1))

    # Convert the list to a numpy array
    labels = np.array(labels)

    train_losses = []
    val_losses = []
    epochs = []
    roc_auc_scores = []
    balanced_accuracy_scores= []

    # set timer
    start_time = time.time()

    # Start the k-fold cross-validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset, labels)):
        # Print
        print(f'\nFOLD {fold+1}/{n_splits}')
        print('--------------------------------')

        # Define the data loaders for training and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)

        # Train the model
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Training"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            train_epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Train Loss: {train_epoch_loss:.4f}')
            train_losses.append(train_epoch_loss)
            epochs.append(epoch)

            # Validation
            model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    running_loss += loss.item() * inputs.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

            all_probs = np.vstack(all_probs)  # Stack the probabilities


            val_epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)

            roc_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
            roc_auc_scores.append(roc_auc)

            bal_acc = balanced_accuracy_score(all_labels, all_preds)
            balanced_accuracy_scores.append(bal_acc)

            print(f'Validation Loss: {val_epoch_loss:.4f}  |  ROC AUC: {roc_auc:.4f}  |  Balanced Accuracy: {bal_acc:.4f}')

            wandb.log({"train_loss": train_epoch_loss, "val_loss": val_epoch_loss, "roc_auc":roc_auc, "balanced_accuracy":bal_acc})
    
    end_time = time.time()
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print(f"\n______________________\nModel training complete.\nTotal training time: {seconds:.2f} seconds ({h:02.0f}:{m:02.0f}:{s:02.0f})")

    # save model weights
    if save_weights is not None:
        torch.save(model.state_dict(), save_weights+'dbt_classification_'+model_name+'.pth')
        print("Weights saved to",save_weights+'dbt_classification_'+model_name+'.pth')
    
    # plot loss
    if save_fig is not None:
        plot_loss(epochs, train_losses, val_losses, save_fig, model_name)
        plot_roc_auc(epochs, roc_auc_scores, save_fig, model_name)

    if evaluate==True:
        metrics = evaluate_model(model, test_loader, device, unique_labels, model_name, save_fig)
        wandb.log(metrics)
        

    else:
        metrics = {}

    wandb.finish()

    metrics['training_time'] = f"{h:02.0f}:{m:02.0f}:{s:02.0f}"

    return metrics

def evaluate_model(model, test_loader, device, unique_labels, model_name, save_fig=None):
    
    print("\n--------------------------------\nEvaluating Model...\n")
    
    model.eval()  # Set the model to evaluation mode
    
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on test set..."):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get the predicted probabilities for all classes
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())  # Append the probabilities as a 2D array

    all_probs = np.vstack(all_probs)  # Stack the probabilities

    # Calculate metrics
    metrics = {}

    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
    metrics['precision'] = precision_score(all_labels, all_preds, average='macro')
    metrics['recall'] = recall_score(all_labels, all_preds, average='macro')
    metrics['f1'] = f1_score(all_labels, all_preds, average='macro')
    metrics['roc_auc'] = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')

    # Display metrics from the dictionary
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")

    classification_report(all_labels, all_preds)#, labels=unique_labels)

    if save_fig is not None:
    
        plot_confusion_matrix(all_labels,all_preds, unique_labels, save_fig, model_name)

        plot_roc_curve(np.eye(len(unique_labels))[all_labels], all_probs, len(unique_labels), unique_labels, save_fig, model_name)
    
    else:
        conf_mat = pd.DataFrame(confusion_matrix(all_labels,all_preds), index=unique_labels, columns=unique_labels)
        print('\nConfusion Matrix:\n',conf_mat,'\n',sep='')
    
    if wandb.run is not None:
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=all_labels, preds=all_preds,
                        class_names=unique_labels)})


    return metrics