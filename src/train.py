from utils import *

def load_model(model_name, num_classes, from_path=None, img_size=512):
    if model_name=="ConvNeXt":
        num_epochs = 6
        batch_size = 16
        n_layers_to_freeze = 2
        if img_size==512:
            model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_512', pretrained=True, num_classes=4)
        else:
            raise Exception("Invalid image size for ConvNeXt model - only 512 accepted.")
        model = freeze_layers(model, n_layers_to_freeze)
        hyperparameters = {
            'learning_rate': 0.001,
            'beta1': 0.8,
            'beta2': 0.999,
            'weight_decay': 0.0001
        }
    elif model_name=="MaxViT":
        num_epochs = 7
        batch_size = 16
        n_layers_to_freeze = 2
        if img_size==224:
            model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=True, num_classes=4)
        elif img_size==384:
            model = timm.create_model('maxvit_base_tf_384.in21k_ft_in1k', pretrained=True, num_classes=4)
        elif img_size==512:
            model = timm.create_model('maxvit_base_tf_512.in21k_ft_in1k', pretrained=True, num_classes=4)
        model = freeze_layers(model, n_layers_to_freeze)
        hyperparameters = {
            'learning_rate': 0.01,
            'beta1': 0.9,
            'beta2': 0.99,
            'weight_decay': 0.001
        }
    else:
        raise Exception("Invalid model name.")

    if from_path is not None:
        model.load_state_dict(torch.load(from_path))

    return model, hyperparameters, num_epochs, batch_size, img_size, n_layers_to_freeze

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
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(gamma=0.7, weights=class_weights)
    print("Using Focal Loss...")
    
    # Define the optimizer
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=hyperparameters['learning_rate'], 
                           betas=(hyperparameters['beta1'], hyperparameters['beta2']),
                           weight_decay=hyperparameters['weight_decay'])
    
    return criterion, optimizer


def train_model(model, criterion, optimizer, train_loader, val_loader, test_loader, \
                    unique_labels, device, num_epochs, batch_size, n_augment, n_freeze, model_name, save_weights=None, \
                    save_fig=None, evaluate=True, trial=None):
    
    # log training process
    run = wandb.init(
        project="breast-cancer-dbt-"+model_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": optimizer.param_groups[-1]['lr'],
            "weight_decay": optimizer.param_groups[-1]['weight_decay'],
            "beta1": optimizer.param_groups[-1]['betas'][0],
            "beta2": optimizer.param_groups[-1]['betas'][1],
            "epochs": num_epochs,
            "batch_size:": batch_size,
            "n_augment": n_augment,
            "n_freeze": n_freeze
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

            # Compute predictions, probabilities, and collect labels for ROC AUC and balanced accuracy
            _, train_preds = torch.max(outputs, 1)
            train_probs = torch.nn.functional.softmax(outputs, dim=1)

            loss = criterion(train_probs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

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

        conf_mat_train = pd.DataFrame(confusion_matrix(all_train_labels, all_train_preds), index=unique_labels, columns=unique_labels)
        print('\nConfusion Matrix (Train):\n',conf_mat_train,'\n',sep='')

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
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                loss = criterion(probs, labels)
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

        conf_mat = pd.DataFrame(confusion_matrix(all_labels, all_preds), index=unique_labels, columns=unique_labels)
        print('\nConfusion Matrix (Val):\n',conf_mat,'\n',sep='')

        wandb.log({"train_loss":train_epoch_loss, "val_loss":val_epoch_loss, "roc_auc_train":train_roc_auc, "roc_auc_val":val_roc_auc, "balanced_accuracy_train":train_bal_acc, "balanced_accuracy_val":val_bal_acc})
    
        # Pruning based on the intermediate value.
        if trial is not None:
            trial.report(val_bal_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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

    print("unique labels\n",unique_labels)
    print('\n',classification_report(all_labels, all_preds, target_names=unique_labels),'\n',sep='')

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