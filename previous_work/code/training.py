import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import functional as F


def consistency_loss(output, labels, edge_index, weight):
    # Cross-entropy loss
    ce_loss = F.cross_entropy(output, labels)

    # Extract predicted labels
    predicted_labels = torch.argmax(output, dim=1)

    # Mask where predictions match the true labels
    correct_mask = (predicted_labels == labels)

    # Create a mask for edges where both nodes have correct predictions
    edge_mask = correct_mask[edge_index[0]] & correct_mask[edge_index[1]]

    # Filter the edge_index to only include edges where the mask is True
    filtered_edge_index = edge_index[:, edge_mask]

    # Compute consistency loss for the filtered edges
    if filtered_edge_index.size(1) > 0:
        node_i = filtered_edge_index[0]
        node_j = filtered_edge_index[1]
        con_loss = F.mse_loss(output[node_i], output[node_j], reduction='mean')
    else:
        con_loss = torch.tensor(0.0, device=output.device)

    total_loss = ce_loss + weight * con_loss
    return total_loss


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, cons_loss=False, lambda_start=0.1,
          lambda_end=1.0, device="cpu"):
    train_accs, val_losses, train_losses, f1_scores, precs, recalls = [], [], [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.float()
            optimizer.zero_grad()
            out = model(data)
            if cons_loss:
                lambda_consistency = lambda_start + (lambda_end - lambda_start) * (epoch / num_epochs)
                loss = consistency_loss(out, data.y.view(-1), data.edge_index, lambda_consistency)
            else:
                loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # * data.x.size(0)

        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.float()
                out = model(data)
                if cons_loss:
                    lambda_consistency = lambda_start + (lambda_end - lambda_start) * (epoch / num_epochs)
                    loss = consistency_loss(out, data.y.view(-1), data.edge_index, lambda_consistency)
                else:
                    loss = criterion(out, data.y.view(-1))
                val_loss += loss.item()  # * data.x.size(0)

                # Get predictions
                pred = out.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(data.y.cpu())

        # Combine all predictions and labels for metric calculation
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        train_accs.append(accuracy)
        val_losses.append(val_loss / len(val_loader))
        train_losses.append(epoch_loss / len(train_loader))
        f1_scores.append(f1)
        precs.append(precision)
        recalls.append(recall)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Training Loss: {epoch_loss / len(train_loader):.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('-' * 50)

    return train_accs, val_losses, train_losses, f1_scores, precs, recalls


def train_with_weights(model, classifier, optimizer, optimizer_classifier, criterion, train_loader, val_loader, num_epochs, cons_loss=False, lambda_start=0.1,
          lambda_end=1.0, device="cpu"):
    train_accs, val_losses, train_losses, f1_scores, precs, recalls = [], [], [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        epoch_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            data.x = data.x.float()
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            embedding = model(data)
            embedding = embedding.detach()
            out = classifier(embedding)
            if cons_loss:
                lambda_consistency = lambda_start + (lambda_end - lambda_start) * (epoch / num_epochs)
                loss = consistency_loss(out, data.y.view(-1), data.edge_index, lambda_consistency)
            else:
                loss = criterion(out, data.y.view(-1))
            loss.backward()
            optimizer_classifier.step()
            optimizer.step()

            epoch_loss += loss.item()  # * data.x.size(0)

        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.float()
                embedding = model(data)
                embedding = embedding.detach()
                out = classifier(embedding)
                if cons_loss:
                    lambda_consistency = lambda_start + (lambda_end - lambda_start) * (epoch / num_epochs)
                    loss = consistency_loss(out, data.y.view(-1), data.edge_index, lambda_consistency)
                else:
                    loss = criterion(out, data.y.view(-1))
                val_loss += loss.item()  # * data.x.size(0)

                # Get predictions
                pred = out.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(data.y.cpu())

        # Combine all predictions and labels for metric calculation
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        train_accs.append(accuracy)
        val_losses.append(val_loss / len(val_loader))
        train_losses.append(epoch_loss / len(train_loader))
        f1_scores.append(f1)
        precs.append(precision)
        recalls.append(recall)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Training Loss: {epoch_loss / len(train_loader):.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('-' * 50)

    return train_accs, val_losses, train_losses, f1_scores, precs, recalls
