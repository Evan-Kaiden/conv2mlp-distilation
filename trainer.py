import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import utils
import dataset

def train(net, epochs, criterion, teachernet=None, alpha=1.0, beta=0.0):
    net.to(utils.device)

    accuracy = 0.0
    
    if teachernet:
        optimizer = optim.Adam(net.parameters(), lr=utils.LR)
        print('Training With Distilation')
        for epoch in range(epochs):
            teachernet.eval()
            net.train()
            total_loss = 0
            total_dist_loss = 0
            total_logits_loss = 0
            correct = 0
            total = 0

            for inputs, labels in dataset.trainloader:
                inputs, labels = inputs.to(utils.device), labels.to(utils.device)
                with torch.no_grad():
                    class_pred_teacher, layer_outputs_teacher = teachernet(inputs)

                optimizer.zero_grad()
                class_pred_student, layer_outputs_student = net(inputs)
                dist_loss = 0
 
                for student_out, teacher_out in zip(layer_outputs_student, layer_outputs_teacher):
                    dist_loss += criterion[0](student_out, teacher_out)

                T = 2.0 
                if epoch > 2:
                    alpha = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
                beta = 1 - alpha
                student_log_probs = F.log_softmax(class_pred_student / T, dim=1)
                teacher_probs = F.softmax(class_pred_teacher / T, dim=1)
                logits_loss = criterion[1](student_log_probs, teacher_probs) * (T ** 2)
                loss = alpha * dist_loss + beta * logits_loss
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item() 
                total_dist_loss += dist_loss
                total_logits_loss += logits_loss


                _, predicted = class_pred_student.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total

            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.5f} | Dist Loss: {total_dist_loss:.5f} | Logits Loss: {total_logits_loss:.5f} | Accuracy: {accuracy:.5f}")
            if (epoch + 1) % 10 == 0:
                print('testing...')
                test(net)
        

    else:
        optimizer = optim.Adam(net.parameters(), lr=utils.LR, weight_decay=1e-4)
        print('Training With Class Labels')
        for epoch in range(epochs):
            net.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in dataset.trainloader:
                inputs, labels = inputs.to(utils.device), labels.to(utils.device)

                optimizer.zero_grad()
                outputs, _ = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total

            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.5f} | Accuracy: {accuracy:.5f}")
            if (epoch + 1) % 10 == 0:
                print('testing...')
                test(net)
    return accuracy

def test(net):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataset.testloader:
            inputs, labels = inputs.to(utils.device), labels.to(utils.device)
            
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
    
    print(f"Test -- Loss: {total_loss:.5f} | Accuracy: {accuracy:.5f}")

    return accuracy