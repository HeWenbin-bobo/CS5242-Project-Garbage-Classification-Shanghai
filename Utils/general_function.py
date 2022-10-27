import torch.nn as nn
import torch
import os
import copy
import gc
import time
from matplotlib import pyplot as plt
import numpy as np
from torchviz import make_dot
from PIL import Image


def weight_initialization(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode = 'fan_in')
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

def train_and_test(train_loader, test_loader, model, model_name, loss_function, optimizer, device, epochs=25, folder_name='./Models'):
    history = []
    best_acc = 0.0
    best_loss = float("inf")
    best_epoch = 0
    es = 0
    
    model_save_path = os.path.join(folder_name, model_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"We create new directory {model_save_path}")
 
    for epoch in range(epochs):
        # if epoch == 0:
        #   print("=============Check memory capacity===============")
        #   # Check memory capacity
        #   !cat /proc/meminfo | grep Cached
          
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
 
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)
 
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

            gc.collect()

 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                test_acc += acc.item() * inputs.size(0)

                gc.collect()
        
 
        avg_train_loss = train_loss/len(train_loader.dataset)
        avg_train_acc = train_acc/len(train_loader.dataset)
 
        avg_test_loss = test_loss/len(test_loader.dataset)
        avg_test_acc = test_acc/len(test_loader.dataset)
 
        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
        
 
        if best_acc < avg_test_acc:
            best_acc = avg_test_acc
            best_epoch = epoch + 1
            es = 0
            torch.save(model, os.path.join(model_save_path, 'garbage_classification_model.pt'))
        else:
            if best_loss > avg_test_loss:
                best_loss = avg_test_loss
                print("Counter {} of 5 for determining early stopping. Test loss keeps decreasing".format(es))
            else:
                es += 1
                print("Counter {} of 5 for determining early stopping".format(es))
                
        if best_loss > avg_test_loss:
            best_loss = avg_test_loss
            
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tTest: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tTime: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_test_loss, avg_test_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for test : {:.4f}% at epoch {:03d}, best loss: {:.4f}\n".format(best_acc*100, best_epoch, best_loss))

        if es > 4:
            print("Early stopping with best_acc: {:.4f}%, best_loss: {:.4f}".format(best_acc*100, best_loss))
            break

    return model, history

def history_save(history, model_name, folder_name='./Models'):
    folder_name = os.path.join(folder_name, model_name)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"We create new directory {folder_name}")

    torch.save(history, os.path.join(folder_name, 'garbage_classification_history.pt'))

def result_figure_save(history, model_name, folder_name='./Figure/Models'):
    folder_name = os.path.join(folder_name, model_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"We create new directory {folder_name}")

    plt.plot(history[:, 0:2])
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 5)
    plt.savefig(os.path.join(folder_name, 'Garbage_classification_loss_curve.png'))
    plt.show()
    
    plt.plot(history[:, 2:4])
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(folder_name, 'Garbage_classification_accuracy_curve.png'))
    plt.show()
    
def weight_plot(model, model_name, folder_name='./Figure/Models'):
    model = copy.deepcopy(model).cpu()
    folder_name = os.path.join(folder_name, model_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"We create new directory {folder_name}")

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # print(module)
            weight = module.weight.detach().numpy().reshape(-1)
            break
    
    
    hist_weights = np.ones_like(weight)/float(len(weight))
    plt.hist(x=weight,
        bins=20,
        color="steelblue",
        edgecolor="black",
        # density=True,
        weights=hist_weights
        )
    plt.savefig(os.path.join(folder_name, 'Weight_Distribution.png'))
    plt.show()
    
def model_structure_plot(model, model_name, crop_size=112, folder_name='MLP'):
    fc_inputs = 3*crop_size**2
    if folder_name=='MLP':
        x = torch.randn(1, fc_inputs).requires_grad_(True)  # 定义一个网络的输入值
    else:
        x = torch.randn(1, 3, crop_size, crop_size).requires_grad_(True)  # 定义一个网络的输入值

    model = copy.deepcopy(model).cpu()
    y = model(x)    # 获取网络的预测值

    MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    MyConvNetVis.format = "png"
    # 指定文件生成的文件夹
    folder_name = os.path.join("./Presentation/modelStructure/", folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"We create new directory {folder_name}")
    MyConvNetVis.directory = folder_name
    MyConvNetVis.filename = model_name
    # 生成文件
    display(Image.open(MyConvNetVis.view()))

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )