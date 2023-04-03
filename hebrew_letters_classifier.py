#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:31:43 2023

@author: menahem_borgesrodrigues_mylinux
"""
#import tkinter as tk
#from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from os import walk
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
###      MODELS ARCHITECTURE TORCH       ######################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class dense_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28*28, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 16)
        self.layer_5 = nn.Linear(16, 7)
      
    def forward(self, x):
        x=F.relu(self.layer_1(x))
        x=F.relu(self.layer_2(x))
        x=F.relu(self.layer_3(x))
        x=F.relu(self.layer_4(x))
        x=F.log_softmax(self.layer_5(x), dim=1)
        return x
    
class cnn_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(1,16,kernel_size=3)
        self.layer_2 = nn.Conv2d(16,32,kernel_size=3)
        self.layer_3 = nn.Linear(32*5*5,32)
        self.layer_4 = nn.Linear(32,7)
    def forward(self,x):
        x=F.relu(F.max_pool2d(self.layer_1(x),2))
        x=F.relu(F.max_pool2d(self.layer_2(x),2))
        x=x.view(-1,32*5*5)
        x=F.relu(self.layer_3(x))
        x=F.log_softmax(self.layer_4(x),dim=1)
        return x
    
class encoder(nn.Module):
    def __init__(self,final_dim):
        super().__init__()
        self.layer_1 = nn.Conv2d(1,16,kernel_size=3)
        self.layer_2 = nn.Conv2d(16,32,kernel_size=3)
        self.layer_3 = nn.Linear(32*5*5,32)
        self.layer_4 = nn.Linear(32,final_dim)
    
    def forward(self,x):
        x=F.relu(F.max_pool2d(self.layer_1(x),2))
        x=F.relu(F.max_pool2d(self.layer_2(x),2))
        x=x.view(-1,32*5*5)
        x=F.relu(self.layer_3(x))
        x=F.relu(self.layer_4(x))
        return x
        
class decoder(nn.Module):
    def __init__(self,final_dim):
        super().__init__()
        self.layer_1 = nn.Linear(final_dim,32)
        self.layer_2 = nn.Linear(32,32*5*5)
        self.layer_un = nn.Unflatten(1,(32,5,5))
        self.layer_3 = nn.ConvTranspose2d(32,16,kernel_size=12)
        self.layer_4 = nn.ConvTranspose2d(16,1,kernel_size=13)
        
    def forward(self,x):
        z=F.relu(self.layer_1(x))
        z=F.relu(self.layer_2(z))
        z= self.layer_un(z)
        z=F.relu(self.layer_3(z))
        z=F.sigmoid(self.layer_4(z))
        return z
        
class auto_encoder(nn.Module):
    def __init__(self,final_dim):
        super().__init__()
        self.encoder=encoder(final_dim)
        self.decoder=decoder(final_dim)
    
    def forward(self,x):
        z=self.encoder(x)
        return self.decoder(z)
    

'''
class vae_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        
class vae_decoder(nn.Module):
    def __init__(self):'''
        
        
##############################################################################
####      IMPLEMENTATION                                      ################
        
class classify_hebrew_letters():
    def __init__(self):
        self.model_type='dense'
        self.epochs=100
        self.model=dense_nn
        self.models={}
        self.loss = F.nll_loss
        self.train_loader=[]
        self.validation_loader=[]
        self.test_loader=[]
        

    def set_model(self,model_type,epochs,data_path_train,data_path_test,batch_size):
        self.model_type=model_type
        self.epochs=epochs
        
        if self.model_type == 'dense':
            self.model=dense_nn()
        elif self.model_type == 'cnn':
            self.model=cnn_nn()
        elif self.model_type == 'auto encoder':
            self.model=auto_encoder(4)
        else:
            raise Exception('sorry, it is not possible implment the model you have requested')
        self.model.to(device)
        self.load('train',data_path_train,batch_size)
        self.load('test',data_path_test,1)
        
    def load(self,data_split,data_dir,batch_s):
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize([28,28]),
                                          transforms.ToTensor()#,transforms.Lambda(lambda x: torch.flatten(x))
                                       ])# TODO: compose transforms here
        dataset = datasets.ImageFolder(data_dir, transform=transform) # TODO: create the ImageFolder
        if data_split=='train':
            train_dataset, valid_dataset= torch.utils.data.random_split(dataset, (len(dataset)-int(0.2*len(dataset)), int(0.2*len(dataset))))
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_s, shuffle=True)
        else:
            self.test_loader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
            
        
    def run(self):
        acc_train,acc_val=[],[]
        optimizer = optim.Adam(self.model.parameters(), lr =0.001)
        for epoch in range(self.epochs):
            correct=0
            total=0
            for X,y in self.train_loader: 
                X, y = X.to(device),y.to(device)
                total+=X.shape[0]
                if self.model_type == 'cnn':
                    output = self.model(X)
                elif self.model_type == 'dense':
                    output = self.model(torch.flatten(X, start_dim=1)) 
                else:
                    y=X.to(device)
                    self.loss=nn.MSELoss()
                    output=self.model(X)  
                optimizer.zero_grad()
                loss = self.loss(output, y)
                loss.backward()
                optimizer.step()
                if self.model_type != 'auto encoder':
                    correct+= (torch.argmax(output, dim=1)==y).sum()
                else:
                    correct+=loss.item()
            acc_train.append(correct/total)
            with torch.no_grad():
                correct_val=0
                total_val=0
                for X,y in self.validation_loader:
                    X, y = X.to(device),y.to(device)
                    total_val+=X.shape[0]
                    if self.model_type == 'cnn':
                        output_val = self.model(X)
                    elif self.model_type == 'dense':
                        output_val = self.model(torch.flatten(X, start_dim=1))
                    else:
                        y=X.to(device)
                        self.loss=nn.MSELoss()
                        output_val = self.model(X)
                    loss = self.loss(output_val, y)
                    if self.model_type != 'auto encoder':
                        correct_val+= (torch.argmax(output, dim=1)==y).sum()
                    else:
                        correct_val+=loss.item()
                acc_val.append(correct_val/total_val)  
        self.models[self.model_type]=[acc_train,acc_val]
    
    def get_history(self):
        return self.models[self.model_type]
        
    def plot_curves(self):
        epochs_array=np.arange(self.epochs)
        for i,j in self.models.items():
            plt.title(f'Classification Accuracy ~{i}')
            plt.plot(epochs_array,torch.tensor(j[0], device = 'cpu'), color='blue', label='train')
            plt.plot(epochs_array,torch.tensor(j[1], device = 'cpu'), color='red', label='validation')
            plt.show()
                
    def evalaute(self):
        with torch.no_grad():
            acc=0
            num=0
            for X,y in self.test_loader:
                X, y = X.to(device),y.to(device)
                if self.model_type == 'cnn':
                    output = self.model(X)
                else:
                    output = self.model(torch.flatten(X, start_dim=1))  
                acc+=(torch.argmax(output, dim=1)==y).sum()
                num+=1
            self.models[self.model_type].append(acc/num)
        
    def see_pred(self):
        c_names=['alef','lamed','mem_sofit','ayin','reish','shin','iud'] 
        rand=np.arange(len(self.test_loader))
        rand=np.random.choice(rand,1)[0]
        this=0
        for X,y in self.test_loader:
            if this == rand:
                X, y = X.to(device),y.to(device)
                with torch.no_grad():
                    if self.model_type == 'cnn':
                        output = self.model(X)
                    else:
                        output = self.model(torch.flatten(X, start_dim=1))  
                    predicted=torch.argmax(output, dim=1)
                    fig, ax = plt.subplots()
                    ax.imshow(torch.tensor(X,device='cpu').reshape(28,28))
                    ax.title.set_text("Predicted: {}, True: {}".format(c_names[predicted],
                                                    c_names[y]))
                    plt.gray()
                    plt.show()
                    return fig
                    break
            else:
                this+=1
                
    def reconstruction(self):
        c_names=['alef','lamed','mem_sofit','ayin','reish','shin','iud'] 
        if self.model_type == 'auto encoder':
            rand=np.arange(len(self.test_loader))
            rand=np.random.choice(rand,1)[0]
            this=0
            for X,y in self.test_loader:
                if this == rand:
                    X, y = X.to(device),X.to(device)
                    with torch.no_grad():
                        output = self.model(X)
                        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                        ax1.imshow(torch.tensor(X,device='cpu').reshape(28,28))
                        ax2.imshow(torch.tensor(output,device='cpu').reshape(28,28))
                        plt.gray()
                        plt.show()
                        return fig
                    break
                else:
                    this+=1
        else:
            print(f'model{self.model_type} has no method reconstruction, try see_pred() instead') 
            
    def img_latent_rec(self):
        if self.model_type == 'auto encoder':
            rand=np.arange(len(self.test_loader))
            rand=np.random.choice(rand,1)[0]
            this=0
            for X,y in self.test_loader:
                if this == rand:
                    X, y = X.to(device),X.to(device)
                    with torch.no_grad():
                        output1 = self.model.encoder(X)
                        print(output1)
                        output2 = self.model(X)
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
                        ax1.imshow(torch.tensor(X,device='cpu').reshape(28,28))
                        ax2.imshow(torch.tensor(output1,device='cpu'))
                        ax3.imshow(torch.tensor(output2,device='cpu').reshape(28,28))
                        plt.gray()
                        plt.show()
                        return fig
                    break
                else:
                    this+=1
        else:
            print(f'model{self.model_type} has no method reconstruction, try see_pred() instead') 
            

hebrew=classify_hebrew_letters()
#hebrew.set_model('dense',10,'data_hebrew/TRAIN','data_hebrew/TEST',64)  
hebrew.set_model('auto encoder',300,'data_hebrew/TRAIN','data_hebrew/TEST',16)   
hebrew.run() 
hebrew.get_history()[0]
hebrew.plot_curves()
hebrew.reconstruction()
hebrew.img_latent_rec()


def main():          
    hebrew=classify_hebrew_letters()
    #hebrew.set_model('dense',100,'data_hebrew/TRAIN','data_hebrew/TEST',64)  
    hebrew.set_model('cnn',100,'data_hebrew/TRAIN','data_hebrew/TEST',64)   
    hebrew.run() 
    hebrew.plot_curves()
    hebrew.evalaute()  
    hebrew.get_history()[2]
    fig=hebrew.see_pred()  
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


# Create the GUI
root = tk.Tk()
root.title("Image Classifier")
root.geometry("800x800")
open_file_button = tk.Button(text="Predict Hebrew character with a Neural Network", command=main)
open_file_button.pack()
root.mainloop()
            




