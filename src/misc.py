from os import stat
import matplotlib.pyplot as plt
import numpy as np
import torch

class Utils:
    
    """ Visualization utilities """

    @staticmethod
    def imshow(img, text=None, label1=None, label2=None, should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic', fontweight='bold',
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})                
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    @staticmethod
    def show_plot(iteration, loss):
        plt.plot(iteration, loss)
        plt.show()

    """saving and loading checkpoint mechanisms"""
    @staticmethod
    def save_checkpoint(save_path, model, optimizer, val_loss):
        if save_path==None:
            return
        save_path = save_path 
        state_dict = {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss}
        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(model, optimizer):
        save_path = f'siameseNet-batchnorm50.pt'
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        val_loss = state_dict['val_loss']
        print(f'Model loaded from <== {save_path}')
        
        return val_loss