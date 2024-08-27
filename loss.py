

#EfficientNet
###########################################################################################################################################################


# import torch
# from torch import nn
# from torchvision.models import efficientnet_b0

# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         # Load pre-trained EfficientNet model
#         efficientnet = efficientnet_b0(pretrained=True)
        
#         # Extract features from EfficientNet
#         # The `features` part of EfficientNet includes all convolutional layers
#         # EfficientNet doesn't have a single `features` attribute like DenseNet or VGG,
#         # so we'll use the `features` from the `features` layer directly.
#         self.loss_network = efficientnet.features.eval()
        
#         # Freeze the parameters of the feature extractor
#         for param in self.loss_network.parameters():
#             param.requires_grad = False
            
#         # Define the loss functions
#         self.mse_loss = nn.MSELoss()
#         self.tv_loss = TVLoss()

#     def forward(self, out_labels, out_images, target_images):
#         # Adversarial Loss
#         adversarial_loss = torch.mean(1 - out_labels)
        
#         # Perception Loss (using features from EfficientNet)
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
#         # Image Loss
#         image_loss = self.mse_loss(out_images, target_images)
        
#         # TV Loss (Total Variation Loss)
#         tv_loss = self.tv_loss(out_images)
        
#         # Combine all losses
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


# class TVLoss(nn.Module):
#     def __init__(self, tv_loss_weight=1):
#         super(TVLoss, self).__init__()
#         self.tv_loss_weight = tv_loss_weight

#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self.tensor_size(x[:, :, 1:, :])
#         count_w = self.tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

#     @staticmethod
#     def tensor_size(t):
#         return t.size()[1] * t.size()[2] * t.size()[3]


# if __name__ == "__main__":
#     # Initialize the GeneratorLoss
#     g_loss = GeneratorLoss()
    
#     # Print the GeneratorLoss instance
#     print(g_loss)
    
#     # Example input tensors
#     batch_size = 1
#     channels = 3
#     height = 224
#     width = 224
    
#     out_labels = torch.rand(batch_size, 1)  # Random adversarial labels
#     out_images = torch.rand(batch_size, channels, height, width)  # Generated images
#     target_images = torch.rand(batch_size, channels, height, width)  # Target high-res images
    
#     # Calculate the loss
#     loss = g_loss(out_labels, out_images, target_images)
    
#     # Print the calculated loss
#     print("Calculated Loss:", loss.item())



#DenseNet
###########################################################################################################################################################



# import torch
# from torch import nn
# from torchvision.models import densenet121

# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         # Load pre-trained DenseNet model
#         densenet = densenet121(pretrained=True)
        
#         # Use all the convolutional layers (features) from DenseNet
#         self.loss_network = densenet.features.eval()
        
#         # Freeze the parameters of the feature extractor
#         for param in self.loss_network.parameters():
#             param.requires_grad = False
            
#         # Define the loss functions
#         self.mse_loss = nn.MSELoss()
#         self.tv_loss = TVLoss()

#     def forward(self, out_labels, out_images, target_images):
#         # Adversarial Loss
#         adversarial_loss = torch.mean(1 - out_labels)
        
#         # Perception Loss (using features from DenseNet)
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
#         # Image Loss
#         image_loss = self.mse_loss(out_images, target_images)
        
#         # TV Loss (Total Variation Loss)
#         tv_loss = self.tv_loss(out_images)
        
#         # Combine all losses
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


# class TVLoss(nn.Module):
#     def __init__(self, tv_loss_weight=1):
#         super(TVLoss, self).__init__()
#         self.tv_loss_weight = tv_loss_weight

#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self.tensor_size(x[:, :, 1:, :])
#         count_w = self.tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

#     @staticmethod
#     def tensor_size(t):
#         return t.size()[1] * t.size()[2] * t.size()[3]


# if __name__ == "__main__":
#     # Initialize the GeneratorLoss
#     g_loss = GeneratorLoss()
    
#     # Print the GeneratorLoss instance
#     print(g_loss)
    
#     # Example input tensors
#     batch_size = 1
#     channels = 3
#     height = 224
#     width = 224
    
#     out_labels = torch.rand(batch_size, 1)  # Random adversarial labels
#     out_images = torch.rand(batch_size, channels, height, width)  # Generated images
#     target_images = torch.rand(batch_size, channels, height, width)  # Target high-res images
    
#     # Calculate the loss
#     loss = g_loss(out_labels, out_images, target_images)
    
#     # Print the calculated loss
#     print("Calculated Loss:", loss.item())




#VGG16
###########################################################################################################################################################


# import torch
# from torch import nn
# from torchvision.models.vgg import vgg16


# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         vgg = vgg16(pretrained=True)
#         loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
#         for param in loss_network.parameters():
#             param.requires_grad = False
#         self.loss_network = loss_network
#         self.mse_loss = nn.MSELoss()
#         self.tv_loss = TVLoss()

#     def forward(self, out_labels, out_images, target_images):
#         # Adversarial Loss
#         adversarial_loss = torch.mean(1 - out_labels)
#         # Perception Loss
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
#         # Image Loss
#         image_loss = self.mse_loss(out_images, target_images)
#         # TV Loss
#         tv_loss = self.tv_loss(out_images)
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


# class TVLoss(nn.Module):
#     def __init__(self, tv_loss_weight=1):
#         super(TVLoss, self).__init__()
#         self.tv_loss_weight = tv_loss_weight

#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self.tensor_size(x[:, :, 1:, :])
#         count_w = self.tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

#     @staticmethod
#     def tensor_size(t):
#         return t.size()[1] * t.size()[2] * t.size()[3]


# if __name__ == "__main__":
#     g_loss = GeneratorLoss()
#     print(g_loss)


#MobileNet2
###########################################################################################################################################################

# import torch
# from torch import nn
# from torchvision.models import mobilenet_v2

# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         mobilenet = mobilenet_v2(pretrained=True)
#         # We use the features (excluding the classifier) of MobileNetV2
#         loss_network = mobilenet.features.eval()
#         for param in loss_network.parameters():
#             param.requires_grad = False
#         self.loss_network = loss_network
#         self.mse_loss = nn.MSELoss()
#         self.tv_loss = TVLoss()

#     def forward(self, out_labels, out_images, target_images):
#         # Adversarial Loss
#         adversarial_loss = torch.mean(1 - out_labels)
#         # Perception Loss
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
#         # Image Loss
#         image_loss = self.mse_loss(out_images, target_images)
#         # TV Loss
#         tv_loss = self.tv_loss(out_images)
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


# class TVLoss(nn.Module):
#     def __init__(self, tv_loss_weight=1):
#         super(TVLoss, self).__init__()
#         self.tv_loss_weight = tv_loss_weight

#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self.tensor_size(x[:, :, 1:, :])
#         count_w = self.tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

#     @staticmethod
#     def tensor_size(t):
#         return t.size()[1] * t.size()[2] * t.size()[3]


# if __name__ == "__main__":
#     g_loss = GeneratorLoss()
#     print(g_loss)



# ResNet50
##########################################################################################################################################################

# import torch
# from torch import nn
# from torchvision.models import resnet50

# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         resnet = resnet50(pretrained=True)
#         # We use the layers up to the 4th block of ResNet50 for feature extraction
#         loss_network = nn.Sequential(*list(resnet.children())[:8]).eval()
#         for param in loss_network.parameters():
#             param.requires_grad = False
#         self.loss_network = loss_network
#         self.mse_loss = nn.MSELoss()
#         self.tv_loss = TVLoss()

#     def forward(self, out_labels, out_images, target_images):
#         # Adversarial Loss
#         adversarial_loss = torch.mean(1 - out_labels)
#         # Perception Loss
#         perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
#         # Image Loss
#         image_loss = self.mse_loss(out_images, target_images)
#         # TV Loss
#         tv_loss = self.tv_loss(out_images)
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


# class TVLoss(nn.Module):
#     def __init__(self, tv_loss_weight=1):
#         super(TVLoss, self).__init__()
#         self.tv_loss_weight = tv_loss_weight

#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self.tensor_size(x[:, :, 1:, :])
#         count_w = self.tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

#     @staticmethod
#     def tensor_size(t):
#         return t.size()[1] * t.size()[2] * t.size()[3]


# if __name__ == "__main__":
#     g_loss = GeneratorLoss()
#     print(g_loss)


#VGG19
###########################################################################################################################################################

import torch
from torch import nn
from torchvision.models import vgg19

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        # We use the first 36 layers of VGG19 (up to 'conv4_4')
        loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
