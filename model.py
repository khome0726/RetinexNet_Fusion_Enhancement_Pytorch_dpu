import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import pytorch_msssim


"""""
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels=16, alpha=5.0):
        
        super(AttentionModule, self).__init__()
        self.alpha = alpha
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        pre_attention_map = torch.mul(x,torch.exp(-x * self.alpha))
        print("pre_attention_map :", pre_attention_map.mean().item())
        attention_map = self.attention_conv(x) * pre_attention_map
        print("attention_map:", attention_map.mean().item())
        return x * attention_map
"""""


"""""
#CBAM
class AttentionModule(nn.Module):
    def __init__(self, dim, in_channels, ratio, kernel_size):
        super(AttentionModule, self).__init__()
        self.avg_pool = getattr(nn, "AdaptiveAvgPool{0}d".format(dim))(1)
        self.max_pool = getattr(nn, "AdaptiveMaxPool{0}d".format(dim))(1)
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.fc1 = conv_fn(in_channels, in_channels // ratio, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = conv_fn(in_channels // ratio, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = conv_fn(2, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # Channel attention module:（Mc(f) = σ(MLP(AvgPool(f)) + MLP(MaxPool(f)))）
        module_input = x
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg + mx)
        x = module_input * x
        # Spatial attention module:Ms (f) = σ( f7×7( AvgPool(f) ; MaxPool(F)] )))
        module_input = x
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg, mx), dim=1)
        x = self.sigmoid(self.conv(x))
        x = module_input * x
        return x
"""""

"""""
class AttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(AttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Spatial attention part
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x

        # Weight the dark regions more
        weighted_input = torch.exp(-module_input) * module_input

        avg = torch.mean(weighted_input, dim=1, keepdim=True)
        mx, _ = torch.max(weighted_input, dim=1, keepdim=True)
        x = torch.cat((avg, mx), dim=1)
        x = self.sigmoid(self.conv(x))
        return module_input * x
"""""
"""""
class AttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(AttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2  # Use integer division for dynamic padding based on kernel size
        
        # Spatial attention part
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.beta = 0.2  # A threshold to define what is considered as dark

    def forward(self, x):
        module_input = x

        # Calculate a mask where the dark regions are 1 and the rest are close to 0
        dark_mask = torch.sigmoid((self.beta - module_input) * 10)  # Scale factor to sharpen the mask

        # Apply the mask to the input by adding emphasis to the dark regions
        weighted_input = module_input * dark_mask

        # Calculate the average and max values across the weighted input's channel dimension
        avg_weighted = torch.mean(weighted_input, dim=1, keepdim=True)
        max_weighted, _ = torch.max(weighted_input, dim=1, keepdim=True)

        # Combine the average and max weighted features
        combined = torch.cat((avg_weighted, max_weighted), dim=1)

        # Apply the spatial attention mechanism
        attention_map = self.sigmoid(self.conv(combined))

        # Multiply the original input with the attention map to enhance dark regions
        return module_input * attention_map
"""""

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2  

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid = nn.Sigmoid()

        self.beta = 0.1 
        self.alpha = 0.8  

    def forward(self, x):
        module_input = x

        dark_enhanced = torch.tanh((self.beta - module_input) * 10)
        #light_suppressed = torch.tanh((module_input - self.beta) * 5) * self.alpha

        #weighted_mask = dark_enhanced * light_suppressed
        weighted_mask = dark_enhanced

        avg_weighted = torch.mean(weighted_mask, dim=1, keepdim=True)
        max_weighted, _ = torch.max(weighted_mask, dim=1, keepdim=True)

        combined = torch.cat((avg_weighted, max_weighted), dim=1)
        attention_map = self.sigmoid(self.conv(combined))

        return module_input * attention_map



"""""
class DecomNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(2, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 2, kernel_size,
                                    padding=1, padding_mode='replicate')
 
    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:1, :, :])
        L        = torch.sigmoid(outs[:, 1:2, :, :])

        return R, L 
"""""
class DecomNet(nn.Module):
    def __init__(self, channel=8, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(2, channel, kernel_size * 3, padding=4, padding_mode='zeros')
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='zeros'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='zeros'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='zeros'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='zeros'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='zeros'),
                                        nn.ReLU())
        self.net1_recon = nn.Conv2d(channel, 2, kernel_size, padding=1, padding_mode='zeros')

        #self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(in_channels=1)

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        #feats0 = self.channel_attention(feats0)
        feats0 = self.spatial_attention(feats0)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:1, :, :])
        L = torch.sigmoid(outs[:, 1:2, :, :])

        return R, L
"""""
class RelightNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(2, channel, kernel_size,
                                      padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)


    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)

        input_img = self.attention(input_img)

        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output
"""""

class RelightNet(nn.Module):
    def __init__(self, channel=8, kernel_size=3):
        super(RelightNet, self).__init__()
        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(2, channel, kernel_size, padding=1, padding_mode='zeros')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='zeros')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='zeros')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='zeros')
        self.net2_deconv1_1 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='zeros')
        self.net2_deconv1_2 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='zeros')
        self.net2_deconv1_3 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='zeros')
        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1, padding=0, padding_mode='zeros')   
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=1, padding_mode='zeros')  

        #self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(in_channels=1)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)

        out0 = self.net2_conv0_1(input_img)
        out0 = self.spatial_attention(out0)

        out1 = self.relu(self.net2_conv1_1(out0))
        out1 = self.spatial_attention(out1)

        out2 = self.relu(self.net2_conv1_2(out1))
        out2 = self.spatial_attention(out2)

        out3 = self.relu(self.net2_conv1_3(out2))
        out3_up = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))

        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1 = self.spatial_attention(deconv1)

        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv1_up = self.spatial_attention(deconv1_up)

        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2 = self.spatial_attention(deconv2)

        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv2_up = self.spatial_attention(deconv2_up)

        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))
        deconv3 = self.spatial_attention(deconv3)

        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv1_rs = self.spatial_attention(deconv1_rs)

        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = self.spatial_attention(deconv2_rs)

        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)

        output = self.net2_output(feats_fus)

        return output
    





class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        #self.channel_attention = ChannelAttention(in_channels=1) 
        self.spatial_attention = SpatialAttention(in_channels=1)

    def forward(self, input_low, input_high):
        # Convert numpy to tensor and send to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_low = torch.FloatTensor(input_low).to(device)
        input_high = torch.FloatTensor(input_high).to(device)

        # Apply Channel and Spatial Attention
        #input_low = self.channel_attention(input_low)
        input_low = self.spatial_attention(input_low)
        #input_high = self.channel_attention(input_high)
        #input_high = self.spatial_attention(input_high)


        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low)

        # Other variables
        # I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)
        # I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        # I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)


        # Compute losses
        self.recon_loss_low  = F.l1_loss(R_low * I_low,  input_low)
        self.recon_loss_high = F.l1_loss(R_high * I_high, input_high)
        self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low, input_low)
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high, input_high)
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())
        self.relight_loss = F.l1_loss(R_low * I_delta, input_high)

        self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)


        # dark loss
        beta = 0.1
        dark_parts_low = torch.where(input_low < beta, input_low, torch.zeros_like(input_low))
        dark_parts_high = torch.where(input_high < beta, input_high, torch.zeros_like(input_high))
        
        dark_loss = F.l1_loss(dark_parts_low, dark_parts_high)



        self.loss_Decom = self.recon_loss_low + \
                        self.recon_loss_high + \
                        0.001 * self.recon_loss_mutal_low + \
                        0.001 * self.recon_loss_mutal_high + \
                        0.1 * self.Ismooth_loss_low + \
                        0.1 * self.Ismooth_loss_high + \
                        0.01 * self.equal_R_loss 
        print("self.loss_Decom:",self.loss_Decom)



        # ssim + relight (loss)
        num_channels = input_high.size(1)  
        ssim_module = pytorch_msssim.SSIM(data_range=1.0, channel=num_channels)
        ssim_loss = 1 - ssim_module(input_high, R_low * I_delta)

        self.loss_Relight = 0.5 * (self.relight_loss + \
                            3 * self.Ismooth_loss_delta) + \
                            0.5 * (ssim_loss) + \
                            0 * dark_loss 
                            

                            
        print("self.loss_Relight:",self.loss_Relight)



        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low.detach().cpu()
        self.output_I_delta = I_delta.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta.detach().cpu()







    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        
        # input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input    = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image= np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                       (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir):
        load_dir   = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts)>0:
                load_ckpt  = load_ckpts[-1]
                global_step= int(load_ckpt[:-4])
                ckpt_dict  = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase= train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 1, patch_size, patch_size), dtype="float32")
                batch_input_high= np.zeros((batch_size, 1, patch_size, patch_size), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    train_high_img= Image.open(train_high_data_names[image_id])
                    train_high_img= np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w       = train_low_img.shape
                    
                    _ = 1
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = np.expand_dims(train_low_img,axis=-1)
                    train_high_img = np.expand_dims(train_high_img,axis=-1)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    
                    
                    train_high_img= train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img= np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img= np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img= np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))

                    train_high_img= np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, : ] = train_low_img
                    batch_input_high[patch_id, :, :, :]= train_high_img
                    self.input_low = batch_input_low
                    self.input_high= batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)


                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)


    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint
        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
             print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path  = test_low_data_names[idx]
            test_img_name  = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            test_low_img   = Image.open(test_img_path).convert('L')
            test_low_img   = np.array(test_low_img, dtype="float32")/255.0
            print(test_low_img.shape)
            test_low_img   = np.tile(test_low_img, (1, 1, 1))
            
            input_low_test = np.expand_dims(test_low_img, axis=0)
            print(input_low_test.shape)

            input_low_tensor = torch.from_numpy(input_low_test).float()


            self.forward(input_low_tensor, input_low_tensor)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)
            result_4 = self.output_S

            print(result_4.shape)

            input = np.expand_dims(input, axis=0) 
            input = np.expand_dims(input, axis=0)  

            print(input.dtype)
            print(result_4.dtype)
            
            
            mean_brightness = input_low_tensor.mean()

            if 0.2 < mean_brightness < 0.5: 
                alpha = 0.5 * (1 + mean_brightness)  # Smaller weight for moderately bright images
            elif mean_brightness > 0.5: 
                alpha = 1  # Larger weight for bright images
            else:
                alpha = 0.5  # Default weight for dark images



            result_4 = result_4.cpu().numpy()
            cat_image = alpha * input + (1 - alpha) * result_4
            cat_image = np.clip(cat_image, 0, 1)
            
            im = Image.fromarray(np.uint8(cat_image[0, 0] * 255.0))
            filepath = res_dir + '/' + test_img_name
            im.save(filepath[:-4] + '.png')

    """""
    def predict(self, test_low_data_names, res_dir, ckpt_dir):
        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)

            test_low_img = Image.open(test_img_path).convert('L')
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.tile(test_low_img, (1, 1, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            # Convert the image to a PyTorch tensor and apply the dark mask operation
            input_low_tensor = torch.from_numpy(input_low_test).float()
            dark_enhanced = torch.tanh((0.1 - input_low_tensor) * 5)

            # Use dark_enhanced as input to the forward pass
            self.forward(dark_enhanced, dark_enhanced)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S

            # Extract results and process
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)

            # Combine the input and result for final output
            alpha = 0.5
            result_4 = result_4.cpu().numpy()
            cat_image = alpha * input + (1 - alpha) * result_4
            cat_image = np.clip(cat_image, 0, 1)

            # Convert the result to an image and save it
            im = Image.fromarray(np.uint8(cat_image[0, 0] * 255.0))
            filepath = res_dir + '/' + test_img_name
            im.save(filepath[:-4] + '.jpg')
        """""