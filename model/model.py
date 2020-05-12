import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Generator(BaseModel):

    def __init__(self):
        super(HDR, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 16, 3, padding=(1,1,1)) #[b, 16, 3, 180, 320]
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 16, 3, padding=(1,1,1)) #[b, 16, 3, 180, 320]
        self.bn2 = nn.BatchNorm3d(16)
        self.mpool1 = nn.MaxPool3d((1,2,2),return_indices=True) #[b, 16, 3, 90, 160]
        
        self.conv3 = nn.Conv3d(16, 32, 3, padding=(1,1,1)) #[b, 32, 3, 90, 160]
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=(1,1,1)) #[b, 32, 3, 90, 160]
        self.bn4 = nn.BatchNorm3d(32)
        self.mpool2 = nn.MaxPool3d((1,2,2),return_indices=True) #[b, 32, 3, 45, 80]
        
        self.conv5 = nn.Conv3d(32, 64, 3, padding=(0,1,1)) #[b, 64, 1, 45, 80]
        self.bn5 = nn.BatchNorm3d(64)
        self.conv6 = nn.Conv3d(64, 64, 3, padding=(1,1,1)) #[b, 64, 1, 45, 80]
        self.bn6 = nn.BatchNorm3d(64) 
        self.conv7 = nn.Conv3d(64, 64, 3, padding=(1,1,1)) #[b, 64, 1, 45, 80]
        self.bn7 = nn.BatchNorm3d(64)
        self.mpool3 = nn.MaxPool3d((1,2,2),return_indices=True) #[b, 64, 1, 22, 40]
        
        self.conv8 = nn.Conv3d(64, 128, 3, padding=(1,1,1)) #[b, 128, 1, 22, 40]
        self.bn8 = nn.BatchNorm3d(128)
        self.conv9 = nn.Conv3d(128, 128, (3,3,3), padding=(1,1,1)) #[b, 128, 1, 22, 40]
        self.bn9 = nn.BatchNorm3d(128)
        self.conv10 = nn.Conv3d(128, 128, 3, padding=(1,1,1)) #[b, 128, 1, 22, 40]
        self.bn10 = nn.BatchNorm3d(128)
        self.mpool4 = nn.MaxPool3d((1,2,2),return_indices=True) #[b, 128, 1, 11, 20]
        
        self.conv11 = nn.Conv3d(128, 256, 3, padding=(1,1,1)) #[b, 256, 1, 11, 20]
        self.bn11 = nn.BatchNorm3d(256)
        self.conv12 = nn.Conv3d(256, 256, 3, padding=(1,1,1)) #[b, 256, 1, 11, 20]
        self.bn12 = nn.BatchNorm3d(256)
        self.conv13 = nn.Conv3d(256, 256, 3, padding=(1,1,1)) #[b, 256, 1, 11, 20]
        self.bn13 = nn.BatchNorm3d(256)
        self.mpool5 = nn.MaxPool3d((1,2,2),return_indices=True) #[b, 256, 1, 5, 10]
        
        #latent-transform
        self.conv14 = nn.Conv3d(256, 256, 3, padding=(1,1,1)) #[b, 256, 1, 5, 10]
        self.bn14 = nn.BatchNorm3d(256)
        
        
        #decoder
        self.unpool5 = nn.MaxUnpool3d((1,2,2)) #[b, 256, 1, 10, 20]
        self.deconv13 = nn.ConvTranspose3d(256, 256, 3, padding=(1,1,1)) #[b, 256, 1, 11, 20]
        self.bn15 = nn.BatchNorm3d(256)
        # skip cat
        self.one_conv13 = nn.Conv3d(512, 256, 1) #[b, 256, 1, 11, 20]
        self.deconv12 = nn.ConvTranspose3d(256, 256, 3, padding=(1,1,1)) #[b, 256, 1, 11, 20]
        self.bn16 = nn.BatchNorm3d(256)
        self.deconv11 = nn.ConvTranspose3d(256, 128, 3, padding=(1,1,1)) #[b, 128, 1, 11, 20]
        self.bn17 = nn.BatchNorm3d(128)
        
        self.unpool4 = nn.MaxUnpool3d((1,2,2)) #[b, 128, 1, 22, 40]
        self.deconv10 = nn.ConvTranspose3d(128, 128, 3, padding=(1,1,1)) #[b, 128, 1, 22, 40]
        self.bn18 = nn.BatchNorm3d(128)
        # skip cat
        self.one_conv10 = nn.Conv3d(256, 128, 1) #[b, 128, 1, 22, 40]
        self.deconv9 = nn.ConvTranspose3d(128, 128, (3,3,3), padding=(1,1,1))
        self.bn19 = nn.BatchNorm3d(128)
        self.deconv8 = nn.ConvTranspose3d(128, 64, 3, padding=(1,1,1)) #[b, 64, 1, 22, 40]
        self.bn20 = nn.BatchNorm3d(64)
        
        self.unpool3 = nn.MaxUnpool3d((1,2,2)) #[b, 64, 3, 45, 80]
        self.deconv7 = nn.ConvTranspose3d(64, 64, 3, padding=(1,1,1)) #[b, 64, 1, 45, 80]
        self.bn21 = nn.BatchNorm3d(64)
        # skip cat
        self.one_conv7 = nn.Conv3d(128, 64, 1)

        self.deconv6 = nn.ConvTranspose3d(64, 64, 3, padding=(1,1,1)) #[b, 64, 1, 45, 80]
        self.bn22 = nn.BatchNorm3d(64)
        self.deconv5 = nn.ConvTranspose3d(64, 32, 3, padding=(0,1,1)) #[b, 32, 3, 45, 80]
        self.bn23 = nn.BatchNorm3d(32)
        
        self.unpool2 = nn.MaxUnpool3d((1,2,2)) #[b, 32, 1, 90, 160]
        self.deconv4 = nn.ConvTranspose3d(32, 32, 3, padding=(1,1,1)) #[b, 32, 3, 90, 160]
        self.bn24 = nn.BatchNorm3d(32)
        self.one_conv4 = nn.Conv3d(64, 32, 1)
        self.deconv3 = nn.ConvTranspose3d(32, 16, 3, padding=(1,1,1)) #[b, 16, 3, 90, 160]
        self.bn25 = nn.BatchNorm3d(16)
        
        self.unpool1 = nn.MaxUnpool3d((1,2,2)) #[b, 16, 3, 180, 320]
        self.deconv2 = nn.ConvTranspose3d(16, 16, 3, padding=(1,1,1)) #[b, 16, 1, 180, 320]
        self.bn26 = nn.BatchNorm3d(16)
        self.one_conv2 = nn.Conv3d(32, 16, 1) #[b, 16, 3, 180, 320]
        
        self.deconv1 = nn.ConvTranspose3d(16, 3, 3, padding=(1,1,1)) #[b, 3, 3, 180, 320]
        self.bn27 = nn.BatchNorm3d(3)
        self.one_convop = nn.Conv3d(6, 3, (3,1,1))
#         self.deconv0 = nn.ConvTranspose3d(3, 3, 3, padding=(2,1,1)) #[b, 3, 1, 180, 320]

        
        
#         self.one_convop2 = nn.Conv3d(3, 3, (3,1,1), padding=(1,0,0))
        
        self._initialize_weights()

    def forward(self, frames):
        X = torch.stack([frames[0],frames[1], frames[2]], dim=2)
        b,n,f,h,w = X.shape

        op_conv1 = F.relu(self.conv1(X))
        op_conv1 = self.bn1(op_conv1)
        op_conv2 = F.relu(self.conv2(op_conv1))
        op_conv2 = self.bn2(op_conv2)
        op_pool1,pool1_ind = self.mpool1(op_conv2)
        
        op_conv3 = F.relu(self.conv3(op_pool1))
        op_conv3 = self.bn3(op_conv3)
        op_conv4 = F.relu(self.conv4(op_conv3))
        op_conv4 = self.bn4(op_conv4)
        op_pool2,pool2_ind = self.mpool2(op_conv4)
        
        op_conv5 = F.relu(self.conv5(op_pool2))
        op_conv5 = self.bn5(op_conv5)
        op_conv6 = F.relu(self.conv6(op_conv5))
        op_conv6 = self.bn6(op_conv6)
        op_conv7 = F.relu(self.conv7(op_conv6))
        op_conv7 = self.bn7(op_conv7)
        op_pool3,pool3_ind = self.mpool3(op_conv7)
        
        op_conv8 = F.relu(self.conv8(op_pool3))
        op_conv8 = self.bn8(op_conv8)
        op_conv9 = F.relu(self.conv9(op_conv8))
        op_conv9 = self.bn9(op_conv9)
        op_conv10 = F.relu(self.conv10(op_conv9))
        op_conv10 = self.bn10(op_conv10)
        op_pool4,pool4_ind = self.mpool4(op_conv10)
        
        op_conv11 = F.relu(self.conv11(op_pool4))
        op_conv11 = self.bn11(op_conv11)
        op_conv12 = F.relu(self.conv12(op_conv11))
        op_conv12 = self.bn12(op_conv12)
        op_conv13 = F.relu(self.conv13(op_conv12))
        op_conv13 = self.bn13(op_conv13)
        op_pool5,pool5_ind = self.mpool5(op_conv13)
        
        #latent
        op_conv14 = F.relu(self.conv14(op_pool5))
        op_conv14 = self.bn14(op_conv14)
        
        #decoder
        op_unpool5 = self.unpool5(op_conv14, pool5_ind, output_size = op_conv13.size())
        op_deconv13 = F.relu(self.deconv13(op_unpool5))
        op_deconv13 = self.bn15(op_deconv13)
        skip_cat13 = torch.cat([op_conv12,op_deconv13], dim=1) ####
        op_skipconv13 = F.relu(self.one_conv13(skip_cat13))
        op_deconv12 = F.relu(self.deconv12(op_skipconv13))
        op_deconv12 = self.bn16(op_deconv12)
        op_deconv11 = F.relu(self.deconv11(op_deconv12))
        op_deconv11 = self.bn17(op_deconv11)
        
        op_unpool4 = self.unpool4(op_deconv11, pool4_ind, output_size = op_conv10.size())
        op_deconv10 = F.relu(self.deconv10(op_unpool4))
        op_deconv10 = self.bn18(op_deconv10)
        skip_cat10 = torch.cat([op_conv9,op_deconv10], dim=1)
        op_skipconv10 = F.relu(self.one_conv10(skip_cat10))

        op_deconv9 = F.relu(self.deconv9(op_skipconv10))
        op_deconv9 = self.bn19(op_deconv9)
        op_deconv8 = F.relu(self.deconv8(op_deconv9))
        op_deconv8 = self.bn20(op_deconv8)
        
        op_unpool3 = self.unpool3(op_deconv8, pool3_ind, output_size = op_conv7.size())
        op_deconv7 = F.relu(self.deconv7(op_unpool3))
        op_deconv7 = self.bn21(op_deconv7)
        skip_cat7 = torch.cat([op_conv6,op_deconv7], dim=1)
        op_skipconv7 = F.relu(self.one_conv7(skip_cat7))

        op_deconv6 = F.relu(self.deconv6(op_skipconv7))
        op_deconv6 = self.bn22(op_deconv6)
        op_deconv5 = F.relu(self.deconv5(op_deconv6))
        op_deconv5 = self.bn23(op_deconv5)
        
        op_unpool2 = self.unpool2(op_deconv5, pool2_ind,output_size = op_conv4.size())

        op_deconv4 = F.relu(self.deconv4(op_unpool2))
        op_deconv4 = self.bn24(op_deconv4)
        skip_cat4 = torch.cat([op_conv3,op_deconv4], dim=1)
        op_skipconv4 = F.relu(self.one_conv4(skip_cat4))

        op_deconv3 = F.relu(self.deconv3(op_skipconv4))
        op_deconv3 = self.bn25(op_deconv3)
        
        op_unpool1 = self.unpool1(op_deconv3, pool1_ind, output_size = op_conv2.size())
        op_deconv2 = F.relu(self.deconv2(op_unpool1))
        op_deconv2 = self.bn26(op_deconv2)
        skip_cat2 = torch.cat([op_conv1,op_deconv2], dim=1)
        op_skipconv2 = F.relu(self.one_conv2(skip_cat2))
        op_deconv1 = torch.tanh(self.deconv1(op_skipconv2)) 
        op_deconv1 = self.bn27(op_deconv1)
        skip_cat0 = torch.cat([X, op_deconv1], dim=1)
        op_skipconv0 = torch.tanh(self.one_convop(skip_cat0))

#         op_skipconv0 = torch.tanh(op_skipconv0)
        
        return op_skipconv0
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                
class Discriminator(BaseModel):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        self._initialize_weights()

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
