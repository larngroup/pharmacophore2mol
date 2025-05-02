import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet3d, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv3d(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv3d(feature * 2, feature))

        # Final Convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # skip connections will be fed backwards relative to the order they were created in
        skip_connections = skip_connections[::-1]

        # Decoder
        for i in range(0, len(self.decoder), 2): #each "layer" has a conv and a transconv (so 2 real elements), that's why the step=2
            x = self.decoder[i](x) #transconv
            skip_connection = skip_connections[i // 2] #concat skip connection (i//2 cause i is 0,2,4,6...)
            if x.shape != skip_connection.shape:
                # x = TF.resize(x, size=skip_connection.shape[2:]) #TODO: reevaluate options here
                raise ValueError(f"Shape mismatch: x shape {x.shape} and skip_connection shape {skip_connection.shape} do not match. Make sure the input size is divisible by 2^n, where n is the number of pooling layers (if using a kernel size of 2).")
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](concat_skip) #double conv
        
        return self.final_conv(x) #convert to the output channels