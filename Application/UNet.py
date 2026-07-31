import torch 
import torch.nn as nn 


class DenoisingUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, features = [16, 32, 64, 128]):
        super(DenoisingUNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.features = features

        # Encoder 
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = input_channels
        for feature in features:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(feature, feature, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2),
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature


        # Bottlenck 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.LeakyReLU(0.2),
        )

        # Decoder 
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        in_channels = features[-1] * 2
        for feature in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(feature, feature, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = feature

        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        # Encoder 
        for block, pool in zip(self.encoder_blocks, self.pools):
            x = block(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder
        for up, block, skip in zip(self.upsamples, self.decoder_blocks, reversed(skip_connections)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        mask = torch.sigmoid(self.final_conv(x))
        return mask 



if __name__ == "__main__":
    model = DenoisingUNet()
    dummy = torch.randn(2, 1, 256, 128)
    mask = model(dummy)
    print("Mask shape:", mask.shape)  # Should be (2, 1, 256, 128
    print("Mask min/max:" , mask.min().item(), mask.max().item())