import torch 
import torch.nn as nn 

class DenoisingRTSEWD(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, features=[16, 32, 64, 128]):
        super(DenoisingRTSEWD, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.features = features

        # Encoder 
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for feature in features: 
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, feature, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(feature, feature * 2, kernel_size=3, stride=1, padding=1),
                    nn.GLU(dim=1),
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(features[-1] * 2, features[-1] * 4, kernel_size=3, stride=1, padding=1),
            nn.GLU(dim=1),
        )

        # Decoder 
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        
        in_channels = features[-1] * 2

        for feature in reversed_features:
            # 1. Upsampling layer: restores sequence length, halves channel depth
            self.upconvs.append(
                nn.ConvTranspose1d(in_channels, feature, kernel_size=2, stride=2)
            )
            
            # 2. Decoder block: processes concatenated features (feature from upconv + feature from skip connection)
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(feature * 2, feature * 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(feature * 2, feature * 2, kernel_size=3, stride=1, padding=1),
                    nn.GLU(dim=1),
                )
            )
            in_channels = feature

        # Final output layer
        self.final_output = nn.Sequential(
            nn.Conv1d(features[0], output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = skip_connections[-(i + 1)]

            x = self.upconvs[i](x)
            
            # Handle potential length mismatches from odd input lengths
            if x.shape[-1] != skip_connection.shape[-1]:
                x = nn.functional.interpolate(x, size=skip_connection.shape[-1])
                
            x = torch.cat((x, skip_connection), dim=1)
            x = decoder_block(x)

        # Final output layer
        x = self.final_output(x)
        return x

if __name__ == "__main__":
    model = DenoisingRTSEWD()
    dummy_input = torch.randn(8, 1, 1024)
    output = model(dummy_input)
    print("Output Shape:", output.shape) 