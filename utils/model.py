import torch
import torch.nn as nn

class ABLE(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.encoder = base_encoder(name=args.arch, head='mlp', feat_dim=args.low_dim, num_class=args.num_class)

    def forward(self, args, img_w=None, images=None, partial_Y=None, is_eval=False):
        if is_eval:
            output_raw, q = self.encoder(img_w)
            return output_raw
        
        outputs, features = self.encoder(images)
        
        batch_size = args.batch_size
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0) 
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) 

        return outputs, features
