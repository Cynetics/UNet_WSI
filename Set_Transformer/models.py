from modules import *

class SetTransformer(nn.Module):
    def __init__(self, set_size=32,  num_outputs=1, dim_out=4, num_inds=32,
                           hidden_dim=128, num_heads=4, ln=False):
        
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(set_size, hidden_dim, num_heads, num_inds, ln=ln),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(0.5),
            PMA(hidden_dim, num_heads, num_outputs, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LeakyReLU(0.3), nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, dim_out)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x).squeeze(-1)
        return x
    
class DeepSetTransformer(nn.Module):
    def __init__(self, input_dim=512, set_size=32,  num_outputs=1, dim_out=4, num_inds=32,
                           hidden_dim=128, num_heads=4, ln=False):
        
        super(DeepSetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(set_size, hidden_dim, num_heads, num_inds, ln=ln),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln),
            nn.Dropout(0.5)
        )
        self.dec = nn.Sequential(
            PMA(hidden_dim, num_heads, num_outputs, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            nn.Dropout(0.5)
        )
        
        
        self.deepset_encoder = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(), nn.Dropout(0.3)
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//2), 
            nn.LeakyReLU(0.3), nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, dim_out)
        )

    def forward(self, x):
        transformer_x = self.enc(x)
        transformer_x = self.dec(transformer_x).squeeze(1)
        deepset_x = self.deepset_encoder(x).mean(dim=1)
        #print("transformer x: ", transformer_x.shape)
        #print("deepset x: ", deepset_x.shape)
        x = torch.cat([transformer_x, deepset_x], dim=1)
        x = self.head(x).squeeze(-1)
        
        #print(x.shape)
        #raise ValueError()
        
        return x

class SmallSetTransformer(nn.Module):
    def __init__(self,set_size=32, hidden_dim=64, dim_out=64, num_heads=4):
        super().__init__()
        print("Set Transformer Model Initialized")
        self.enc = nn.Sequential(
            SAB(dim_in=set_size, dim_out=hidden_dim, num_heads=num_heads),
            SAB(dim_in=hidden_dim, dim_out=dim_out, num_heads=num_heads),
        )
        self.dec = nn.Sequential(
            PMA(dim=dim_out, num_heads=num_heads, num_seeds=1),
            nn.Linear(in_features=dim_out, out_features=4),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)

class DeepSet(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(dim_hidden, dim_hidden//2),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(dim_hidden//2, dim_output))

    def forward(self, X):
        X = self.enc(X).mean(dim=1)
        X = self.dec(X)
        return X
    
