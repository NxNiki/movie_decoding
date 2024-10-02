import torch
import torch.nn as nn

from brain_decoding.param.base_param import device_name


class Ensemble(nn.Module):
    def __init__(self, lfp_model, spike_model, config, branch_model=None):
        super().__init__()
        self.spike_model = spike_model
        self.lfp_model = lfp_model
        self.branch_model = branch_model
        # self.combiner_1 = nn.Sequential(
        #     nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
        #     nn.PReLU(),
        #     nn.Linear(config['hidden_size'] // 2, config['num_labels']),
        # )
        if self.branch_model:
            self.mlp_head_lfp = nn.Sequential(
                nn.LayerNorm(config["hidden_size_lfp"]),
                nn.Linear(config["hidden_size_lfp"], config["num_labels"]),
            )
            self.mlp_head_spike = nn.Sequential(
                nn.LayerNorm(config["hidden_size_spike"]),
                nn.Linear(config["hidden_size_spike"], config["num_labels"]),
            )
        elif self.lfp_model and self.spike_model:
            self.mlp_head_lfp = nn.Sequential(
                nn.LayerNorm(config["hidden_size_lfp"]),
                nn.Linear(config["hidden_size_lfp"], 32),
            )
            self.mlp_head_spike = nn.Sequential(
                nn.LayerNorm(config["hidden_size_spike"]),
                nn.Linear(config["hidden_size_spike"], 32),
            )
            self.mlp_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, config["num_labels"]))
        elif self.lfp_model and not self.spike_model:
            self.mlp_head = nn.Linear(config.model["hidden_size"], config.model["num_labels"])
        elif not self.lfp_model and self.spike_model:
            self.mlp_head = nn.Linear(config.model["hidden_size"], config.model["num_labels"])

        # self.combiner_2 = nn.Sequential(
        #     nn.Linear(config.hidden_size * 2, config.hidden_size),
        #     nn.PReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size // 2),
        #     nn.PReLU(),
        #     nn.Linear(config.hidden_size // 2, config.num_labels),
        # )
        # self.sigmoid = nn.Sigmoid()
        self.device = device_name
        print(f"start ensemble model on device: {device_name}")

    def forward(self, lfp, spike):
        if self.spike_model and not self.lfp_model:
            spike_emb = self.spike_model(spike)
            lfp_emb = None
            combined_emb = spike_emb
            combined_emb = self.mlp_head(combined_emb)
        elif not self.spike_model and self.lfp_model:
            spike_emb = None
            lfp_emb = self.lfp_model(lfp)
            combined_emb = lfp_emb
            combined_emb = self.mlp_head(combined_emb)
        elif self.branch_model:
            spike_emb, lfp_emb = self.branch_model(spike, lfp)
            xs = self.mlp_head_spike(spike_emb)
            xl = self.mlp_head_lfp(lfp_emb)
            combined_emb = xs + xl
        elif self.spike_model and self.lfp_model:
            spike_emb = self.spike_model(spike)
            lfp_emb = self.lfp_model(lfp)
            xs = self.mlp_head_spike(spike_emb)
            xl = self.mlp_head_lfp(lfp_emb)
            x = torch.cat((xs, xl), dim=1)
            combined_emb = self.mlp_head(x)

        # combined_emb = self.sigmoid(combined_emb)
        return spike_emb, lfp_emb, combined_emb
