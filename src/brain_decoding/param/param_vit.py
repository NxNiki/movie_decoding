from brain_decoding.param.base_param import param_dict

# modify parameters:
param_dict["epochs"] = 10  # 50
param_dict["hidden_size"] = 256
param_dict["num_hidden_layers"] = 6
param_dict["num_attention_heads"] = 8
param_dict["patch_size"] = (1, 5)
