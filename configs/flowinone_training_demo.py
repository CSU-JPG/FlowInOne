import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    adapter_in_embed=2048,
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    clip_dim=768,                                              
    num_clip_token=77,
    gradient_checking=True,
    cfg_indicator=0.1,
    VAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,  
        hidden_token_length = 256,  
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
    stage_configs = [
            Args(
                block_type = "TransformerBlock", 
                dim = 1024,  
                hidden_dim = 2048,  
                num_attention_heads = 16,  
                num_blocks = 65,  
                max_height = 16,
                max_width = 16,
                image_input_ratio = 1,
                input_feature_ratio = 2,
                final_kernel_size = 3,
                dropout_prob = 0,
                use_cross_attention = True,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 512,  
                hidden_dim = 1024,  
                kernel_size = 7, 
                num_blocks = 33,  
                max_height = 32,
                max_width = 32,
                image_input_ratio = 1,
                input_feature_ratio = 1,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
    ],
)


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234                                          # random seed
    config.z_shape = (4, 32, 32)                                # image latent size

    config.autoencoder = d(
        pretrained_path='/path/to/stable-diffusion/autoencoder_kl.pth', # path of pretrained VAE CKPT from LDM
        scale_factor=0.23010
    )

    config.pretrained_path = "/path/to/t2i_256px_clip_dimr.pth"

    config.train = d(
        n_steps=600000,                                        # total training iterations
        batch_size=256,                                           # overall batch size across ALL gpus, where batch_size_per_gpu == batch_size / number_of_gpus
        mode='cond',
        log_interval=10,
        eval_interval=1000,                                       # iteration interval for visual testing on the specified prompt
        save_interval=60000,                                      # iteration interval for saving checkpoints and testing FID
        n_samples_eval=15,                                       
    )

    config.optimizer = d(
        name='adamw',   
        lr=0.00001,                                             # learning rate
        weight_decay=0.03,
        betas=(0.9, 0.9),
        adamw_impl='AdamW',
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000                                       # warmup steps
    )

    global model
    config.nnet = d(
        name='dimr',
        model_args=model,
    )
    config.loss_coeffs = [1/4, 1]                          # weight on loss, only needed for DiMR. Here, loss = 1/4 * loss_block1 + 1/2 * loss_block2 + 1 * loss_block3
    
    config.dataset = d(
        name='online_features',  
        task='visual_instruction',
        resolution=256,
        
        train_tar_pattern='/path/to/pairs-{000000..000009}.tar',
        test_tar_pattern='/path/to/pairs-000010.tar',
        vis_image_root='/path/to/run_vis/',
        
        shuffle_buffer=300,
        resampled=True,
        split_data_by_node=True,
        estimated_samples_per_shard=600,
        
        cfg=False
    )

    config.sample = d(
        sample_steps=50,                                       
        n_samples=30000,                                        
        mini_batch_size=16,                                     # batch size for testing (the number of images generated per GPU)
        cfg=False,
        scale=7,                                                # cfg scale
        path='/path/to/sample/samplesave'
    )

    return config