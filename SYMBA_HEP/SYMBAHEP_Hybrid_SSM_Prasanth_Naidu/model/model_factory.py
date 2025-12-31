from .mamba_encdec import MambaEncDec

def construct_model(config):

    mamba_config = {
        "enc_n_layer": 4,
        "d_model": 512,
        "dec_n_layer": 6,
        "rms_norm": True,
        "fused_add_norm": True,
        "use_fast_path": False,
        # "learning_rate": config.learning_rate,
        # "warmup_steps": config.warmup_steps,
        # "weight_decay": config.weight_decay,
        # "devices": config.devices
    }
    # mamba_config = {
    #     "enc_n_layer": config.num_encoder_layers,
    #     "d_model": config.d_model,
    #     "n_layer": config.n_layer,
    #     "rms_norm": config.rms_norm,
    #     "fused_add_norm": config.fused_add_norm,
    #     "use_fast_path": config.use_fast_path,
    #     "learning_rate": config.learning_rate,
    #     "warmup_steps": config.warmup_steps,
    #     "weight_decay": config.weight_decay,
    #     "devices": config.devices
    # }
    model = MambaEncDec(
        **mamba_config,
        config = mamba_config,
        src_vocab_size=config.src_voc_size,
        tgt_vocab_size=config.tgt_voc_size
    )

    return model