from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 8
    max_epochs = 20
    save_interval = 0.25
    eval_interval = 1
    save_epoch_end = True
    n_samples = 8  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 5e-4
    weight_decay = 0.0
    clip_grad = -1.  # no clipping
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    dataset = 'physion_training'
    data_root = './data/Physion'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 1  # train on single frames
    frame_offset = 1  # no offset
    video_len = 150  # take the first 150 frames of each video
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'VQVAE'
    resolution = (128, 128)
    enc_dec_dict = dict(
        resolution=resolution[0],
        in_channels=3,
        z_channels=3,
        ch=64,  # base_channel
        ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
        num_res_blocks=2,
        attn_resolutions=[],
        out_ch=3,
        dropout=0.0,
    )
    vq_dict = dict(
        n_embed=4096,  # vocab_size
        embed_dim=enc_dec_dict['z_channels'],
        percept_loss_w=1.0,
    )

    recon_loss_w = 1.
    quant_loss_w = 1.
    percept_loss_w = vq_dict['percept_loss_w']
