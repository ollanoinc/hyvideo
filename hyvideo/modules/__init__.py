from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG


def load_model(args, in_channels, out_channels):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if args.model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            text_states_dim=args.text_states_dim,
            text_states_dim_2=args.text_states_dim_2,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
        )
        return model
    else:
        raise NotImplementedError()
