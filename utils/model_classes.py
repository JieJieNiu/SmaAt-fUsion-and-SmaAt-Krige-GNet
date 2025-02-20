from models import unet_precip_regression_lightning as unet_regr
import lightning.pytorch as pl


def get_model_class(model_file) -> tuple[type[pl.LightningModule], str]:
    # This is for some nice plotting
    if "UNet_Attention" in model_file:
        model_name = "UNet Attention"
        model = unet_regr.UNet_Attention
    elif "Node_SmaAt_root" in model_file:
        model_name = "Node_SmaAt_root"
        model = unet_regr.Node_SmaAt_root
    elif "Node_SmaAt_bridge" in model_file:
        model_name = "Node_SmaAt_bridge"
        model = unet_regr.Node_SmaAt_bridge
    elif "Bridge" in model_file:
        model_name = "Node_SmaAt_bridge"
        model = unet_regr.Node_SmaAt_bridge        
    elif "Smaat" in model_file:
        model_name = "SmaAt_UNet"
        model = unet_regr.SmaAt_UNet
    elif "Krige_GNet" in model_file:
        model_name = "Krige_GNet"
        model = unet_regr.Krige_GNet
    elif "Krige" in model_file:
        model_name = "Krige_GNet"
        model = unet_regr.Krige_GNet
    elif "UNetDS_Attention_4kpl" in model_file:
        model_name = "UNetDS Attention with 4kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_1kpl" in model_file:
        model_name = "UNetDS Attention with 1kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_4CBAMs" in model_file:
        model_name = "UNetDS Attention 4CBAMs"
        model = unet_regr.UNetDS_Attention_4CBAMs
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS" in model_file:
        model_name = "UNetDS"
        model = unet_regr.UNetDS
    elif "UNet" in model_file:
        model_name = "UNet"
        model = unet_regr.UNet
    else:
        raise NotImplementedError("Model not found")
    return model, model_name
