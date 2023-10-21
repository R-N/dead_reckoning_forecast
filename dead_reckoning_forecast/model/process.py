import torch
from torch import nn
import torch.nn.functional as F
from ..data.video import get_frames
from ..util import array_to_image, show_video
import gc
from .metrics import mape, mse

def train_epoch(model, loader, opt, loss_fn=nn.MSELoss(reduction="none"), val=False, test=False, reduction=torch.linalg.vector_norm):
    val = val or test
    if val:
        model.eval()
    else:
        model.train()
        
    avg_loss = 0
    avg_prediction_loss = 0
    avg_internal_prediction_loss = 0
    avg_reconstruction_loss = 0
    n = 0

    preds = []
    ys = []

    for i, batch in enumerate(loader):
        x, frames, y, w, xy = batch
        ys.extend(y.detach().cpu())
        b = x.size(0)
        x = x.to(model.device)
        frames = frames.to(model.device)
        y = y.to(model.device)
        w = w.to(model.device)
        if not test:
            xy = xy.to(model.device)

        if not val:
            opt.zero_grad()
        
        pred, (x_0, x_1), xy_pred = model(x, frames)
        prediction_loss = loss_fn(pred, y)
        internal_prediction_loss = 0
        reconstruction_loss = 0
        if not test:
            internal_prediction_loss = loss_fn(x_1, x_0)
            reconstruction_loss = loss_fn(xy_pred, xy)

        #print(prediction_loss.shape, w.shape, internal_prediction_loss.shape, reconstruction_loss.shape)


        prediction_loss = reduction(prediction_loss, dim=-1)
        if not test:
            internal_prediction_loss = reduction(internal_prediction_loss, dim=-1)
            reconstruction_loss = reduction(reconstruction_loss, dim=-1)

        prediction_loss = prediction_loss * w
        prediction_loss = torch.sum(prediction_loss, dim=-1) #this is 1 because it has been normed, this is sum because it's weighted sum
        if not test:
            internal_prediction_loss = torch.mean(internal_prediction_loss, dim=-1) #this is 1 because it has been normed
            reconstruction_loss = torch.mean(reconstruction_loss, dim=-1) #this is 1 because it has been normed

        # these should result in 0 dim tensor
        prediction_loss = torch.sum(prediction_loss, dim=-1) 
        if not test:
            internal_prediction_loss = torch.sum(internal_prediction_loss, dim=-1) 
            reconstruction_loss = torch.sum(reconstruction_loss, dim=-1) 
        
        loss = prediction_loss + internal_prediction_loss + reconstruction_loss
        
        if not val:
            (loss/b).backward() # take batch mean
            opt.step()
        
        loss = loss.item()
        prediction_loss = prediction_loss.item()
        if not test:
            internal_prediction_loss = internal_prediction_loss.item()
            reconstruction_loss = reconstruction_loss.item()
        
        avg_loss += loss
        avg_prediction_loss += prediction_loss
        avg_internal_prediction_loss += internal_prediction_loss
        avg_reconstruction_loss += reconstruction_loss

        n += b

        print("batch", loss/b, prediction_loss/b, internal_prediction_loss/b, reconstruction_loss/b)

        preds.extend(pred.detach().cpu())
        #ys.extend(y.detach().cpu())

        gc.collect()
        
    avg_loss /= n
    avg_prediction_loss /= n
    avg_internal_prediction_loss /= n
    avg_reconstruction_loss /= n

    preds = torch.stack(preds)
    ys = torch.stack(ys)

    rmse_ = torch.sqrt(torch.mean(mse(preds, ys, dim=-2))).item()
    wrmse_ = torch.sqrt(torch.mean(mse(preds, ys, weights=True, dim=-2))).item()
    mape_ = torch.mean(mape(preds, ys, dim=-2)).item()
    wmape_ = torch.mean(mape(preds, ys, weights=True, dim=-2)).item()
    
    ret = {
        "avg_loss": avg_loss,
        "avg_prediction_loss": avg_prediction_loss,
        "avg_internal_prediction_loss": avg_internal_prediction_loss,
        "avg_reconstruction_loss": avg_reconstruction_loss,
        "mape": mape_,
        "wmape": wmape_,
        "rmse": rmse_,
        "wrmse": wrmse_,
    }
        
    return ret

def infer_video(model, label_encoder, transformer, file_path):
    frames, v_len = get_frames(file_path, n_frames=16)
    frames = [transformer(array_to_image(f)) for f in frames]
    frames = torch.stack(frames).unsqueeze(0)
    frames = frames.to(model.device)
    pred = model(frames)
    pred = torch.argmax(pred, dim=-1)
    pred = pred.cpu().numpy()
    pred = label_encoder.inverse_transform(pred)[0]
    return pred

def test_infer_video(model, label_encoder, transformer, file_info):
    file_path = str(file_info["file_path"])
    label = file_info["tag"]
    pred = infer_video(model, label_encoder, file_path)
    print(file_path, "pred", pred, "==" if pred == label else "!=", label)
    return show_video(file_path)

def infer(model, x, frames):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    x = x.to(model.device)
    frames = frames.to(model.device)
    pred, *_ = model(x, frames)
    pred = pred.detach().cpu().numpy()
    return pred

def eval(model, x, frames, y):
    pred = infer(model, x, frames)
    rmse_ = torch.sqrt(torch.mean(mse(pred, y, dim=-2))).item()
    wrmse_ = torch.sqrt(torch.mean(mse(pred, y, weights=True, dim=-2))).item()
    mape_ = torch.mean(mape(pred, y, dim=-2)).item()
    wmape_ = torch.mean(mape(pred, y, weights=True, dim=-2)).item()
    
    ret = {
        "mape": mape_,
        "wmape": wmape_,
        "rmse": rmse_,
        "wrmse": wrmse_,
    }
    return ret
