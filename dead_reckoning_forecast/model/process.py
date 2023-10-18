import torch
from torch import nn
import torch.nn.functional as F
from ..data.video import get_frames
from ..util import array_to_image, show_video
import gc

def train_epoch(model, loader, opt, loss_fn=nn.MSELoss(reduction="none"), val=False, test=False, reduction=torch.linalg.vector_norm):
    val = val or test
    if val:
        model.eval()
    else:
        model.train()
        
    avg_loss = 0
    n = 0

    for i, batch in enumerate(loader):
        x, frames, y, w, xy = batch
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
        loss = loss_fn(pred, y)
        if not test:
            internal_prediction_loss = loss_fn(x_1, x_0)
            reconstruction_loss = loss_fn(xy_pred, xy)

        #print(loss.shape, w.shape, internal_prediction_loss.shape, reconstruction_loss.shape)


        loss = reduction(loss, dim=-1)
        if not test:
            internal_prediction_loss = reduction(internal_prediction_loss, dim=-1)
            reconstruction_loss = reduction(reconstruction_loss, dim=-1)

        loss = loss * w
        loss = torch.sum(loss, dim=-1) #this is 1 because it has been normed, this is sum because it's weighted sum
        if not test:
            internal_prediction_loss = torch.mean(internal_prediction_loss, dim=-1) #this is 1 because it has been normed
            reconstruction_loss = torch.mean(reconstruction_loss, dim=-1) #this is 1 because it has been normed

        if not test:
            loss = loss + internal_prediction_loss + reconstruction_loss
        loss = torch.sum(loss, dim=-1) # this should result in 0 dim tensor
        
        if not val:
            loss.backward()
            opt.step()
        
        loss = loss.item()
        #print(f"Step loss: {loss}")
        avg_loss += loss

        n += b

        print("batch", loss/b)

        gc.collect()
        
    avg_loss /= n
    
    ret = {
        "avg_loss": avg_loss,
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
    