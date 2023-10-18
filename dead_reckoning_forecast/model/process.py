import torch
from ..data.video import get_frames
from ..util import array_to_image, show_video

def train_epoch(model, loader, loss_fn, opt, val=False):
    if val:
        model.eval()
    else:
        model.train()
        
    avg_loss = 0

    for i, batch in enumerate(loader):
        x, frames, y, w = batch
        x = x.to(model.device)
        frames = frames.to(model.device)
        y = y.to(model.device)
        w = w.to(model.device)

        if not val:
            opt.zero_grad()
        
        pred = model(x, frames)
        loss = loss_fn(pred, y)

        loss = loss * w
        
        if not val:
            loss.backward()
            opt.step()
        
        loss = loss.item()
        #print(f"Step loss: {loss}")
        avg_loss += loss
        
    avg_loss /= (i+1)
    
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
    