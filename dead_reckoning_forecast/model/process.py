import torch
from ..data.video import get_frames
from ..util import array_to_image, show_video

def train_epoch(model, loader, loss_fn, opt, val=False):
    if val:
        model.eval()
    else:
        model.train()
        
    avg_loss = 0
    total_correct = 0
    total_samples = 0
    
    
    for i, batch in enumerate(loader):
        x, label = batch
        x = x.to(model.device)
        label = label.to(model.device)

        if not val:
            opt.zero_grad()
        
        pred = model(x)
        loss = loss_fn(pred, label)
        
        if not val:
            loss.backward()
            opt.step()
        
        loss = loss.item()
        pred_1 = torch.argmax(pred, dim=-1)
        n_correct = (pred_1 == label).sum().item()
        n_samples = label.size(0)
        accuracy = n_correct / n_samples

        #print(f"Step loss: {loss}, accuracy: {accuracy}")
        avg_loss += loss
        total_correct += n_correct
        total_samples += n_samples
        
    avg_loss /= (i+1)
    accuracy = total_correct / total_samples
    
    ret = {
        "avg_loss": avg_loss,
        "accuracy": accuracy
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
    