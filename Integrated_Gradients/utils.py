import torch
import torch.nn.functional as F
import cv2
import numpy as np

def calculate_outputs_and_gradients(inputs, model, target_label_idx):
    # do the pre-processing
    predict_idx = None
    gradients = []
    for input in inputs:
        input = pre_processing(input)
        output = model(input)
        output = F.softmax(output, dim=1)
        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()
        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(1, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients, target_label_idx

def pre_processing(obs):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device, requires_grad=True)
    return obs_tensor

# generate the entire images
def generate_entrie_images(img, img_grad, img_grad_overlay, img_integrad, img_integrad_overlay):
# Print dimensions of images
    print(f"Original Image Shape: {img.shape}")
    print(f"Gradient Image Shape: {img_grad.shape}")
    print(f"Gradient Overlay Image Shape: {img_grad_overlay.shape}")
    print(f"Integrated Gradient Image Shape: {img_integrad.shape}")
    print(f"Integrated Gradient Overlay Image Shape: {img_integrad_overlay.shape}")
    blank = np.ones((img_grad.shape[0], 10, 3), dtype=np.uint8) * 255
    blank_hor = np.ones((10, 20 + img_grad.shape[0] * 3, 3), dtype=np.uint8) * 255
    upper = np.concatenate([img[:, :, (2, 1, 0)], blank, img_grad_overlay, blank, img_grad], 1)
    down = np.concatenate([img[:, :, (2, 1, 0)], blank, img_integrad_overlay, blank, img_integrad], 1)
    total = np.concatenate([upper, blank_hor, down], 0)
    total = cv2.resize(total, (550, 364))

    return total
