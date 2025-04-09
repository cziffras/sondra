import torch 
import numpy as np
import matplotlib.pyplot as plt 
import os 

def to_complex_tensor(img):
    """
    Converts a numpy array or PyTorch tensor to a complex PyTorch tensor.
    Handles different input formats and ensures proper shape.
    
    Args:
        img: Input image (numpy array or PyTorch tensor)
        
    Returns:
        Complex PyTorch tensor with shape (C, H, W)
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
 
    if not torch.is_complex(img):
        # If the tensor is real, assume it's magnitude only and set imaginary part to zero
        if img.ndim == 2:  # (H, W)
            img = img.unsqueeze(0)  # Add channel dimension: (1, H, W)
        elif img.ndim == 3 and img.shape[-1] == 1:  # (H, W, 1)
            img = img.permute(2, 0, 1)  # Reorder to (1, H, W)
        elif img.ndim == 3 and img.shape[0] > 3:  # Likely (H, W, C)
            img = img.permute(2, 0, 1)  # Reorder to (C, H, W)
            
        img = torch.complex(img, torch.zeros_like(img))
    else:
        # If already complex, ensure proper shape
        if img.ndim == 2:  # (H, W)
            img = img.unsqueeze(0)  # Add channel dimension: (1, H, W)
    
    return img

def visualize_complex_img(img, normalize=True):
    """
    Convert complex tensor to visualization format.
    Returns magnitude, phase, real, and imaginary components as numpy arrays.
    
    Args:
        img: Complex tensor with shape (C, H, W)
        normalize: Whether to normalize magnitude to [0, 1]
        
    Returns:
        Dictionary of numpy arrays for visualization
    """
    img = to_complex_tensor(img)
    
    magnitude = torch.abs(img).cpu().numpy()
    phase = torch.angle(img).cpu().numpy()
    real = img.real.cpu().numpy()
    imag = img.imag.cpu().numpy()
    
    # Normalize magnitude if requested
    if normalize:
        for c in range(magnitude.shape[0]):
            channel_min = magnitude[c].min()
            channel_max = magnitude[c].max()
            if channel_max > channel_min:
                magnitude[c] = (magnitude[c] - channel_min) / (channel_max - channel_min)
    
    # Scale phase to [0, 1] for visualization
    phase = (phase + np.pi) / (2 * np.pi)
    
    return {
        'magnitude': magnitude,
        'phase': phase,
        'real': real,
        'imag': imag
    }

def visualize_augmentations(img_tensor, aug, title=None, num_examples=3, channels=False):
    """Helper method to visualize augmentations."""
    
    C, H, W = img_tensor.shape

    if channels: 
        total_rows = num_examples * C
        plt.figure(figsize=(15, 5 * total_rows))
    else:
        total_rows = num_examples
        plt.figure(figsize=(15, 5 * total_rows))
    
    for i in range(num_examples):
        if channels:
            for c in range(C):
                try:
                    view1, view2 = aug(img_tensor)
                    
                    orig_vis = visualize_complex_img(img_tensor)
                    view1_vis = visualize_complex_img(view1)
                    view2_vis = visualize_complex_img(view2)
                    
                    row_index = i * C + c
                    
                    plt.subplot(total_rows, 3, row_index * 3 + 1)
                    plt.imshow(orig_vis['magnitude'][c], cmap='viridis')
                    plt.title(f'Original (Ex {i+1}, Ch {c})')
                    plt.axis('off')
                    
                    plt.subplot(total_rows, 3, row_index * 3 + 2)
                    plt.imshow(view1_vis['magnitude'][c], cmap='viridis')
                    plt.title(f'View 1 (Ex {i+1}, Ch {c})')
                    plt.axis('off')
 
                    plt.subplot(total_rows, 3, row_index * 3 + 3)
                    plt.imshow(view2_vis['magnitude'][c], cmap='viridis')
                    plt.title(f'View 2 (Ex {i+1}, Ch {c})')
                    plt.axis('off')

                except Exception as e:
                    print(f"Error in visualization example {i+1}, channel {c}: {str(e)}")
        else:
            try:
                view1, view2 = aug(img_tensor)
                
                orig_vis = visualize_complex_img(img_tensor)
                view1_vis = visualize_complex_img(view1)
                view2_vis = visualize_complex_img(view2)
                
                plt.subplot(total_rows, 3, i * 3 + 1)
                plt.imshow(orig_vis['magnitude'][0], cmap='viridis')
                plt.title(f'Original (Example {i+1})')
                plt.axis('off')
                
                plt.subplot(total_rows, 3, i * 3 + 2)
                plt.imshow(view1_vis['magnitude'][0], cmap='viridis')
                plt.title(f'View 1 (Example {i+1})')
                plt.axis('off')
                
                plt.subplot(total_rows, 3, i * 3 + 3)
                plt.imshow(view2_vis['magnitude'][0], cmap='viridis')
                plt.title(f'View 2 (Example {i+1})')
                plt.axis('off')
            except Exception as e:
                print(f"Error in visualization example {i+1}: {str(e)}")
    
    plt.tight_layout()

    target_dir = os.path.join("..", "..", "img")  
    os.makedirs(target_dir, exist_ok=True)

    if title:
        file_path = os.path.join(target_dir, f"{title}.png")
    else:
        file_path = os.path.join(target_dir, "augmentations")

    plt.savefig(file_path, bbox_inches='tight')
