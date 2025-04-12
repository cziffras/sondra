import torch
import torch.fft
import numpy as np
import random
from torch.nn import functional as F

from ..utils.transforms_utils import to_complex_tensor

##############################################
# 1. RandomHole Transformation               #
##############################################
class RandomHole:
    """
    Applies random masking on a complex image, one can see it as a "hole".
    The masked area is replaced with zeros (both real and imaginary parts).
    
    Args:
        hole_scale (tuple): Range of proportions for hole size relative to image dimensions.
                            E.g., (0.1, 0.3) means hole height and width will be between 
                            10% and 30% of image dimensions.
        num_holes (int or tuple): Number of holes to apply. If tuple, randomly selects 
                                  a number between the two values.
    """
    def __init__(self, hole_scale=(0.1, 0.25), num_holes=(1, 3)):
        self.hole_scale = hole_scale
        self.num_holes = num_holes
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        C, H, W = img.shape
        
        # Determine number of holes
        if isinstance(self.num_holes, tuple):
            num_holes = random.randint(self.num_holes[0], self.num_holes[1])
        else:
            num_holes = self.num_holes
        
        # Create mask with ones
        mask = torch.ones((H, W), device=img.device, dtype=img.real.dtype)
        
        # Apply multiple holes
        for _ in range(num_holes):
            # Calculate hole size (can be different for height and width)
            hole_H = int(random.uniform(self.hole_scale[0], self.hole_scale[1]) * H)
            hole_W = int(random.uniform(self.hole_scale[0], self.hole_scale[1]) * W)
            
            # Choose random position
            top = random.randint(0, H - hole_H)
            left = random.randint(0, W - hole_W)
            
            # Set hole area to 0 in mask
            mask[top:top+hole_H, left:left+hole_W] = 0
        
        # Expand mask to match channels and apply
        mask = mask.unsqueeze(0).expand_as(img.real)
        return torch.complex(img.real * mask, img.imag * mask)

##############################################
# 2. IntensityAugment                        #
##############################################
class IntensityAugment:
    """
    Randomly modifies the intensity of each channel in a controlled manner.
    
    Args:
        intensity_factor (float): Maximum variation around 1.
        per_channel (bool): Whether to apply different factors to each channel.
    """
    def __init__(self, intensity_factor=0.05, per_channel=True):
        self.intensity_factor = intensity_factor
        self.per_channel = per_channel
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        C, H, W = img.shape
        
        if self.per_channel:
            # Different factor for each channel
            factors = torch.empty(C, device=img.device).uniform_(
                1 - self.intensity_factor, 1 + self.intensity_factor
            )
            factors = factors.view(C, 1, 1)
        else:
            # Same factor for all channels
            factors = torch.tensor(
                [random.uniform(1 - self.intensity_factor, 1 + self.intensity_factor)],
                device=img.device
            ).view(1, 1, 1)
            
        return img * factors

##############################################
# 3. FourierAugment + Geometric              #
##############################################
class FourierAugment:
    """
    Transforms the image to Fourier domain, applies random masking on the spectrum,
    then transforms back to spatial domain.
    
    Args:
        mask_prob (float): Probability of masking (zeroing) each Fourier coefficient.
        structured_mask (bool): If True, applies structured patterns rather than random masking.
        preserve_center (bool): If True, preserves the center of the spectrum (DC component).
        center_radius (float): Radius of the center area to preserve (as proportion of dimensions).
    """
    def __init__(self, mask_prob=0.1, structured_mask=True, preserve_center=True, center_radius=0.1):
        self.mask_prob = mask_prob
        self.structured_mask = structured_mask
        self.preserve_center = preserve_center
        self.center_radius = center_radius
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        C, H, W = img.shape
        out = torch.empty_like(img)
        
        for c in range(C):
            # Compute FFT for channel c
            fft_channel = torch.fft.fft2(img[c])
            fft_channel = torch.fft.fftshift(fft_channel)
            
            if self.structured_mask:
                # Create structured pattern
                mask = self._create_structured_mask(H, W, img.device)
            else:
                # Random mask with preservation
                mask = (torch.rand((H, W), device=img.device) > self.mask_prob).to(torch.float32)
            
            if self.preserve_center:
                # Create center mask to preserve DC component
                center_h, center_w = H // 2, W // 2
                radius_h = int(H * self.center_radius)
                radius_w = int(W * self.center_radius)
                
                y, x = torch.meshgrid(
                    torch.arange(H, device=img.device),
                    torch.arange(W, device=img.device),
                    indexing='ij'
                )
                
                # Create circular mask
                dist = ((y - center_h)**2 / (radius_h**2) + 
                    (x - center_w)**2 / (radius_w**2))
                center_mask = (dist <= 1).to(torch.float32)  # Convert to float32 explicitly
                
                # Combine masks: preserve center and apply pattern elsewhere
                # Make sure mask and center_mask are both float tensors, not complex
                mask = torch.where(center_mask > 0, torch.ones_like(mask), mask)
                
            # Apply mask
            fft_channel = fft_channel * mask
            
            # Back to spatial domain
            fft_channel = torch.fft.ifftshift(fft_channel)
            out[c] = torch.fft.ifft2(fft_channel)
            
        return out
            
    def _create_structured_mask(self, H, W, device):
        """Creates a structured mask for Fourier domain augmentation."""
        mask_type = random.choice(["lines", "rings", "sectors"])
        mask = torch.ones((H, W), device=device)
        
        if mask_type == "lines":
            # Create line pattern
            num_lines = random.randint(3, 8)
            for _ in range(num_lines):
                line_width = random.randint(1, max(2, int(min(H, W) * 0.02)))
                if random.random() < 0.5:  # Horizontal
                    pos = random.randint(0, H - 1)
                    mask[max(0, pos-line_width):min(H, pos+line_width), :] = 0
                else:  # Vertical
                    pos = random.randint(0, W - 1)
                    mask[:, max(0, pos-line_width):min(W, pos+line_width)] = 0
                    
        elif mask_type == "rings":
            # Create concentric rings
            center_h, center_w = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            max_radius = min(H, W) // 2
            
            num_rings = random.randint(2, 5)
            for i in range(num_rings):
                inner_r = random.uniform(0.1, 0.9) * max_radius
                outer_r = inner_r + random.uniform(0.05, 0.15) * max_radius
                ring_mask = (dist >= inner_r) & (dist <= outer_r)
                mask[ring_mask] = 0
                
        elif mask_type == "sectors":
            # Create pie-slice sectors
            center_h, center_w = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            
            # Convert to angles
            angles = torch.atan2(y - center_h, x - center_w)
            
            num_sectors = random.randint(2, 6)
            sector_width = 2 * np.pi / (2 * num_sectors)
            
            for i in range(num_sectors):
                start_angle = i * 2 * sector_width
                sector_mask = ((angles >= start_angle) & 
                              (angles < start_angle + sector_width))
                mask[sector_mask] = 0
                
        return mask

###########################################################
# 4. SAR-Specific augmentation: SpeckleAugment            #
###########################################################
class SpeckleAugment:
    """
    TOFIX = multiplicative 

    Applies realistic speckle noise typical of SAR images.
    See : https://core.ac.uk/download/pdf/17332528.pdf p.4-5
    
    Args:
        intensity (float): Controls the intensity of the speckle noise.
        distribution (str): Type of distribution for speckle ('gamma' or 'rayleigh').
    """
    def __init__(self, intensity=0.3, distribution='gamma'):
        self.intensity = intensity
        self.distribution = distribution.lower()
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        C, H, W = img.shape
        
        if self.distribution == 'gamma':
            # Gamma distribution for multiplicative speckle
            k = 1.0 / self.intensity  # Shape parameter
            noise = torch.from_numpy(
                np.random.gamma(k, scale=1.0/k, size=(C, H, W))
            ).to(img.device).to(img.real.dtype)
        else:  
            # Rayleigh distribution
            sigma = self.intensity / np.sqrt(2)
            noise_real = torch.randn(C, H, W, device=img.device) * sigma
            noise_imag = torch.randn(C, H, W, device=img.device) * sigma
            noise = torch.sqrt(noise_real**2 + noise_imag**2)
        
        # Apply as multiplicative noise to magnitude
        magnitude = torch.abs(img)
        phase = torch.angle(img)
        
        # Apply noise
        new_magnitude = magnitude * noise
        
        # Convert back to complex
        new_real = new_magnitude * torch.cos(phase)
        new_imag = new_magnitude * torch.sin(phase)
        
        return torch.complex(new_real, new_imag)

##############################################
# 5. SAR-Specific: PhaseAugment              #
##############################################
class PhaseAugment:
    """
    Modifies the phase component of SAR data while preserving magnitude.
    
    Args:
        max_shift (float): Maximum phase shift in radians.
        spatial_freq (int): Spatial frequency of phase variations.
        apply_pattern (bool): Whether to apply structured phase patterns.
    """
    def __init__(self, max_shift=0.3, spatial_freq=5, apply_pattern=True):
        self.max_shift = max_shift
        self.spatial_freq = spatial_freq
        self.apply_pattern = apply_pattern
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        C, H, W = img.shape

        magnitude = torch.abs(img)
        phase = torch.angle(img)
        
        if self.apply_pattern:
            # Create spatially varying phase pattern
            y, x = torch.meshgrid(
                torch.arange(H, device=img.device) / H,
                torch.arange(W, device=img.device) / W,
                indexing='ij'
            )
            
            pattern_type = random.choice(["waves", "circular", "random"])
            
            # Sinusoidal waves
            if pattern_type == "waves":
                freq = random.uniform(1, self.spatial_freq)
                direction = random.uniform(0, 2 * np.pi)
                # Convert direction to tensor
                direction_cos = torch.tensor(np.cos(direction), device=img.device)
                direction_sin = torch.tensor(np.sin(direction), device=img.device)
                phase_pattern = self.max_shift * torch.sin(
                    2 * np.pi * freq * (x * direction_cos + y * direction_sin)
                )
                
            # Circular pattern
            elif pattern_type == "circular":
                center_y = random.uniform(0, 1)
                center_x = random.uniform(0, 1)
                dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
                phase_pattern = self.max_shift * torch.sin(2 * np.pi * self.spatial_freq * dist)
                
            else:  
                # Random spatial pattern
                phase_pattern = torch.randn(H, W, device=img.device) * self.max_shift
                # Apply smoothing to create spatial correlation
                kernel_size = max(3, int(min(H, W) / 20))
                if kernel_size % 2 == 0:  # Ensure odd kernel size
                    kernel_size += 1
                phase_pattern = F.avg_pool2d(
                    phase_pattern.unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2
                ).squeeze(0)
            
            # Apply pattern to all channels
            new_phase = phase + phase_pattern.unsqueeze(0)
            
        else:
            # Simple random phase shift (please avoid to use it)
            phase_shift = torch.randn(C, 1, 1, device=img.device) * self.max_shift
            new_phase = phase + phase_shift
            
        new_real = magnitude * torch.cos(new_phase)
        new_imag = magnitude * torch.sin(new_phase)
        
        return torch.complex(new_real, new_imag)

##############################################
# 6. Geometric: RotateFlip  (TO FIX)         #
##############################################
class RotateFlip:
    """
    Applies rotation (90° increments) and/or flips to the complex image.
    
    Args:
        rotate_prob (float): Probability of applying rotation.
        flip_prob (float): Probability of applying flip.
    """
    def __init__(self, rotate_prob=0.5, flip_prob=0.5):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        
        # Random rotation by 90, 180, or 270 degrees
        if random.random() < self.rotate_prob:
            k = random.randint(1, 3) 
            img = torch.rot90(img, k, dims=(1, 2)) # Caution : permutes dimensions
        
        # Random flip (horizontal, vertical, or both)
        if random.random() < self.flip_prob:
            flip_mode = random.randint(0, 2)
            if flip_mode == 0 or flip_mode == 2:  # Horizontal or both
                img = torch.flip(img, dims=(2,))
            if flip_mode == 1 or flip_mode == 2:  # Vertical or both
                img = torch.flip(img, dims=(1,))
                
        return img

##############################################
# 7. Elastic Deformation (SAR-specific)      #
##############################################
class ElasticDeformation:
    """
    Applies elastic deformation to SAR images while preserving complex values.
    Uses PyTorch's grid_sample for better hardware acceleration.
    
    Args:
        alpha (float): Controls the intensity of deformation.
        sigma (float): Controls the smoothness of deformation.
        p (float): Probability of applying the transformation.
    """
    def __init__(self, alpha=10.0, sigma=5.0, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        img = to_complex_tensor(img)
        C, H, W = img.shape
        
        # Create displacement field
        dx = torch.randn(H, W, device=img.device)
        dy = torch.randn(H, W, device=img.device)
        
        # Apply Gaussian filter to create smooth deformation field
        kernel_size = int(self.sigma * 4) | 1  # Ensure odd kernel size
        dx = F.avg_pool2d(
            dx.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        ).squeeze(0).squeeze(0)
        
        dy = F.avg_pool2d(
            dy.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        ).squeeze(0).squeeze(0)
        
        # Normalize and scale
        dx = dx * self.alpha / dx.abs().max()
        dy = dy * self.alpha / dy.abs().max()
        
        # Create sampling grid
        y, x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing='ij'
        )
        
        x = x.float() + dx
        y = y.float() + dy
        
        # Normalize coordinates to [-1, 1]
        x = 2 * (x / (W - 1)) - 1
        y = 2 * (y / (H - 1)) - 1
        
        # Stack coordinates
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        
        # Apply to real and imaginary parts separately
        real_part = F.grid_sample(
            img.real.unsqueeze(0).float(), grid, mode='bilinear', align_corners=True
        ).squeeze(0)
        
        imag_part = F.grid_sample(
            img.imag.unsqueeze(0).float(), grid, mode='bilinear', align_corners=True
        ).squeeze(0)
        
        return torch.complex(real_part, imag_part)

##############################################
# 8. Complete Contrastive Pipeline          #
##############################################
class SARContrastiveAugmentations:
    """
    Generates two views with complementary transformations to encourage the model 
    to learn robust, domain-specific features.
    
    Args:
        hole_scale (tuple): Range for hole sizes.
        num_holes (tuple): Range for number of holes.
        intensity_factor (float): Intensity variation.
        speckle_options (tuple): (intensity, distribution) for SpeckleAugment.
        phase_shift (float): Maximum phase shift.
        elastic_options (tuple): (alpha, sigma) for ElasticDeformation.
        fourier_options (tuple): (mask_prob, preserve_center) for FourierAugment.
        use_elastic (bool): Whether to use elastic deformation.
        rotate_flip_prob (float): Probability of geometric transformations.
    """
    def __init__(self,
                 hole_scale=(0.05, 0.15),
                 num_holes=(1, 2),
                 intensity_factor=0.05,
                 speckle_options=(0.2, 'gamma'),
                 phase_shift=0.3,
                 elastic_options=(5.0, 3.0),
                 fourier_options=(0.1, True),
                 use_elastic=True,
                 rotate_flip_prob=0.5):
        
        self.random_hole = RandomHole(hole_scale=hole_scale, num_holes=num_holes)
        self.intensity_aug = IntensityAugment(intensity_factor=intensity_factor)
        self.speckle_aug = SpeckleAugment(
            intensity=speckle_options[0], distribution=speckle_options[1]
        )
        self.phase_aug = PhaseAugment(max_shift=phase_shift)
        self.fourier_aug = FourierAugment(
            mask_prob=fourier_options[0], preserve_center=fourier_options[1]
        )
        self.rotate_flip = RotateFlip(
            rotate_prob=rotate_flip_prob, flip_prob=rotate_flip_prob
        )
        
        self.use_elastic = use_elastic
        if use_elastic:
            self.elastic_aug = ElasticDeformation(
                alpha=elastic_options[0], sigma=elastic_options[1]
            )
        
    def __call__(self, img):
        img = to_complex_tensor(img)
        
        # Apply common geometric transformations first
        # (both views get the same rotation/flipping for better correspondence)
        if random.random() < 0.5:
            img = self.rotate_flip(img)
        
        # Key Thing here :
        # First view - focus on spatial/intensity domain
        view1 = img.clone()
        view1 = self.random_hole(view1)
        view1 = self.intensity_aug(view1)
        view1 = self.speckle_aug(view1)
        if self.use_elastic and random.random() < 0.5:
           view1 = self.elastic_aug(view1)
            
        # Second view - focus on frequency/phase domain
        view2 = img.clone()
        view2 = self.random_hole(view2)
        view2 = self.phase_aug(view2)
        view2 = self.fourier_aug(view2)
        
        return view1, view2

