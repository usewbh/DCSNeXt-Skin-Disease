import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BalancedHarmonicFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=1, gamma=2, epsilon=0.1,adaptive=False):
        super(BalancedHarmonicFocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.adaptive = adaptive 
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs, targets):
        device = inputs.device
        class_weights = self.class_weights.to(device)
        num_classes = inputs.size(1)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float().to(device)

        smooth_labels = (1 - self.epsilon) * one_hot_targets + self.epsilon / num_classes

        log_probs = F.log_softmax(inputs, dim=1)
        BCE_loss = -(smooth_labels * log_probs).sum(dim=1)
        probs = torch.exp(-BCE_loss)

        class_counts = smooth_labels.sum(dim=0)
        scaling_factor = self.alpha / (class_counts + 1e-8)

        focal_loss = scaling_factor[targets] * (1 - probs) ** self.gamma * BCE_loss

        cb_loss = focal_loss * class_weights[targets]

        cosine_sim = F.cosine_similarity(inputs, smooth_labels, dim=1)
        harmonic_loss = (1 - cosine_sim).mean()

        total_loss = cb_loss.mean() + harmonic_loss

        if self.adaptive:
            with torch.no_grad():
                self.alpha = torch.clamp(self.alpha + 0.01 * (BCE_loss.mean().item() - total_loss.item()), 0.1, 2.0)
                self.gamma = torch.clamp(self.gamma + 0.01 * (BCE_loss.mean().item() - total_loss.item()), 1.0, 5.0)

        return total_loss
class_weights = torch.tensor([])
loss_fn = BalancedHarmonicFocalLoss(class_weights)
