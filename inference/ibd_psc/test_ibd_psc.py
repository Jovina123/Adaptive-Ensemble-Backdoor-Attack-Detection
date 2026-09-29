import torch
import torch.nn as nn

from ibd_psc import IBDPSCDetector


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = SmallCNN()

detector = IBDPSCDetector(
    model=model,
    n=2,
    threshold=0.9,
    scale=1.5
)

dummy_input = torch.randn(2, 3, 32, 32)

psc_scores, is_backdoor = detector.detect(dummy_input)

print("PSC scores:", psc_scores)
print("Backdoor detection:", is_backdoor)
print("Test completed successfully.")