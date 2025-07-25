import torch

TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 150
LR = 1e-3


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
print(f'Device: {device}')