import net
import utils
import trainer

import torch.nn as nn

teacher = net.convModel(3)
student = net.linearModel(teacher)
student_no_dist = net.linearModel(teacher)

cross_entropy = nn.CrossEntropyLoss()
student_criterion = (nn.MSELoss(), nn.KLDivLoss(reduction="batchmean"))

print('Training Teacher')
_ = trainer.train(teacher, utils.TEACHER_EPOCHS, cross_entropy)
_ = trainer.test(teacher)
print('\nTraining Student on Teacher')
_ = trainer.train(student, utils.STUDENT_EPOCHS, student_criterion, teacher)
_ = trainer.test(student)
print('\nTraining Re-initialized Student Only On Labels')
_ = trainer.train(student_no_dist, utils.STUDENT_EPOCHS, cross_entropy)
_ = trainer.test(student_no_dist)