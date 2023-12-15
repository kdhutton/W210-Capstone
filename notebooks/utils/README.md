### How to use loss_function.py 

__Classic Knowledge Distillation__
```
from loss_functions import tkd_kdloss
KD_loss = tkd_kdloss(student_outputs, teacher_outputs, temperature=temperature)
```

__Relational Knowledge Distillation__
```
from loss_functions import DD_loss, AD_loss, RKDDistanceLoss, RKDAngleLoss
distance_loss = RKDDistanceLoss()(student_outputs, teacher_outputs)
angle_loss = RKDAngleLoss()(student_outputs, teacher_outputs)
loss = criterion(student_outputs, target) + 0.1 * (distance_loss + angle_loss)
```

__Curriculum Temperature Knowledge Distillation__
```
from loss_functions import CTKD_loss
loss = CTKD_loss(outputs, labels, teacher_outputs, temp, alpha)
```

__Norm and Direction__
```
from loss_functions import 
```
