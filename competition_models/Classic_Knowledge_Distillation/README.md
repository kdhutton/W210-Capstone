## Classic Knowledge Distillation

*This [notebook](https://github.com/kdhutton/W210-Capstone/blob/main/competition_models/Classic_Knowledge_Distillation/Classic_Knowledge_Distillation_Final_CIFAR.ipynb) is based on the research paper, "[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)"*

#### Summary

The research paper, "Distilling the Knowledge in a Neural Network" by Geoffrey Hinton, Oriol Vinyals, and Jeff Dean introduces the concept of "knowledge distillation." The authors propose a method where the knowledge from a large, cumbersome model (referred to as the "teacher") is transferred to a smaller, more efficient model (the "student"). This transfer is achieved by training the student model to mimic the softened output distributions (or logits) of the teacher model rather than the hard labels. The softening is performed using a temperature parameter. The resulting student model, while being compact, retains much of the accuracy and generalization capabilities of the larger teacher model, making it suitable for deployment in resource-constrained environments.

#### Methodology

"knowledge distillation" is the model to transfer knowledge from a large, complex neural network, "the teacher" to a smaller, more efficient one, "the student". Rather than using the hard labels traditionally used in training, the student is trained to mimic the output distributions (softened logits) of the teacher. This softening is done using a temperature parameter which, when increased, makes the model outputs smoother and easier for the student model to learn. By learning from these softer probabilities, the student model can generalize better, capturing the nuanced patterns and knowledge embedded in the teacher's outputs. The resulting student model is not only more lightweight but also retains much of the performance of its teacher counterpart.
