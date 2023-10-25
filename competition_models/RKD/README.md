## Relational Knowledge Distillation

*This [notebook](https://github.com/kdhutton/W210-Capstone/blob/main/competition_models/RKD/Relational_Knowledge_Distillation_Final_CIFAR.ipynb) is based on the research paper, "[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)"*

### Summary

"Relational Knowledge Distillation" presents an innovative technique for the knowledge distillation process, focusing on relationships between the intermediate feature representations in deep neural networks. Instead of merely transferring knowledge based on output distributions or intermediate activations, this approach emphasizes the mutual relationships between data points in the feature space. The authors argue that these relational perspectives capture the structured knowledge of a teacher model more effectively, leading to improved student model performance.

### Methodology

The core of the Relational Knowledge Distillation (RKD) methodology revolves around preserving the distance-wise and angle-wise relationships between data points in the intermediate feature spaces of teacher and student networks. In a given layer, for every pair of data points, the distances and angles between their feature vectors in the teacher network are computed. The student network is then trained to mimic these relationships in its own feature space. This is achieved through the distance distillation loss and angle distillation loss. By preserving such spatial relationships, RKD ensures that the student model learns the structural knowledge embedded in the teacher model's features, going beyond simple activation matching or final output imitation.

