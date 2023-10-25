## Relational Knowledge Distillation

*This [notebook](https://github.com/kdhutton/W210-Capstone/blob/main/competition_models/Relational_Knowledge_Distillation/Relational_Knowledge_Distillation_Final_CIFAR.ipynb) is based on the research paper, "[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)"*

### Summary

"Relational Knowledge Distillation" presents an innovative method for knowledge distillation that places a primary focus on the relationships between intermediate feature representations in deep neural networks. Instead of simply transferring knowledge via output distributions or intermediate activations, this approach emphasizes the interconnectedness among data points within the feature space. According to the authors, this relational approach enhances the performance of the student model by more adeptly capturing the structured knowledge embedded in the teacher model.

### Methodology

The essence of the Relational Knowledge Distillation (RKD) method lies in preserving the distance-based and angle-based relationships between data points in the intermediate feature spaces of both teacher and student networks. At each layer, the methodology calculates the distances and angles between the feature vectors of every pair of data points in the teacher network. Subsequently, the student network is trained to replicate these relationships within its own feature space, accomplished through the distance distillation loss and angle distillation loss. By maintaining these spatial relationships, RKD ensures that the student model grasps the structural knowledge inherent in the features of the teacher model, surpassing mere activation matching or imitation of the final output.

