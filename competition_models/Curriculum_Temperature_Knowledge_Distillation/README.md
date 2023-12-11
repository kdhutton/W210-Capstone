## Curriculum Temperature for Knowledge Distillation

*This [notebook](https://github.com/kdhutton/W210-Capstone/blob/main/competition_models/Curriculum_Temperature_Knowledge_Distillation/CTKD_Final_CIFAR.ipynb) is based on the research paper, "[Curriculum Temperature for Knowledge Distillation](https://arxiv.org/abs/2211.16231)"*


#### Summary

The paper "Curriculum Temperature for Knowledge Distillation" explores the limitations of fixed temperatures in knowledge distillation, highlighting their impact on the learning process. It acknowledges the reduced effectiveness of higher temperatures in advanced stages of training and introduces a dynamic strategy to modify the distillation temperature throughout the training period.

#### Methodology

Central to the methodology is the Curriculum Temperature strategy. Rather than using a fixed temperature, the approach starts with a high temperature, which aids in capturing the overall distribution of the teacher model's knowledge. As training advances, the temperature is systematically reduced, focusing on fine-tuning the student model's learning from specific and nuanced patterns of the teacher. This curriculum-based approach to adjusting temperature ensures an optimized and balanced transfer of knowledge from the teacher to the student model across the entirety of the training process.



