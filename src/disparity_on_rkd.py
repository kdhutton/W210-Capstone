from utils.disparity_tools import one_hot_encode, calculate_recall_multiclass, evaluate_model_with_gender_multiclass 
from train_teacher import train_teacher
from data_tools.wider_dataloader import remap_classes, custom_collate, make_wider_datasets
from utils.disparity_tools import one_hot_encode, calculate_recall_multiclass, evaluate_model_with_gender_multiclass 
from utils.compare_tools import compare_model_size, compare_inference_time, compare_performance_metrics
from utils.model_metrics_report import evaluate_accuracy, evaluate_precision, evaluate_recall, evaluate_f1, evaluate_model_size, evaluate_inference, evaluate_disparity 
import critic

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from torchvision.models import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# Suppress all warnings
warnings.filterwarnings("ignore")



def train_student(student, teacher, student_optimizer, student_loss_fn, critic, critic_optimizer, critic_loss_fn,
                  lambda_factor, temperature, alpha, epsilon=epsilon, margin=margin, patience=student_patience, 
                  epochs=student_epochs, device=device, base_save_dir=base_save_dir,
                  student_scheduler=None, critic_scheduler=None, plot=False):

    
    train_accuracies = []
    train_disparities = []
    train_mean_non_zero_abs_disparities = []
    train_losses = []
    train_main_losses = []
    train_critic_losses = []
    train_kd_losses = []
    val_accuracies = []
    val_disparities = []
    val_mean_non_zero_abs_disparities = []
    val_losses = []
    val_main_losses = []
    val_critic_losses = []
    val_kd_losses = []
    
    patience_counter = 0 
    best_val_accuracy = 0
    best_val_loss = float('inf')
    best_val_mean_abs_disparity = 0
    student_best_model_state = None

    teacher.eval()
    teacher.to(device)

    student.to(device)
    critic.to(device)
    
    # Create a subdirectory for the current lambda_factor
    lambda_dir = os.path.join(base_save_dir, f'STUDENT_lambda_{lambda_factor}')
    os.makedirs(lambda_dir, exist_ok=True)

    print(f'Training Student with Lambda Value of {lambda_factor}')
    
    # Training and Validation Loop
    for epoch in range(epochs):
        # Initialize metrics for each epoch
        epoch_train_disparities = []
        epoch_train_losses = []
        epoch_train_accuracies = []
        epoch_val_disparities = []
        epoch_val_losses = []
        epoch_val_accuracies = []
    
        confusion_male = np.zeros((num_classes, num_classes))
        confusion_female = np.zeros((num_classes, num_classes))
        total_correct = 0
        total_samples = 0
        total_val_loss = 0.0
        num_batches = 0

        # Training
        student.train()
        for batch_data in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}, Training'):
            # Load data to device
            images = batch_data["img"].to(device)
            labels = batch_data["label"].to(device)
            gender_scores = batch_data["target"].to(device)
            
            # Forward pass through student
            student_output = student(images)
            class_predictions = torch.argmax(student_output, dim=1)
            
            # Compute bias
            total_samples += labels.size(0)
            total_correct += (class_predictions == labels).sum().item()
            num_batches += 1
            recall_diff = evaluate_model_with_gender_multiclass(class_predictions, labels, gender_scores, num_classes=num_classes)
            confusion_male += recall_diff[1]
            confusion_female += recall_diff[2]
            bias = np.mean(recall_diff[0])
            bias_mean = torch.tensor([bias], device=device, dtype=torch.float32)

            critic_optimizer.zero_grad()
            
            for param in critic.parameters():
                param.requires_grad = True
            for param in student.parameters():
                param.requires_grad = False
                
            critic.train()
            student.eval()
            
            critic_output = critic(student_output)
            critic_loss = critic_loss_fn(critic_output, bias_mean)
            critic_loss.backward(retain_graph=True)
    
            critic_optimizer.step()
    
            for param in critic.parameters():
                param.requires_grad = False
            for param in student.parameters():
                param.requires_grad = True
                
            student.train()
            critic.eval()
    
            student_optimizer.zero_grad()
    
            critic_output = critic(student_output)
            main_loss = student_loss_fn(student_output, labels)

            with torch.no_grad():
                teacher_output = teacher(images)


            distance_loss = RKDDistanceLoss()(student_output, teacher_output)
            angle_loss = RKDAngleLoss()(student_output, teacher_output)
            kd_loss = main_loss + 2 * (distance_loss + angle_loss)
            
            
            combined_loss = (kd_loss) * max(1, lambda_factor * (abs(critic_output[0][0]) - epsilon + margin) + 1) 
    
            combined_loss.backward()
            student_optimizer.step()
    
            # Calculate and accumulate metrics
            accuracy = (class_predictions == labels).float().mean().item()
            epoch_train_accuracies.append(accuracy)
            epoch_train_disparities.append(bias)
        
            # Record the losses
            epoch_train_losses.append((combined_loss.item(), main_loss.item(), critic_loss.item(), kd_loss.item()))
    
        confusion_male /= num_batches
        confusion_female /= num_batches
    
        # Calculate training metrics for the epoch
        train_epoch_disparity = calculate_recall_multiclass(confusion_male) - calculate_recall_multiclass(confusion_female)
        train_non_zero_abs_values = np.abs(train_epoch_disparity[train_epoch_disparity != 0])
        
        # Store average training metrics for the epoch
        train_accuracy = np.mean(epoch_train_accuracies)
        train_disparity = np.mean(epoch_train_disparities)
        train_mean_non_zero_abs_disparity = np.mean(train_non_zero_abs_values)
        train_combined_loss = np.mean([x[0] for x in epoch_train_losses])
        train_main_loss = np.mean([x[1] for x in epoch_train_losses])
        train_critic_loss = np.mean([x[2] for x in epoch_train_losses])
        train_kd_loss = np.mean([x[3] for x in epoch_train_losses])

        train_accuracies.append(train_accuracy)
        train_disparities.append(train_disparity)
        train_mean_non_zero_abs_disparities.append(train_mean_non_zero_abs_disparity)
        train_losses.append(train_combined_loss)
        train_main_losses.append(train_main_loss)
        train_critic_losses.append(train_critic_loss)
        train_kd_losses.append(train_kd_loss)

        # Validation Phase
        confusion_male = np.zeros((num_classes, num_classes))
        confusion_female = np.zeros((num_classes, num_classes))
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        student.eval()
        with torch.no_grad():
            for batch_data in tqdm(testloader, desc=f'Epoch {epoch+1}/{epochs}, Validation'):
                # Load data to device
                images = batch_data["img"].to(device)
                labels = batch_data["label"].to(device)
                gender_scores = batch_data["target"].to(device)
        
                # Forward pass
                student_output = student(images)
                val_critic_output = critic(student_output)
                class_predictions = torch.argmax(student_output, dim=1)
                teacher_output = teacher(images)
                
                # Calculate and accumulate validation metrics
                accuracy = (class_predictions == labels).float().mean().item()
    
                # Compute bias
                num_batches += 1
                recall_diff = evaluate_model_with_gender_multiclass(class_predictions, labels, gender_scores, num_classes=num_classes)
                confusion_male += recall_diff[1]
                confusion_female += recall_diff[2]
                
                # Calculate validation losses (similar to training losses)
                batch_bias = np.mean(recall_diff[0])
                mean_batch_bias = torch.tensor([batch_bias], device=device, dtype=torch.float32)
                val_main_loss = student_loss_fn(student_output, labels)
                val_critic_loss = critic_loss_fn(val_critic_output, mean_batch_bias)
                val_distance_loss = RKDDistanceLoss()(student_output, teacher_output)
                val_angle_loss = RKDAngleLoss()(student_output, teacher_output) 
                val_kd_loss = val_main_loss + 2 * (val_distance_loss + val_angle_loss)
                val_combined_loss = (val_kd_loss) * max(1, lambda_factor * (abs(val_critic_output[0][0]) - epsilon + margin) + 1)
        
                
                val_combined_loss = alpha * val_kd_loss + (1 - alpha) * max(1, lambda_factor * (abs(val_critic_output[0][0]) - epsilon + margin) + 1) * val_main_loss
 
                epoch_val_accuracies.append(accuracy)
                epoch_val_losses.append((val_combined_loss.item(), val_main_loss.item(), val_critic_loss.item(), val_kd_loss.item()))
                
            confusion_male /= num_batches
            confusion_female /= num_batches
    
            val_epoch_disparity = calculate_recall_multiclass(confusion_male) - calculate_recall_multiclass(confusion_female)
            val_non_zero_abs_values = np.abs(val_epoch_disparity[val_epoch_disparity != 0])
    
            # Store average training metrics for the epoch
            val_accuracy = np.mean(epoch_val_accuracies)
            val_disparity = np.mean(epoch_val_disparities)
            val_mean_non_zero_abs_disparity = np.mean(val_non_zero_abs_values)
            val_combined_loss = np.mean([x[0] for x in epoch_val_losses])
            val_main_loss = np.mean([x[1] for x in epoch_val_losses])
            val_critic_loss = np.mean([x[2] for x in epoch_val_losses])
            val_kd_loss = np.mean([x[3] for x in epoch_val_losses])

            val_accuracies.append(val_accuracy)
            val_disparities.append(val_disparity)
            val_mean_non_zero_abs_disparities.append(val_mean_non_zero_abs_disparity)
            val_losses.append(val_combined_loss)
            val_main_losses.append(val_main_loss)
            val_critic_losses.append(val_critic_loss)
            val_kd_losses.append(val_kd_loss)

            critic_scheduler.step(val_critic_loss)
            student_scheduler.step(val_main_loss)
        
            # Check if current validation combined loss is lower than the best combined loss
        if val_combined_loss < best_val_loss:
            best_val_loss = val_combined_loss
            best_val_accuracy = val_accuracy
            best_val_mean_non_zero_abs_disparity = val_mean_non_zero_abs_disparity
        
            # Create a mapping of class recall disparities
            class_recall_mapping = {class_name: val_epoch_disparity[int(class_label)] for class_label, class_name in class_idx.items()}
        
            student_best_model_state = {
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'best_val_mean_abs_disparity': best_val_mean_non_zero_abs_disparity,
                'class_recall_mapping': class_recall_mapping
            }
            save_path = os.path.join(lambda_dir, f'STUDENT_best_model_lambda_{lambda_factor}.pth')
            torch.save(student_best_model_state, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        print("\n" + "="*50)
        print(f"STUDENT - Lambda {lambda_factor} - Epoch {epoch + 1} Metrics:")
        print("-"*50)
        print(f"TRAINING Accuracy: {train_accuracy:.6f}, VALIDATION Accuracy: {val_accuracy:.4f}")
        print(f"TRAINING Disparity: {train_mean_non_zero_abs_disparity:.6f}, VALIDATION Disparity: {val_mean_non_zero_abs_disparity:.4f}")
        print(f"TRAINING Combined Loss: {train_combined_loss:.6f}, VALIDATION Combined Loss: {val_combined_loss:.4f}")
        print("-"*50 + "\n")
        # Print disparities by class label
        for class_label, recall_diff in class_recall_mapping.items():
            print(f"Class {class_label}: Val Disparity = {recall_diff}")
        print("="*50 + "\n")

        if plot:
          
            # Plotting
            plt.figure(figsize=(15, 10))
            
            # Plot Training and Validation Accuracy
            plt.subplot(2, 2, 1)
            plt.plot(train_accuracies, label='Training Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.title('Student Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot Training and Validation Disparity
            plt.subplot(2, 2, 2)
            plt.plot(train_mean_non_zero_abs_disparities, label='Training Mean Absolute Disparity')
            plt.plot(val_mean_non_zero_abs_disparities, label='Validation Mean Absolute Disparity')
            plt.title('Student Training and Validation Mean Absolute Disparity')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Disparity')
            plt.legend()
            
            # Plot Training Loss Components, Including Combined Loss
            plt.subplot(2, 2, 3)
            plt.plot(train_losses, label='Training Combined Loss')
            plt.plot(train_main_losses, label='Training Main Loss')
            plt.plot(train_critic_losses, label='Training Critic Loss')
            plt.plot(train_kd_losses, label='Training KD Loss')
            plt.title('Student Training Loss Components')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
            # Plot Validation Loss Components, Including Combined Loss
            plt.subplot(2, 2, 4)
            plt.plot(val_losses, label='Validation Combined Loss')
            plt.plot(val_main_losses, label='Validation Main Loss')
            plt.plot(val_critic_losses, label='Validation Critic Loss')
            plt.plot(val_kd_losses, label='Validation KD Loss')
            plt.title('Student Validation Loss Components')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
                
            plt.tight_layout()
            plt.show()

    best_epoch = student_best_model_state['epoch'] + 1 if student_best_model_state else epochs
    print(f"Finished Training STUDENT with lambda value of {lambda_factor}. Best epoch number: {best_epoch}")

    return student_best_model_state


if __name__ == "__main__":
    student_names = sorted(name for name in Models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(Models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument('-f') # add to make this run in collab
    parser.add_argument("--student_name", type=str, default="efficientnetb0", choices=student_names, help="student architecture")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
    parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--teacher", type=str, default="efficientnetb3", help="teacher architecture")
    parser.add_argument("--cls_loss_factor", type=float, default=1.0, help="cls loss weight factor")
    parser.add_argument("--kd_loss_factor", type=float, default=1.0, help="KD loss weight factor")
    parser.add_argument("--temperature", type=float, default=4.0, help="temperature")
    parser.add_argument("--nd_loss_factor", type=float, default=1.0, help="ND loss weight factor")
    parser.add_argument("--alpha", type=float, default=0.9, help="Used for rkd loss function")
    parser.add_argument("--epsilon", type=float, default=0.5, help="adversarial epsilon")
    parser.add_argument("--margin", type=float, default=0.01, help="adversarial margin")

    parser.add_argument("--warm_up", type=float, default=20.0, help='loss weight warm up epochs')
    parser.add_argument("--teacher_patience", type=int, default=1, help='patience for early stopping')
    parser.add_argument("--student_patience", type=int, default=10, help='patience for early stopping')
    parser.add_argument("--teacher_lambda_factor_list", type=list, default=[0], help='lambda factor to reduce disparity')
    parser.add_argument("--student_lambda_factor_list", type=list, default=[0,0.5,1,2], help='lambda factor to reduce disparity')

    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument("--resume", type=str, help="best ckpt's path to resume most recent training")
    parser.add_argument("--base_save_dir", type=str, default="./run/RKD", help="save path, eg, acc_loss, weights, tensorboard, and so on")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [line:%(lineno)d] %(message)s', 
                        datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    # args.batch_size = args.batch_size * len(args.gpus)
    args.batch_size = args.batch_size * 1

    logger.info(colorstr('green', "Distribute train, total batch size:{}, epoch:{}".format(args.batch_size, args.epochs)))

    # pull in datasets
    train_dataset, test_dataset = make_wider_datasets()

    # Hyperparameters / Inputs
    teacher_learning_rate =  args.lr
    student_learning_rate = teacher_learning_rate/5
    teacher_epochs = args.epochs
    student_epochs = teacher_epochs
    teacher_patience = args.teacher_patience
    student_patience = args.student_patience
    temperature = args.temperature
    alpha = args.alpha
    momentum = args.momentum
    batch_size = args.batch_size
    num_workers = args.num_workers
    weight_decay = args.weight_decay
    epsilon = args.epsilon
    margin = args.margin
    num_classes = 16
    base_save_dir = args.base_save_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # list of lambda values to loop through for grid search
    teacher_lambda_factor_list = args.teacher_lambda_factor_list
    student_lambda_factor_list = args.student_lambda_factor_list
    
    
    
    train_dataset, test_dataset = make_wider_datasets()
    
    # store class id mappings
    class_idx = remap_classes()
    
    num_classes = 16
    
    
    # Create dict to store best model states
    teacher_model_states_best = {}
    student_model_states_best = {}
    
    
    # Loop through the lambda_factor_list for teacher debiasing
    for lambda_factor in teacher_lambda_factor_list:
        
        # Load EfficientNet B3 model for Teacher
        teacher = models.efficientnet_b3(pretrained=True)
        
        # Determine the number of output features from the feature extractor part of EfficientNet B3
        num_ftrs = teacher.classifier[1].in_features
        
        # Modify the classifier layer of the EfficientNet model to match the number of classes
        teacher.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # Redefine the main model optimizer if needed
        teacher_optimizer = optim.Adam(teacher.parameters(), lr=teacher_learning_rate)
        teacher_loss_fn = nn.CrossEntropyLoss()
        
        # Train the teacher 
        best_model_state = train_teacher(teacher, teacher_optimizer, teacher_loss_fn, batch_size, train_dataset, test_dataset, lambda_factor, num_classes, class_idx,
                                       teacher_patience, teacher_epochs, device, base_save_dir=base_save_dir, plot = False)
        
        teacher_model_states_best[lambda_factor] = best_model_state
    
    # Save the collective best model states to a file
    teacher_collective_save_path = os.path.join(base_save_dir, 'teacher_model_states_best.pth')
    torch.save(teacher_model_states_best, teacher_collective_save_path)
    
    
    
    # Specify the lambda_factor for the teacher model to load
    lambda_factor = 0
    
    # Define the path to the saved model file for this lambda_factor
    lambda_dir = os.path.join(base_save_dir, f'TEACHER_lambda_{lambda_factor}')
    teacher_path = os.path.join(lambda_dir, f'TEACHER_best_model_lambda_{lambda_factor}.pth')
    
    # Initialize the EfficientNet model without pre-trained weights
    teacher = models.efficientnet_b3(pretrained=False)
    
    # Adjust the classifier layer to match the number of classes
    num_ftrs = teacher.classifier[1].in_features
    teacher.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Load the model state
    teacher_best_model_state = torch.load(teacher_path)
    teacher.load_state_dict(teacher_best_model_state['teacher_state_dict'])
    
    # Loop through the lambda_factor_list for student debiasing
    for lambda_factor in student_lambda_factor_list:
            # Load EfficientNet B0 model for Student
        student = models.efficientnet_b0(pretrained=True)
        
        # Determine the number of output features from the feature extractor part of EfficientNet B0
        num_ftrs = student.classifier[1].in_features  # This is the correct number of input features for the adversarial classifier
        
        # Modify the classifier layer of the EfficientNet model to match the number of classes
        student.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # Initialize the Critic model
        critic = critic.Critic(input_size=num_classes) # Adjust the input size based on the model's output
        critic_optimizer = optim.Adam(critic.parameters(), lr=student_learning_rate, weight_decay=weight_decay)
        critic_scheduler = lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', factor=0.2, patience=5, min_lr=0.0001)
        critic_loss_fn = torch.nn.MSELoss()
    
        # Redefine the main model optimizer if needed
        student_optimizer = optim.Adam(student.parameters(), lr=student_learning_rate, weight_decay=weight_decay)
        student_scheduler = lr_scheduler.ReduceLROnPlateau(student_optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001)
        student_loss_fn = nn.CrossEntropyLoss()
    
        # Train the model
        student_best_model_state = train_student(student, teacher, student_optimizer, student_loss_fn, critic, critic_optimizer, critic_loss_fn,
                                       lambda_factor, temperature, alpha, epsilon, margin, student_patience, student_epochs, device, base_save_dir=base_save_dir, 
                                        student_scheduler=student_scheduler, critic_scheduler=critic_scheduler, plot=False)
        student_model_states_best[lambda_factor] = student_best_model_state
    
    
    # Save the collective best model states to a file
    student_collective_save_path = os.path.join(base_save_dir, 'student_model_states_best.pth')
    torch.save(student_model_states_best, student_collective_save_path)

    # Evaluate models on different lambdas
    lambda_results = {}

    # Loop through each lambda value
    for lmda_teacher in teacher_lambda_factor_list:
        for lmda_student in student_lambda_factor_list:
    
            # Define the path to the saved model file for this lambda_factor
            lambda_dir = os.path.join(base_save_dir, f'TEACHER_lambda_{lmda_teacher}')
            teacher_path = os.path.join(lambda_dir, f'TEACHER_best_model_lambda_{lmda_teacher}.pth')
            
            # Initialize the EfficientNet model without pre-trained weights
            teacher_model = models.efficientnet_b3(pretrained=False)
            
            # Adjust the classifier layer to match the number of classes
            num_ftrs = teacher.classifier[1].in_features
            teacher_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
            # Load the model state
            teacher_best_model_state = torch.load(teacher_path)
            teacher_model.load_state_dict(teacher_best_model_state['teacher_state_dict'])
            teacher_model = teacher_model.to(device)
            
            # Define the path to the saved model file for this lambda_factor
            lambda_dir = os.path.join(base_save_dir, f'STUDENT_lambda_{lmda_student}')
            student_path = os.path.join(lambda_dir, f'STUDENT_best_model_lambda_{lmda_student}.pth')
            
            # Initialize the EfficientNet model without pre-trained weights
            student_model = models.efficientnet_b0(pretrained=False)
            
            # Adjust the classifier layer to match the number of classes
            num_ftrs = student_model.classifier[1].in_features
            student_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
            # Load the model state
            student_best_model_state = torch.load(student_path)
            student_model.load_state_dict(student_best_model_state['student_state_dict'])
            student_model = student_model.to(device)
            
            # Compute performance metrics
            performance_metrics = compare_performance_metrics(teacher_model, student_model, testloader)
        
            # Compute model sizes
            teacher_params, student_params = compare_model_size(teacher_model, student_model)
        
            # Construct a unique key for the current combination of lambda values
            lambda_key = (lmda_teacher, lmda_student)
    
            # Update results for the current lambda value
            if lambda_key in lambda_results:
                lambda_results[lambda_key].update({
                    'performance_metrics': performance_metrics,
                    'teacher_params': teacher_params,
                    'student_params': student_params
                })
            else:
                lambda_results[lambda_key] = {
                    'performance_metrics': performance_metrics,
                    'teacher_params': teacher_params,
                    'student_params': student_params
                }

    # Generate accuracy report
    evaluate_accuracy(lambda_results)
    # Generate precision report
    evaluate_precision(lambda_results)
    # Generate recall report
    evaluate_recall(lambda_results)
    # Generate f1 report
    evaluate_f1(lambda_results)
    # Generate disparity report
    evaluate_disparity(lambda_results)
    # Generate model size reduction report
    evaluate_model_size(lambda_results)
    # Generate inference report
    evaluate_inference(lambda_results)
    