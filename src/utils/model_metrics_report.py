import numpy as np
import matplotlib.pyplot as plt


def evaluate_accuracy(lambda_results):
    
    # Initialize lists to store accuracies
    teacher_accuracies = []
    student_accuracies = []
    lambda_pairs = list(lambda_results.keys())
    
    # Iterate over the keys in lambda_results
    for key in lambda_pairs:
        # Check if the key is a tuple (indicating a lambda pair)
        if isinstance(key, tuple) and len(key) == 2:
            lmda_teacher, lmda_student = key
        else:
            # If the key is not a tuple, skip this iteration
            continue
    
        # Access the performance metrics for each pair
        teacher_accuracy = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['accuracy'][0]
        student_accuracy = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['accuracy'][1]
    
        # Append accuracies to the lists
        teacher_accuracies.append((lmda_teacher, teacher_accuracy))
        student_accuracies.append((lmda_student, student_accuracy))
    
    # To plot, you might need to separate the lambda values and accuracies
    teacher_lambdas, teacher_acc = zip(*teacher_accuracies)
    student_lambdas, student_acc = zip(*student_accuracies)
    
    # Plotting only with markers and no lines
    plt.scatter(teacher_lambdas, teacher_acc, label='Teacher Accuracy', marker='o')
    plt.scatter(student_lambdas, student_acc, label='Student Accuracy', marker='o')
    
    # Adding labels and title
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Lambdas')
    plt.legend()
    
    # Show plot
    plt.show()

def evaluate_precision(lambda_results):
    # Initialize lists to store precisions
    teacher_precisions = []
    student_precisions = []
    lambda_pairs = list(lambda_results.keys())
    
    # Iterate over the keys in lambda_results
    for key in lambda_pairs:
        # Check if the key is a tuple (indicating a lambda pair)
        if isinstance(key, tuple) and len(key) == 2:
            lmda_teacher, lmda_student = key
            # Access the precision metrics for each pair
            teacher_precision = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['precision'][0]
            student_precision = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['precision'][1]
        else:
            # If the key is not a tuple, skip this iteration
            continue
    
        # Append precisions to the lists along with lambda values
        teacher_precisions.append((lmda_teacher, teacher_precision))
        student_precisions.append((lmda_student, student_precision))
    
    # Extracting lambda values and precisions
    teacher_lambdas, teacher_prec = zip(*teacher_precisions)
    student_lambdas, student_prec = zip(*student_precisions)
    
    # Creating a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(teacher_lambdas, teacher_prec, label='Teacher Precision', marker='o')
    plt.scatter(student_lambdas, student_prec, label='Student Precision', marker='o')

    # Adding labels and title
    plt.xlabel('Lambda')
    plt.ylabel('Precision')
    plt.title('Precision Comparison Across Lambdas')
    plt.legend()
    
    # Displaying the plot
    plt.show()

def evaluate_recall(lambda_results):
    # Initialize lists to store recalls
    teacher_recalls = []
    student_recalls = []
    lambda_pairs = list(lambda_results.keys())
    
    # Iterate over the keys in lambda_results
    for key in lambda_pairs:
        # Check if the key is a tuple (indicating a lambda pair)
        if isinstance(key, tuple) and len(key) == 2:
            lmda_teacher, lmda_student = key
            # Access the recall metrics for each pair
            teacher_recall = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['recall'][0]
            student_recall = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['recall'][1]
        else:
            # If the key is not a tuple, skip this iteration
            continue
    
        # Append recalls to the lists along with lambda values
        teacher_recalls.append((lmda_teacher, teacher_recall))
        student_recalls.append((lmda_student, student_recall))
    
    # Extracting lambda values and recalls
    teacher_lambdas, teacher_rec = zip(*teacher_recalls)
    student_lambdas, student_rec = zip(*student_recalls)
    
    # Creating a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(teacher_lambdas, teacher_rec, label='Teacher Recall', marker='o')
    plt.scatter(student_lambdas, student_rec, label='Student Recall', marker='o')
    
    # Adding labels and title
    plt.xlabel('Lambda')
    plt.ylabel('Recall')
    plt.title('Recall Comparison Across Lambdas')
    plt.legend()
    
    # Displaying the plot
    plt.show()


def evaluate_f1(lambda_results):
    # Initialize lists to store F1 scores
    teacher_f1s = []
    student_f1s = []
    lambda_pairs = list(lambda_results.keys())
    
    # Iterate over the keys in lambda_results
    for key in lambda_pairs:
        # Check if the key is a tuple (indicating a lambda pair)
        if isinstance(key, tuple) and len(key) == 2:
            lmda_teacher, lmda_student = key
            # Access the F1 scores for each pair
            teacher_f1 = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['f1'][0]
            student_f1 = lambda_results[(lmda_teacher, lmda_student)]['performance_metrics']['metrics']['f1'][1]
        else:
            # If the key is not a tuple, skip this iteration
            continue
    
        # Append F1 scores to the lists along with lambda values
        teacher_f1s.append((lmda_teacher, teacher_f1))
        student_f1s.append((lmda_student, student_f1))
    
    # Extracting lambda values and F1 scores
    teacher_lambdas, teacher_f1_scores = zip(*teacher_f1s)
    student_lambdas, student_f1_scores = zip(*student_f1s)
    
    # Creating a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(teacher_lambdas, teacher_f1_scores, label='Teacher F1 Score', marker='o')
    plt.scatter(student_lambdas, student_f1_scores, label='Student F1 Score', marker='o')
    
    # Adding labels and title
    plt.xlabel('Lambda')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison Across Lambdas')
    plt.legend()
    
    # Displaying the plot
    plt.show()

def evaluate_model_size(lambda_results):
    # Initialize lists to store model sizes
    teacher_sizes = []
    student_sizes = []
    lambda_pairs = list(lambda_results.keys())
    
    # Iterate over the keys in lambda_results
    for key in lambda_pairs:
        # Check if the key is a tuple (indicating a lambda pair)
        if isinstance(key, tuple) and len(key) == 2:
            lmda_teacher, lmda_student = key
            # Access the model sizes for each pair
            teacher_size = lambda_results[(lmda_teacher, lmda_student)]['teacher_params'] / 1e6  # Convert to millions
            student_size = lambda_results[(lmda_teacher, lmda_student)]['student_params'] / 1e6
        else:
            # If the key is not a tuple, skip this iteration
            continue
    
        # Append model sizes to the lists along with lambda values
        teacher_sizes.append((lmda_teacher, teacher_size))
        student_sizes.append((lmda_student, student_size))
    
    # Extracting lambda values and model sizes
    teacher_lambdas, teacher_model_sizes = zip(*teacher_sizes)
    student_lambdas, student_model_sizes = zip(*student_sizes)
    
    # Creating a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(teacher_lambdas, teacher_model_sizes, label='Teacher Model Size', marker='o')
    plt.scatter(student_lambdas, student_model_sizes, label='Student Model Size', marker='o')
    
    # Adding labels and title
    plt.xlabel('Lambda')
    plt.ylabel('Model Size (Millions of Parameters)')
    plt.title('Model Size Comparison Across Lambdas')
    plt.legend()
    
    # Displaying the plot
    plt.show()

def evaluate_inference(lambda_results):
    # Initialize dictionaries to store inference times for each lambda value
    teacher_times = {}
    student_times = {}
    
    # Loop through each lambda value
    for lmda_teacher in teacher_lambda_factor_list:
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
    
        teacher_time, _ = compare_inference_time(teacher_model, None, testloader)
        teacher_times[lmda_teacher] = teacher_time  # Store the inference time for the teacher model
    
    for lmda_student in student_lambda_factor_list:
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
    
        _, student_time = compare_inference_time(None, student_model, testloader)
        student_times[lmda_student] = student_time  # Store the inference time for the student model
    
    # Extracting lambda values and inference times
    teacher_lambdas, teacher_inference_times = zip(*teacher_times.items())
    student_lambdas, student_inference_times = zip(*student_times.items())
    
    # Creating a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(teacher_lambdas, teacher_inference_times, label='Teacher Inference Time', marker='o')
    plt.scatter(student_lambdas, student_inference_times, label='Student Inference Time', marker='o')
    
    # Adding labels and title
    plt.xlabel('Lambda')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time Comparison Across Lambdas')
    plt.legend()
    
    # Displaying the plot
    plt.show()

def evaluate_disparity(lambda_results):
    teacher_disparities = []
    student_disparities = []
    
    # Loop through each lambda_factor in the teacher_lambda_factor_list and extract the disparity values for the teacher
    for lambda_factor in teacher_lambda_factor_list:
        # Load teacher model
        teacher_path = os.path.join(base_save_dir, f'TEACHER_lambda_{lambda_factor}', f'TEACHER_best_model_lambda_{lambda_factor}.pth')
        teacher_best_model_state = torch.load(teacher_path)
        teacher_disparities.append((lambda_factor, teacher_best_model_state['best_val_mean_abs_disparity']))
    
    # Loop through each lambda_factor in the student_lambda_factor_list and extract the disparity values for the student
    for lambda_factor in student_lambda_factor_list:
        # Load student model
        student_path = os.path.join(base_save_dir, f'STUDENT_lambda_{lambda_factor}', f'STUDENT_best_model_lambda_{lambda_factor}.pth')
        student_best_model_state = torch.load(student_path)
        student_disparities.append((lambda_factor, student_best_model_state['best_val_mean_abs_disparity']))
    
    # Unpack the lambda factors and disparities for plotting
    teacher_lambdas, teacher_disp_values = zip(*teacher_disparities)
    student_lambdas, student_disp_values = zip(*student_disparities)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(teacher_lambdas, teacher_disp_values, label='Teacher', color='blue')
    plt.scatter(student_lambdas, student_disp_values, label='Student', color='red')
    plt.xlabel('Lambda Factor')
    plt.ylabel('Best Val Mean Abs Disparity')
    plt.title('Best Validation Mean Absolute Disparity vs Lambda Factor')
    plt.legend()
    plt.show()



    