## Training script

def train(student, teacher, T_EMB, train_dataloader, optimizer, criterion, kd_loss, nd_loss, epoch, batch_size, temperature, adv, adv_criterion, optimizer_adv, lmda):

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # First train adversary in this epoch
    train_adversary(adv, student, optimizer_adv, train_dataloader, adv_criterion, 1)

    # test T_EMB
    T_EMB = T_EMB

    # Model on train mode
    student.train()
    teacher.eval()
    running_loss = 0.0 
    epoch_loss = 0.0  
    num_batches = 0 
    confusion_male = np.zeros((num_classes, num_classes))
    confusion_female = np.zeros((num_classes, num_classes))
 
    step_per_epoch = len(train_dataloader)

    for step, data in enumerate(tqdm(train_dataloader)):
        
        start = time.time()
        s_FEATS = []
        features = {}

        inputs = data['img'].to(device)
        labels = data['label'].to(device)
        targets = data['target'].to(device)

        curr_batch_size = len(inputs)

        # register hook for feature embeddings
        student.avgpool.register_forward_hook(get_features('feats'))
        
        # compute output
        optimizer.zero_grad()
        s_logits = student(inputs)

        s_FEATS.append(features['feats'].cpu().numpy())
        s_emb = np.concatenate(s_FEATS)
        # print(f'before reshaping s_emb: {s_emb.shape}')
        # reshape embedding features to flatten 
        s_emb = s_emb.reshape((curr_batch_size, s_emb.shape[1]))
        s_emb = torch.from_numpy(s_emb)
        s_emb = s_emb.to(device)

        # fix embedding output on student model
        s_emb_size = 1280
        t_emb_size = 1536
        
        emb_inflate = nn.Sequential(
            nn.BatchNorm1d(s_emb_size),
            nn.Dropout(0.5),
            nn.Linear(s_emb_size, t_emb_size)
            )
        
        ## clean model
        for m in student.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        
        emb_inflate.to(device)

        s_emb = emb_inflate(s_emb)
        
        with torch.no_grad():
                        
            ####
            
            t_FEATS = []
            features = {}
    
            # register hook for feature embeddings
            teacher.avgpool.register_forward_hook(get_features('feats'))
            
            # compute output
            t_logits = teacher(inputs)
    
            t_FEATS.append(features['feats'].cpu().numpy())
            t_emb = np.concatenate(t_FEATS)
            # reshape embedding features to flatten 
            t_emb = t_emb.reshape((curr_batch_size, t_emb.shape[1]))


        ## save s_emb and t_emb as torch tensors 
        t_emb = torch.from_numpy(t_emb)

        # s_emb = s_emb.to(device)
        t_emb = t_emb.to(device)


        # print(s_emb.size() == s_emb.size())
        # print(s_emb.size())
        # print(s_emb.size())
        
        ###

        # detach student_outputs to avoid exploding gradients by passing same inputs (with gradience) into two different models. 
        studentached = s_logits.detach()
        # One-hot encode labels and concatenate with student's predictions
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
        concatenated_output = torch.cat((studentached, one_hot_labels), dim=1)

        # Run the adversarial model on concatenated true labels, and predicted labels
        with torch.no_grad():
            adversary_output = adv(concatenated_output)

        
         # Calc adversary loss, which is an MSE loss, because this is a regression output.       
        adversary_loss = adv_criterion(adversary_output, targets)
        # cls loss
        cls_loss = criterion(s_logits, labels) * cls_loss_factor
        # KD loss
        div_loss = kd_loss(s_out = s_logits, t_out = t_logits) * min(1.0, epoch/warm_up)
        # ND loss
        norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

        if lmda != 0:
            loss = cls_loss + div_loss + norm_dir_loss + (cls_loss + div_loss + norm_dir_loss)/adversary_loss - lmda * adversary_loss
        else:
            loss = cls_loss + div_loss + norm_dir_loss
        
        # measure accuracy and record loss
        batch_size = inputs.size(0)
        _, pred = s_logits.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        Cls_loss.update(cls_loss.item(), batch_size)
        Div_loss.update(div_loss.item(), batch_size)
        Norm_Dir_loss.update(norm_dir_loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        num_batches += 1

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - nd_loss: {:.3f} - kd_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), norm_dir_loss.item(), div_loss.item(), cls_loss.item(), train_loss.val, 1-train_error.val)

        print(s1+s2, end='', flush=True)

    epoch_loss /= num_batches
    # student_epoch_losses.append(epoch_loss)


    print()
    return Norm_Dir_loss.avg, Div_loss.avg, Cls_loss.avg, train_loss.avg, train_error.avg


def test(student, teacher, test_dataloader, criterion, adv, lmda, kd_loss, nd_loss, epoch, optimizer):

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    test_loss = AverageMeter()
    test_error = AverageMeter()

    # Model on eval mode
    student.eval()
    teacher.eval()
    total_correct = 0
    total_samples = 0
    total_val_loss = 0.0
    num_batches = 0


    with torch.no_grad():
        for step, data in enumerate(tqdm(test_dataloader)):

            inputs = data['img'].to(device)
            labels = data['label'].to(device)
            targets = data['target'].to(device)

            # compute logits
            s_logits = student(inputs)
            t_logits = teacher(inputs)

            # get feature embeddings

            #########

            curr_batch_size = len(inputs)

            # register hook for feature embeddings
            student.avgpool.register_forward_hook(get_features('feats'))
            
            # compute output
            optimizer.zero_grad()
            s_logits = student(inputs)
    
            s_FEATS.append(features['feats'].cpu().numpy())
            s_emb = np.concatenate(s_FEATS)
            # print(f'before reshaping s_emb: {s_emb.shape}')
            # reshape embedding features to flatten 
            s_emb = s_emb.reshape((curr_batch_size, s_emb.shape[1]))
            s_emb = torch.from_numpy(s_emb)
            s_emb = s_emb.to(device)
    
            # fix embedding output on student model
            s_emb_size = 1280
            t_emb_size = 1536
            
            emb_inflate = nn.Sequential(
                nn.BatchNorm1d(s_emb_size),
                nn.Dropout(0.5),
                nn.Linear(s_emb_size, t_emb_size)
                )

            ## clean model
            for m in student.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
    
            
            emb_inflate.to(device)
    
            s_emb = emb_inflate(s_emb)


            t_FEATS = []
            features = {}
    
            # register hook for feature embeddings
            teacher.avgpool.register_forward_hook(get_features('feats'))
            
            # compute output
            t_logits = teacher(inputs)
    
            t_FEATS.append(features['feats'].cpu().numpy())
            t_emb = np.concatenate(t_FEATS)
            # reshape embedding features to flatten 
            t_emb = t_emb.reshape((curr_batch_size, t_emb.shape[1]))


            ## save s_emb and t_emb as torch tensors 
            t_emb = torch.from_numpy(t_emb)
    
            # s_emb = s_emb.to(device)
            t_emb = t_emb.to(device)
        
            ##########


            
            studentached = s_logits.detach()   
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
            concatenated_output = torch.cat((studentached, one_hot_labels), dim=1)

            adversary_output = adv(concatenated_output)
            adversary_loss = adv_criterion(adversary_output, targets)
            # cls loss
            cls_loss = criterion(s_logits, labels) * cls_loss_factor
            # KD loss
            div_loss = kd_loss(s_out = s_logits, t_out = t_logits) * min(1.0, epoch/warm_up)
            # ND loss
            norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

            if lmda != 0:
                test_loss = cls_loss + div_loss + norm_dir_loss + (cls_loss + div_loss + norm_dir_loss)/adversary_loss - lmda * adversary_loss
            else:
                test_loss = cls_loss + div_loss + norm_dir_loss
            

            # measure accuracy and record loss
            batch_size = inputs.size(0)
            _, pred = s_logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)

            
            total_val_loss += val_loss.item()

            # Compute the validation accuracy
            _, predicted = torch.max(s_logits, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            num_batches += 1
            recall_diff = evaluate_model_with_gender_multiclass(predicted, labels, targets, num_classes=num_classes)
            confusion_male += recall_diff[1]
            confusion_female += recall_diff[2]

        ##
        total_val_loss /= num_batches
        confusion_male /= num_batches
        confusion_female /= num_batches

        epoch_disparity = calculate_recall_multiclass(confusion_male) - calculate_recall_multiclass(confusion_female)
        val_losses.append(total_val_loss)
        non_zero_abs_values = np.abs(epoch_disparity[epoch_disparity != 0])
        mean_non_zero_abs_disparity = np.mean(non_zero_abs_values)
        val_disparities.append(mean_non_zero_abs_disparity)
        accuracy = total_correct / total_samples
        val_accuracies.append(accuracy)
        print(f'*****Epoch {epoch + 1}/{epochs}*****\n' 
        f'*****Train Loss: {epoch_loss: .6f} Val Loss: {total_val_loss: .6f}*****\n'
        f'*****Validation Accuracy: {accuracy * 100:.2f}%*****\n'
        f'*****Total Avg Disparity: {mean_non_zero_abs_disparity}*****\n')
        class_recall_mapping = {class_name: epoch_disparity[int(class_label)] for class_label, class_name in class_idx.items()}
        
        # Print disparities by class label
        for class_label, recall_diff in class_recall_mapping.items():
            print(f"Class {class_label}: Recall Difference = {recall_diff}")


    # Check for early stopping
    if abs(total_val_loss) < abs(best_total_val_loss):
        best_total_val_loss = total_val_loss
        patience_counter = 0
        best_epoch_mean_abs_disparity = mean_non_zero_abs_disparity
        torch.save(student.state_dict(), f'weights/student_model_weights_efficientnetb0_wider_checkpoint_lmda_{lmda}.pth')
        torch.save(student, f'weights/student_model_efficientnetb0_wider_checkpoint_lmda_{lmda}.pth')
    else:
        patience_counter += 1 

    file_path = os.path.join(output_dir, f'student_validation_{lmda}.txt')
    
    # Append data to the text file
    with open(file_path, 'a') as file:
        file.write(f'********Epoch: {epochs}***********')
        
        file.write("Student Val Accuracies:\n")
        for accuracy in val_accuracies:
            file.write(f"{accuracy}\n")
    
        file.write("\nStudent Val Disparities:\n")
        for disparity in val_disparities:
            file.write(f"{disparity}\n")

        for class_label, recall_diff in class_recall_mapping.items():
            file.write(f"Class {class_label}: Recall Difference = {recall_diff}\n")

    if patience_counter >= patience:
        print('Early stopping')
        return test_loss.avg, test_error.avg
    
    print(f"Data has been appended to {file_path}")

    return test_loss.avg, test_error.avg


def epoch_loop(student, teacher, train_loader, test_loader, num_class, T_EMB, save_dir, batch_size, logger, lr, lmda):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # student = nn.DataParallel(student, device_ids=args.gpus)
    # student = nn.DataParallel(student)
    student = student
    student.to(device)
    # teacher = nn.DataParallel(teacher, device_ids=args.gpus)
    # teacher = nn.DataParallel(teacher)
    teacher = teacher
    teacher.to(device)

    best_val_accuracy = 0
    best_total_val_loss = float('inf')
    best_epoch_accuracy = 0.0
    best_epoch_disparity = 0.0
    patience_counter = 0 
    student_epoch_losses = []
    val_losses = []
    val_disparities = []
    val_accuracies = []

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    kd_loss = KDLoss(kl_loss_factor=kd_loss_factor, T=t).to(device)
    nd_loss = DirectNormLoss(num_class=num_class, nd_loss_factor=nd_loss_factor).to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    # weights
    save_dir = Path(save_dir)
    weights = save_dir / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    last = weights / 'last'
    best = weights / 'best'

    # acc,loss
    acc_loss = save_dir / 'acc_loss'
    acc_loss.mkdir(parents=True, exist_ok=True)

    train_acc_savepath = acc_loss / 'train_acc.npy'
    train_loss_savepath = acc_loss / 'train_loss.npy'
    val_acc_savepath = acc_loss / 'val_acc.npy'
    val_loss_savepath = acc_loss / 'val_loss.npy'

    # tensorboard
    logdir = save_dir / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(logdir, flush_secs=120)


    start_epoch = 0
    best_error = 0
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    logger = logger

    # Train model
    best_error = 1
    for epoch in range(start_epoch, epochs):
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        print("Epoch {}/{}".format(epoch + 1, epochs))
        norm_dir_loss, div_loss, cls_loss, train_epoch_loss, train_error = train(student=student,
                                                                                 teacher=teacher,
                                                                                 T_EMB=T_EMB,
                                                                                 train_dataloader=train_loader,
                                                                                 optimizer=optimizer,
                                                                                 criterion=criterion,
                                                                                 kd_loss=kd_loss,
                                                                                 nd_loss=nd_loss,
                                                                                 epoch=epoch,
                                                                                 batch_size = batch_size,
                                                                                 temperature = temperature, 
                                                                                 adv = adv, 
                                                                                 adv_criterion = adv_criterion, 
                                                                                 optimizer_adv=optimizer_adv,
                                                                                 lmda = lmda)
        test_epoch_loss, test_error = test(student=student,
                                           teacher=teacher,
                                           test_dataloader=test_loader,
                                           criterion=criterion,
                                           adv = adv,
                                           lmda = lmda,
                                           kd_loss = kd_loss,
                                           nd_loss = nd_loss,
                                           epoch=epoch,
                                           optimizer=optimizer)

        s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.5f}".format(
            train_epoch_loss, 1-train_error, test_epoch_loss, 1-test_error, optimizer.param_groups[0]['lr'])
        logger.info(colorstr('green', s))

        # save acc,loss
        train_loss.append(train_epoch_loss)
        train_acc.append(1-train_error)
        test_loss.append(test_epoch_loss)
        test_acc.append(1-test_error)

        # save model
        is_best = test_error < best_error
        best_error = min(best_error, test_error)
        state = {
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_error': best_error,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }

        last_path = last / 'epoch_{}_loss_{:.3f}_acc_{:.3f}'.format(
            epoch + 1, test_epoch_loss, 1-test_error)
        best_path = best / 'epoch_{}_acc_{:.3f}'.format(
                epoch + 1, 1-best_error)

        Save_Checkpoint(state, last, last_path, best, best_path, is_best)

        # tensorboard
        if epoch == 1:
            # images, labels = next(iter(train_loader))
            data = next(iter(train_loader))
            images = data['img'].to(device)
            labels = data['label'].to(device)

            img_grid = torchvision.utils.make_grid(images)
            summary_writer.add_image('Image', img_grid)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
        summary_writer.add_scalar('train_error', train_error, epoch)
        summary_writer.add_scalar('val_loss', test_epoch_loss, epoch)
        summary_writer.add_scalar('val_error', test_error, epoch)

        summary_writer.add_scalar('nd_loss', norm_dir_loss, epoch)
        summary_writer.add_scalar('kd_loss', div_loss, epoch)
        summary_writer.add_scalar('cls_loss', cls_loss, epoch)

    summary_writer.close()
    import os
    if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
        np.save(train_acc_savepath, train_acc)
        np.save(train_loss_savepath, train_loss)
        np.save(val_acc_savepath, test_acc)
        np.save(val_loss_savepath, test_loss)

    plot_loss_curve(val_losses)

    return best_epoch_mean_abs_disparity