## Training script

def train(model, teacher, T_EMB, train_dataloader, optimizer, criterion, kd_loss, nd_loss, epoch, batch_size):

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    train_loss = AverageMeter()
    train_error = AverageMeter()

    Cls_loss = AverageMeter()
    Div_loss = AverageMeter()
    Norm_Dir_loss = AverageMeter()

    # test T_EMB
    T_EMB = T_EMB

    # set up hooks
    # model.avgpool.register_forward_hook(get_features('feats'))
    # teacher.avgpool.register_forward_hook(get_features('feats'))

    # Model on train mode
    model.train()
    teacher.eval()

    # # fix embedding output on student model
    # s_emb_size = 1280
    # t_emb_size = 1536
    # model.fc1 = nn.Sequential(
    #     nn.BatchNorm1d(s_emb_size),
    #     nn.Dropout(0.5),
    #     nn.Linear(s_emb_size, t_emb_size)
    #     )
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm1d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()
    #     elif isinstance(m, nn.Linear):
    #         m.bias.data.zero_()

    # ##
    # model.fc1.to(device)
    

    
    step_per_epoch = len(train_dataloader)

    for step, (images, labels) in enumerate(train_dataloader):
        
        start = time.time()
        s_FEATS = []
        features = {}
        curr_batch_size = len(images)
        images, labels = images.cuda(), labels.cuda()

        # compute output
        # emb_fea, logits = model(images, embed=True)
        model.avgpool.register_forward_hook(get_features('feats'))

        s_logits = model(images)

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
        # # clean_model
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
        ##
        emb_inflate.to(device)

        s_emb = emb_inflate(s_emb)
        # model.fc1.to(device)
        
        # s_emb = model.fc1(s_emb)

        with torch.no_grad():
                        
            ####
            
            t_FEATS = []
            features = {}
    
            # compute output
            # emb_fea, logits = model(images, embed=True)
            teacher.avgpool.register_forward_hook(get_features('feats'))
            
            t_logits = teacher(images)
    
            t_FEATS.append(features['feats'].cpu().numpy())
            t_emb = np.concatenate(t_FEATS)
            # reshape embedding features to flatten 
            t_emb = t_emb.reshape((curr_batch_size, t_emb.shape[1]))


        ## save s_emb and t_emb as torch tensors 
        # s_emb = torch.from_numpy(s_emb)
        t_emb = torch.from_numpy(t_emb)

        # s_emb = s_emb.to(device)
        t_emb = t_emb.to(device)


        # print(s_emb.size() == s_emb.size())
        # print(s_emb.size())
        # print(s_emb.size())
        
        ###

        # cls loss
        cls_loss = criterion(s_logits, labels) * cls_loss_factor
        # KD loss
        div_loss = kd_loss(s_out = s_logits, t_out = t_logits) * min(1.0, epoch/warm_up)
        # ND loss
        norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

        loss = cls_loss + div_loss + norm_dir_loss
        # measure accuracy and record loss
        batch_size = images.size(0)
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

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - nd_loss: {:.3f} - kd_loss: {:.3f} - cls_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time() - start), norm_dir_loss.item(), div_loss.item(), cls_loss.item(), train_loss.val, 1-train_error.val)

        print(s1+s2, end='', flush=True)

    print()
    return Norm_Dir_loss.avg, Div_loss.avg, Cls_loss.avg, train_loss.avg, train_error.avg


def test(model, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()

        total_val_loss = 0.0

    # Model on eval mode
    model.eval()


    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            # compute logits
            logits = model(images)

            loss = criterion(logits, labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            _, pred = logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)

    return test_loss.avg, test_error.avg


def epoch_loop(model, teacher, train_loader, test_loader, num_class, T_EMB, save_dir, batch_size, logger):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = nn.DataParallel(model, device_ids=args.gpus)
    # model = nn.DataParallel(model)
    model = model
    model.to(device)
    # teacher = nn.DataParallel(teacher, device_ids=args.gpus)
    # teacher = nn.DataParallel(teacher)
    teacher = teacher
    teacher.to(device)

    # model.avgpool.register_forward_hook(get_features('s_feats'))
    # teacher.avgpool.register_forward_hook(get_features('t_feats'))

    # loss
    criterion = nn.CrossEntropyLoss().to(device)
    kd_loss = KDLoss(kl_loss_factor=kd_loss_factor, T=t).to(device)
    nd_loss = DirectNormLoss(num_class=num_class, nd_loss_factor=nd_loss_factor).to(device)
    # optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

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
        norm_dir_loss, div_loss, cls_loss, train_epoch_loss, train_error = train(model=model,
                                                                                 teacher=teacher,
                                                                                 T_EMB=T_EMB,
                                                                                 train_dataloader=train_loader,
                                                                                 optimizer=optimizer,
                                                                                 criterion=criterion,
                                                                                 kd_loss=kd_loss,
                                                                                 nd_loss=nd_loss,
                                                                                 epoch=epoch,
                                                                                 batch_size = batch_size)
        test_epoch_loss, test_error = test(model=model,
                                           test_dataloader=test_loader,
                                           criterion=criterion)

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
                'model_state_dict': model.state_dict(),
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
            images, labels = next(iter(train_loader))

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
