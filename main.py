from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from torch.utils.tensorboard import SummaryWriter
import copy

# ----------------------------- Helper Classes -----------------------------
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)

class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

def create_model():
    model = ResNet18(num_classes=args.num_class)
    return model.cuda()

# ----------------------------- JoSRC Functions -----------------------------
def update_mean_teacher(model, mean_teacher, alpha=0.99):
    """Update mean teacher model with exponential moving average"""
    for mean_param, param in zip(mean_teacher.parameters(), model.parameters()):
        mean_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def js_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions"""
    p = p + 1e-10  
    q = q + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(p.log(), m, reduction='none').sum(dim=1) + \
           0.5 * F.kl_div(q.log(), m, reduction='none').sum(dim=1)

def label_smoothing(labels, num_classes, epsilon=0.1):
    """Apply label smoothing regularization"""
    smoothed = torch.full((labels.size(0), num_classes), epsilon / (num_classes - 1)).cuda()
    smoothed.scatter_(1, labels.unsqueeze(1), 1.0 - epsilon)
    return smoothed

def eval_train_josrc(net1, net2, mean_teacher1, mean_teacher2, all_loss, epoch):
    """
    JoSRC sample selection: uses JS divergence for clean sample detection
    Returns same format as GMM: (predictions, probabilities, pseudo_labels, all_loss)
    """
    net1.eval()
    net2.eval()
    mean_teacher1.eval()
    mean_teacher2.eval()
    
    # Get dataset size dynamically
    num_samples = len(eval_loader.dataset)
    losses = torch.zeros(num_samples)
    p_clean = torch.zeros(num_samples)
    pseudo_labels = torch.zeros(num_samples, args.num_class)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size(0)
            
            # Get predictions from both networks
            outputs1_net1 = net1(inputs)
            outputs1_net2 = net2(inputs)
            
            # Average predictions from both networks for robustness
            outputs_avg = (outputs1_net1 + outputs1_net2) / 2
            probs = F.softmax(outputs_avg, dim=1)
            
            # Smooth the label distribution
            targets_smooth = label_smoothing(targets, args.num_class, epsilon=0.1)
            
            # Calculate JS divergence between predictions and given labels
            js_div = js_divergence(probs, targets_smooth)
            
            # P_clean: likelihood of being clean (1 - JS divergence)
            # Clamp to ensure it's in valid range
            p_clean_batch = 1.0 - torch.clamp(js_div / np.log(2), 0, 1)
            
            # Get mean teacher predictions for pseudo-labeling
            outputs_mt1 = mean_teacher1(inputs)
            outputs_mt2 = mean_teacher2(inputs)
            pseudo_probs = F.softmax((outputs_mt1 + outputs_mt2) / 2, dim=1)
            
            # Store results
            for b in range(batch_size):
                idx = index[b].item()
                p_clean[idx] = p_clean_batch[b].item()
                pseudo_labels[idx] = pseudo_probs[b].cpu()
                
                # Also compute standard CE loss for reference
                loss = F.cross_entropy(outputs_avg[b:b+1], targets[b:b+1])
                losses[idx] = loss.item()
    
    # Normalize losses for tracking 
    if losses.max() > losses.min():
        losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)
    
    # Dynamic threshold adjustment 
    if epoch < args.warmup_epochs:
        tau_clean = (epoch / args.warmup_epochs) * args.tau_clean
    else:
        # Gradually increase threshold
        progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
        tau_clean = args.tau_clean + progress * (0.95 - args.tau_clean)
    
    # Select clean samples 
    pred_clean = (p_clean > tau_clean)
    
    # Convert to numpy arrays to match GMM output format
    return pred_clean.cpu().numpy(), p_clean.cpu().numpy(), pseudo_labels.cpu().numpy(), all_loss

# ----------------------------- Training Functions -----------------------------
def train(epoch, net, net2, mean_teacher, mean_teacher2, optimizer, 
          labeled_trainloader, unlabeled_trainloader, pseudo_labels_dict, writer=None):
    """
    Enhanced training with JoSRC pseudo-labeling and DivideMix semi-supervised learning
    """
    net.train()
    net2.eval()  # Peer network for co-training
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    
    consistency_loss_meter = 0

    for batch_idx, batch_data in enumerate(labeled_trainloader):
        # Handle different batch formats 
        if len(batch_data) == 5:
            inputs_x, inputs_x2, labels_x, w_x, indices = batch_data
        else:
            inputs_x, inputs_x2, labels_x, w_x = batch_data
            indices = None
            
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        
        # Get pseudo labels for this batch (for samples with low w_x, these are noisy)
        pseudo_targets = torch.zeros(batch_size, args.num_class)
        if indices is not None and len(pseudo_labels_dict) > 0:
            for i, idx in enumerate(indices):
                idx_val = idx.item() if torch.is_tensor(idx) else idx
                if idx_val in pseudo_labels_dict:
                    pseudo_targets[i] = pseudo_labels_dict[idx_val]
                else:
                    # If not in dict, use one-hot of given label
                    pseudo_targets[i].scatter_(0, labels_x[i].view(-1), 1)
        else:
            # No pseudo labels available, use one-hot encoding
            for i in range(batch_size):
                pseudo_targets[i].scatter_(0, labels_x[i].view(-1), 1)
        
        # Original one-hot labels
        labels_x_onehot = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).float()

        inputs_x, inputs_x2 = inputs_x.cuda(), inputs_x2.cuda()
        labels_x_onehot, w_x = labels_x_onehot.cuda(), w_x.cuda()
        pseudo_targets = pseudo_targets.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # Unlabeled co-guessing
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, 1) + torch.softmax(outputs_u12, 1) +
                  torch.softmax(outputs_u21, 1) + torch.softmax(outputs_u22, 1)) / 4
            ptu = pu ** (1 / args.T)
            targets_u = (ptu / ptu.sum(1, keepdim=True)).detach()

            # Labeled refinement with JoSRC pseudo labels
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)
            px = (torch.softmax(outputs_x, 1) + torch.softmax(outputs_x2, 1)) / 2
            
            # Use w_x to blend original labels with pseudo labels from mean teacher
            # High w_x (clean) -> use original labels
            # Low w_x (noisy) -> use pseudo labels
            px = w_x * labels_x_onehot + (1 - w_x) * px
            
            ptx = px ** (1 / args.T)
            targets_x = (ptx / ptx.sum(1, keepdim=True)).detach()

        # JoSRC Consistency Regularization: encourage agreement between two views
        if epoch >= warm_up and args.consistency_lambda > 0:
            outputs_x_view1 = net(inputs_x)
            outputs_x_view2 = net(inputs_x2)
            
            p1 = F.softmax(outputs_x_view1, dim=1)
            p2 = F.softmax(outputs_x_view2, dim=1)
            
            # Bidirectional KL divergence for consistency
            consistency_loss = F.kl_div(p1.log(), p2.detach(), reduction='batchmean') + \
                             F.kl_div(p2.log(), p1.detach(), reduction='batchmean')
            consistency_loss = consistency_loss / 2
            consistency_loss_meter = consistency_loss.item()
        else:
            consistency_loss = torch.tensor(0.0).cuda()
            consistency_loss_meter = 0.0

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], 0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], 0)
        idx = torch.randperm(all_inputs.size(0))
        mixed_input = l * all_inputs + (1 - l) * all_inputs[idx]
        mixed_target = l * all_targets + (1 - l) * all_targets[idx]

        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]

        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2],
                                 logits_u, mixed_target[batch_size*2:],
                                 epoch, warm_up)

        # Regularization
        prior = torch.ones(args.num_class).cuda() / args.num_class
        pred_mean = torch.softmax(logits, 1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # Total loss with JoSRC consistency
        loss = Lx + lamb * Lu + penalty
        if epoch >= warm_up:
            loss = loss + args.consistency_lambda * consistency_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update mean teacher
        update_mean_teacher(net, mean_teacher, alpha=args.ema_decay)

        if writer:
            step = epoch*num_iter + batch_idx
            writer.add_scalar('Train/Labeled_Loss', Lx.item(), step)
            writer.add_scalar('Train/Unlabeled_Loss', Lu.item(), step)
            writer.add_scalar('Train/Total_Loss', loss.item(), step)
            if epoch >= warm_up:
                writer.add_scalar('Train/Consistency_Loss', consistency_loss_meter, step)

        sys.stdout.write(f'\rEpoch [{epoch}/{args.num_epochs}] Iter [{batch_idx+1}/{num_iter}] '
                         f'Labeled: {Lx.item():.4f} Unlabeled: {Lu.item():.4f} '
                         f'Consist: {consistency_loss_meter:.4f}')
        sys.stdout.flush()

def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        loss.backward()
        optimizer.step()

        sys.stdout.write(f'\rWarmup Epoch [{epoch}/{warm_up}] Iter [{batch_idx+1}/{num_iter}] CE-Loss: {loss.item():.4f}')
        sys.stdout.flush()

def test(epoch, net1, net2, writer=None):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net1(inputs) + net2(inputs)
            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    acc = 100.*correct/total
    print(f'\nTest Epoch [{epoch}] Accuracy: {acc:.2f}%')
    test_log.write(f'Epoch {epoch} Accuracy: {acc:.2f}\n')
    test_log.flush()
    if writer:
        writer.add_scalar('Test/Accuracy', acc, epoch)

def eval_train(model, all_loss):    
    """Original GMM-based evaluation (kept as fallback)"""
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]         
    losses = (losses - losses.min()) / (losses.max() - losses.min())    
    all_loss.append(losses)

    if args.r == 0.9:
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:, gmm.means_.argmin()]         
    return prob, all_loss

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--alpha', default=4, type=float)
    parser.add_argument('--lambda_u', default=25, type=float)
    parser.add_argument('--p_threshold', default=0.5, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--r', default=0.5, type=float)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--data_path', default='./cifar-10', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    # JoSRC specific arguments
    parser.add_argument('--tau_clean', default=0.5, type=float,
                       help='Initial threshold for clean sample selection')
    parser.add_argument('--consistency_lambda', default=0.5, type=float,
                       help='Weight for consistency regularization')
    parser.add_argument('--ema_decay', default=0.99, type=float,
                       help='EMA decay rate for mean teacher')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                       help='Number of warmup epochs')
    
    args = parser.parse_args()

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Logs
    os.makedirs('./checkpoint', exist_ok=True)
    stats_log = open(f'./checkpoint/{args.dataset}_stats.txt', 'w')
    test_log = open(f'./checkpoint/{args.dataset}_acc.txt', 'w')

    # Warmup
    warm_up = args.warmup_epochs
    base_logdir = f'./runs/{args.dataset}_{args.r}_{args.noise_mode}_josrc'

    os.makedirs(base_logdir, exist_ok=True)
    run_name = f'epochs={args.num_epochs}_warmup={warm_up}_tau={args.tau_clean}'
    run_logdir = os.path.join(base_logdir, run_name)
    os.makedirs(run_logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_logdir)
    print(f"Logging this run to: {run_logdir}")

    # Loaders
    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,
                                         batch_size=args.batch_size, num_workers=5,
                                         root_dir=args.data_path, log=stats_log,
                                         noise_file=f'{args.data_path}/{args.r}_{args.noise_mode}.json')
    print('| Building networks')
    net1 = create_model()
    net2 = create_model()
    
    # Create mean teacher models 
    mean_teacher1 = create_model()
    mean_teacher2 = create_model()
    mean_teacher1.load_state_dict(net1.state_dict())
    mean_teacher2.load_state_dict(net2.state_dict())
    
    cudnn.benchmark = True

    criterion = SemiLoss()
    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    all_loss = [[], []]

    for epoch in range(args.num_epochs):
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print(f'Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            # Update mean teacher during warmup too
            update_mean_teacher(net1, mean_teacher1, alpha=args.ema_decay)
            
            print(f'\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader)
            update_mean_teacher(net2, mean_teacher2, alpha=args.ema_decay)
        else:
            
            print('Evaluating with JoSRC...')
            pred1, prob1, pseudo_labels1_np, all_loss[0] = eval_train_josrc(
                net1, net2, mean_teacher1, mean_teacher2, all_loss[0], epoch
            )
            pred2, prob2, pseudo_labels2_np, all_loss[1] = eval_train_josrc(
                net2, net1, mean_teacher2, mean_teacher1, all_loss[1], epoch
            )
            
            # Convert pseudo labels to tensors and create dictionaries
            pseudo_labels1 = torch.from_numpy(pseudo_labels1_np).float()
            pseudo_labels2 = torch.from_numpy(pseudo_labels2_np).float()
            pseudo_dict1 = {i: pseudo_labels1[i] for i in range(len(pseudo_labels1))}
            pseudo_dict2 = {i: pseudo_labels2[i] for i in range(len(pseudo_labels2))}

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
            train(epoch, net1, net2, mean_teacher1, mean_teacher2, optimizer1, 
                  labeled_trainloader, unlabeled_trainloader, pseudo_dict2, writer)

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
            train(epoch, net2, net1, mean_teacher2, mean_teacher1, optimizer2, 
                  labeled_trainloader, unlabeled_trainloader, pseudo_dict1, writer)

        test(epoch, net1, net2, writer)

    writer.close()