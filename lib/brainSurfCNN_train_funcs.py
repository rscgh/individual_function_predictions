# https://github.com/ngohgia/brain-surf-cnn/blob/56b3c62bf84a0c37180944b1d25280d6e4940689/utils/experiment.py#L10

import numpy as np
import torch 


def run_epoch(model, train_loader, contrasts, optimizer, loss_fn, \
  loss_type, within_subj_margin, across_subj_margin, train = True, ret_detailed_corrs=False):
    
    if train: 
      model.train()
    else:
      model.eval()
    
    total_loss = 0
    total_corr = []
    count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
      #data, target = data.cuda(), target.cuda()
      
      if train: 
        optimizer.zero_grad()
      
      output = model(data) # silently fails here
      
      if loss_type == 'mse':
         loss = loss_fn(output, target)
      elif loss_type == 'rc':
         within_subj_loss, across_subj_loss = loss_fn(output, target)
         loss = torch.clamp(within_subj_loss - within_subj_margin, min=0.0) + torch.clamp(within_subj_loss - across_subj_loss + across_subj_margin, min = 0.0)
      else:
         raise("Invalid loss type")
      
      if train:
        loss.backward()
        optimizer.step()
      
      total_loss += loss.item()
      
      reshaped_output = np.swapaxes(output.cpu().detach().numpy(), 0, 1)
      reshaped_target = np.swapaxes(target.cpu().detach().numpy(), 0, 1)
      corrs = np.diag(compute_corr_coeff(reshaped_output.reshape(reshaped_output.shape[0], -1), reshaped_target.reshape(reshaped_target.shape[0], -1)))
      if batch_idx == 0:
        total_corr = corrs
      else:
        total_corr = total_corr + corrs
            
    total_loss /= len(train_loader)
    total_corr /= len(train_loader)
    
    print((" Train" if train else "Val") + ': avg loss: {:.4f} - avg corr: {:.4f}'.format(total_loss, np.mean(total_corr)))
    for j in range(len(contrasts)):
        print("      %s: %.4f" % (contrasts[j], total_corr[j]))
    
    if ret_detailed_corrs:
      return total_loss, np.mean(total_corr), total_corr

    return total_loss, np.mean(total_corr)


# row wise correlation of two arrays 
def compute_corr_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    
