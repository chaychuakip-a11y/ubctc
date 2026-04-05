import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class ACC(nn.Module):
    def __init__(self):
        super(ACC, self).__init__()


    def forward(self, logit, label):
        target = self.getlabel(label)
        lprobs = torch.nn.functional.log_softmax(logit, dim=1)
        acc = self.getacc(lprobs, target)
        return acc

    @jit.ignore
    def getlabel(self, label):
        target = label.clone()
        target = target.flatten()
        target[target < 0] = -1
        target = target.long()
        target = target.unsqueeze(-1)
        return target

    @jit.ignore
    def getacc(self, lprob, target):
        num_class = lprob.size()[1]
        _, new_target = torch.broadcast_tensors(lprob, target)

        remove_pad_mask = new_target.ne(-1)
        lprob = lprob[remove_pad_mask]

        target = target[target != -1]
        target = target.unsqueeze(-1)

        lprob = lprob.reshape((-1, num_class))

        preds = torch.argmax(lprob, dim=1)
        correct_holder = torch.eq(preds.squeeze(), target.squeeze()).float()

        num_corr = correct_holder.sum()
        num_sample = torch.numel(correct_holder)
        acc = num_corr / num_sample
        return acc


class AccCtc(nn.Module):
    def __init__(self):
        super(AccCtc, self).__init__()

    def forward(self, logit, label):
        # print('logit', logit.shape)
        # print('label', label.shape)
        target = self.getlabel(label)
        # print('target2', target.shape)
        lprobs = F.log_softmax(logit, dim=1)
        # print('lprobs2', lprobs.shape)
        acc = self.getacc(lprobs, target)
        return acc

    @jit.ignore
    def edit_dist(self, labs:list, recs:list):
        n_lab = len(labs)
        n_rec = len(recs)

        dist_mat = torch.zeros((n_lab+1, n_rec+1))
        for j in range(n_rec + 1):
            dist_mat[0, j] = j
        
        for i in range(n_lab + 1):
            dist_mat[i, 0] = i

        for i in range(1, n_lab+1):
            for j in range(1, n_rec+1):
                hit_score = dist_mat[i-1,j-1] + (labs[i-1]!=recs[j-1])
                ins_score = dist_mat[i,j-1] + 1
                del_score = dist_mat[i-1,j] + 1

                err = hit_score
                if err > ins_score:
                    err = ins_score
                if err > del_score:
                    err = del_score
                dist_mat[i, j]=err

        return dist_mat[n_lab, n_rec]

    @jit.ignore
    def getlabel(self, label):
        target = label.clone()
        target = target.flatten()
        target[target < 0] = -1
        target = target.long()
        target = target.unsqueeze(-1)
        return target

    @jit.ignore
    def getacc(self, lprob, target):
        num_class = lprob.size(1)
        # _, new_target = torch.broadcast_tensors(lprob, target)
        
        target = target[target != -1]
        if target.shape[0] == 0:
            return 1.0
        # target = target.unsqueeze(-1)

        # remove_pad_mask = new_target.ne(-1)
        # lprob = lprob[remove_pad_mask]
        lprob = lprob.reshape((-1, num_class))
        preds = torch.argmax(lprob, dim=1)
        
        # print('preds', preds.shape, preds)
        # print('target', target.shape, target)
        
        preds = preds.tolist()
        target = target.tolist()
        
        preds_new = list()
        id = -1
        for pred in preds:
            if pred == num_class-1:
                continue
            if pred != id:
                id = pred
                preds_new.append(pred)
            else:
                continue
        
        # print('preds_new', preds_new);exit()
        
        dist = self.edit_dist(preds_new, target)
        acc = 1-(dist/len(target))
        # print('dist:', dist, ', acc:', acc)
        return acc

