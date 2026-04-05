# by dlp/yjliu
import torch
import torch.nn as nn
import numpy as np
LZERO = 1e30

class BeamSearcher(object):
    def __init__(self,sos,eos,pad,beam_size=10):
        self.beam_size = beam_size
        self.pad = pad
        self.sos = sos
        self.eos = eos

        self.scores = torch.FloatTensor(self.beam_size).zero_()
        self.lens = torch.FloatTensor(self.beam_size).zero_() + 1
        self.prevKs = []
        self.nextYs = [torch.LongTensor(self.beam_size).fill_(self.pad)]
        self.nextYs[0][0] = self.sos
        self.not_finished = torch.ByteTensor(self.beam_size).zero_() + 1
        self.done = False

    def get_current_out(self):
        return self.nextYs[-1], self.not_finished.sum()

    def get_current_origin(self):
        return self.prevKs[-1]

    def step(self, out_scores):
        #import pdb;pdb.set_trace()

        out_scores = out_scores.data
        voc_size = out_scores.size(1)

        if len(self.prevKs) > 0:
            not_finished = self.not_finished.float()
            not_finished_expand = not_finished.unsqueeze(1).expand_as(out_scores)
            out_scores = out_scores * not_finished_expand - LZERO * (1 - not_finished_expand) 
            out_scores[:, self.pad] = out_scores[:, self.pad] * not_finished
            beam_score = out_scores + self.scores.unsqueeze(1).expand_as(out_scores)
            norm_beam_score = beam_score / self.lens.unsqueeze(1)

        else:
            beam_score = out_scores[0]
            norm_beam_score = beam_score
            
        flat_beam_score = beam_score.view(-1)
        flat_norm_beam_score = norm_beam_score.view(-1)
        best_scores, best_ids = flat_norm_beam_score.topk(self.beam_size, 0, True, True) # best_id range from 0 to beam_size*num_class -1

        self.scores = flat_beam_score.gather(0, best_ids)
        prev_k = torch.floor_divide(best_ids, voc_size) # prev_k range from 0 to beam_size - 1 

        self.prevKs.append(prev_k)
        nextY = best_ids - prev_k*voc_size
        self.nextYs.append(nextY)

        self.not_finished = self.not_finished[prev_k]
        self.not_finished = self.not_finished.cuda() * nextY.ne(self.eos).byte()
        lens_sel = self.lens[prev_k]
        self.lens = lens_sel.cuda() + self.not_finished.float()

        if self.not_finished.sum() == 0:
            self.done = True

        return self.done

    def get_best(self, topk=1):
        assert topk>=1, "topk must be greater than 1"
        scores_norm = self.scores/self.lens
        scores, ids = torch.sort(scores_norm, 0, True)
        return scores[:topk].tolist(), ids[:topk].tolist()

    def get_hyp(self, k):
        hyp = []
        hyp.append(self.pad)
        for j in range(len(self.prevKs)-1, -1, -1):
            y = self.nextYs[j+1][k]
            if y != self.pad:
                hyp.append(y.item())
            k = self.prevKs[j][k]
        return hyp[::-1]




    
