import torchintx
import torch
import asr
import torch.nn.functional as F
import torch.onnx
import configparser
import sys
from asr.utils import train_helper
from asr.data import Pfileinfo, PfileDataLoader, TestPfileDataLoader
from asr.data import clip_mask
from asr.optim import SGD
from asr.utils import clip_grad_norm
# import editdistance
import scipy.io as sio
import numpy as np
import os

import sentencepiece as spm

spmodel="/raw15/asrdictt/permanent/zhyou2/lm/arabic/spm_ar_bpe8000.model"
spvocab="/raw15/asrdictt/permanent/zhyou2/lm/arabic/spm_ar_bpe8000.vocab.blank"
sp = spm.SentencePieceProcessor(model_file=spmodel)

iter_idx = sys.argv[1]
part_idx = sys.argv[2]
device = int(sys.argv[3])
config = configparser.ConfigParser()
config.read(sys.argv[4])
print(sys.argv[4])

test_mlf = "/train20/intern/permanent/lxju/code/test_other/data/fleurs/arabic/arabic_test_opensrc_fleurs.mlf"
test_fea_pfile = "/train20/intern/permanent/lxju/code/test_other/data/fleurs/arabic/ed/"
test_setname =[
"arabic_test_opensrc_fleurs"
]

word_dict = "./lexicon_norm3_final_w0.1_preplus.20200507.mark.wlist"
word_dict_lab = "./lexicon_norm3_final_w0.1_preplus.20200507.mark.wlist"
# word_dict_lab = "/work/sppro/dyliu2/getpfile/ED/10wh_chneng6.5kh/lib/lexicon_norm3_final_w0.1_preplus.wlist"

print('here')
class VocabularyRec(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        self.idx2word.append("<BLANK>")
        with open(f_dic, "r", encoding="GBK") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                self.idx2word.append(x)
                self.word2idx[x] = idx + 1

    def list_to_sent(self, li):
        words = [self.idx2word[i] for i in li]
        return " ".join(words)

class VocabularyLab(object):
    def __init__(self,f_dic):
        self.idx2word = []
        self.word2idx = {}
        with open(f_dic, "r", encoding="GBK") as f:
            for idx, line in enumerate(f):
                x = line.strip()
                self.idx2word.append(x)
                self.word2idx[x] = idx

    def list_to_sent(self, li):
        # print(len(self.idx2word))
        words = [self.idx2word[i] for i in li[1:]]
        return " ".join(words)


def edit_dist(labs, recs):
    n_lab = len(labs)
    n_rec = len(recs)

    dist_mat = np.zeros((n_lab+1, n_rec+1))
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

# def computer_cer(preds, labels):
#     dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
#     total = sum(len(l) for l in labels)
#     return dist, total


def decode(enc_state, lengths):
    token_list = []

    dec_state, hidden = model.decoder(zero_token)

    decs = []
    decs.append(dec_state)
    probs = []

    for t in range(lengths):
        logits = model.joint(enc_state[t].view(-1), dec_state.view(-1))
        # print(logits.shape);exit()  #torch([8002])
        probs.append(logits)
        out = F.softmax(logits[:-1], dim=0).detach()

        pred = torch.argmax(out, dim=0)
        pred = int(pred.item())

        if pred != 0:
            token_list.append(pred-1)
            token = torch.LongTensor([[pred]])

            if enc_state.is_cuda:
                token = token.cuda()

            dec_state, hidden = model.decoder(token, hidden=hidden)
            decs.append(dec_state)

    return token_list


if __name__ == "__main__":
    for name in test_setname:
        max_sent = 1
        test_lab_pfile = test_fea_pfile + name+".lab.pfile"
        test_feature_pfile = test_fea_pfile + name+ ".fea.pfile"
        test_scp = test_fea_pfile + name+ ".lab.scp"

        print(test_feature_pfile)
        print(test_lab_pfile)
        print(test_scp)
        decode_dir = "./res/{}_{}".format(iter_idx,part_idx)
        total_sentnum = Pfileinfo(test_feature_pfile).num_sentences
        with torch.cuda.device(device):
            # d = PfileDataLoader(file_fea=test_feature_pfile,
            #                             file_lab=test_lab_pfile,
            #                             file_norm=config["DataSetting"]["NormFile"],
            #                             batch_num=int(total_sentnum/max_sent),
            #                             bunchsize=2048,
            #                             maxsentframe=1200,
            #                             maxnumsent=max_sent,
            #                             start_sent=0,
            #                             end_sent=total_sentnum,
            #                             ndivide=1,
            #                             divide_index=0,
            #                             num_workers=1,
            #                             nmod_pad=4,
            #                             shuffle_batch=False)
            d = TestPfileDataLoader(file_fea=test_feature_pfile,
                                file_lab=test_lab_pfile,
                                file_norm=config["DataSetting"]["NormFile"],
                                batch_num=total_sentnum,
                                # bunchsize=2048,
                                # maxsentframe=1200,
                                # maxnumsent=1,
                                start_sent=0,
                                end_sent=total_sentnum,
                                # ndivide=1,
                                # divide_index=0,
                                nmod_pad=4,
                                # shuffle_batch=False,
                                # random_seed=0
                                )
            
                                
            model = train_helper.get_module(config["Model"]["ModelName"])()
            clamp_modules=(torch.nn.BatchNorm2d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Conv1d,torch.nn.LSTM)
            model = torchintx.clamp_layers(model, clamp_modules = clamp_modules, clamp_weight_value=8, clamp_bias_value=8, clamp_output_value=None,clamp_dynamic_percent=1.0)
            test_model_name = "./{}/model.iter{}.part{}".format(config["TrainSetting"]["OutDir"], iter_idx, part_idx)
            state = torch.load(test_model_name)["state_dict"]
            model.load_state_dict(state)
            model = model.cuda()
            
            voc_rec = VocabularyRec(word_dict)
            voc_lab = VocabularyLab(word_dict_lab)
            
            if not os.path.exists(decode_dir):
                os.makedirs(decode_dir)

            rec_file = decode_dir + '/{}_rec.txt'.format(name)

            ## dec
            f = open(rec_file, 'w', encoding='gbk')
            f.close()

            count = 0
            total_dist = 0
            total_word = 0
            total_sc = 0
            index = 0

            for idx, (data, meta) in enumerate(d):
                # f = open(rec_file, 'a', encoding='gbk')
                model = model.eval()
                data = data.cuda()
                
                x=data
                (b, c, f, t) = x.size()
                if f != 40:
                    x = x.permute(0,1,3,2).reshape(b,c,-1,40).permute(0,1,3,2)
                    data=x
                f = open(rec_file, 'a', encoding='utf8')
                for key in meta:
                    meta[key] = meta[key].cuda()

                targets = meta["att_label"].permute(1,0).contiguous()
                att_mask = meta["att_mask"].permute(1,0).contiguous()
                targets[targets == -1] = -2
                targets = targets.add(1)
                targets_length = att_mask.sum(1)
                # print("targets_length",targets_length)
                # exit()
                
                enc_state = model.encoder(data, meta)
                data_mask = clip_mask(meta["rnn_mask"], enc_state.size(1), 0)
                data_mask = data_mask.squeeze(2).permute(1,0)
                # print("data_mask",data_mask)
                inputs_length =  data_mask.sum(1)

                zero_token = torch.LongTensor([[0]])
                if data.is_cuda:
                    zero_token = zero_token.cuda()

                results = []
                batch_size = data.size(0)
                print("SENT",idx)
                for i in range(batch_size):
                    # print("SENT",i)
                    # print(enc_state.shape, inputs_length[i])
                    decoded_seq = decode(enc_state[i], inputs_length[i].int())
                    # print("decoded_seq",decoded_seq)
                    results.append(decoded_seq)
                    # print(results)
                    
                
                transcripts = [targets.cpu().numpy()[i][:targets_length[i].int().item()]
                            for i in range(targets.size(0))]
                #print("transcripts",transcripts)
                
                label = meta["att_label"]
                label[label<-1] = -1
                label = label.flatten()
                label = label[label != -1]
                label = label.long().tolist()
                # print("label",label)
                #ground_label = voc_lab.list_to_sent(label)
                
                #"""
                # print(index)
                f.write("Sent#%d\n"%index)
                index += 1
                f.write("label:" + voc_lab.list_to_sent(label) + '\n')
                #f.write("label:" + voc_lab.list_to_sent(transcripts[0][1:]) + '\n')
                # f.write("preds:" + voc_rec.list_to_sent(results[0]) + '\n')
                # print(results[0], sp.decode(results[0]))
                f.write("preds:" + sp.decode(results[0]) + '\n')

                f.flush()
        ### dec end

            #     # print(label)
            #     # print(results[0])
            #     # exit()
            #     # print(label[1:])
            #     # print(results[0][1:-1])
            #     # dist = edit_dist(label[1:], results[0][1:-1])
            #     # print(dist)
            #     # num_words = len(label) - 2
            #     # total_dist += dist
            #     # total_word += num_words

            #     # cer = total_dist / total_word * 100
            #     # acc = 100 - cer
            #     # count += 1
            #     # if(dist == 0): total_sc += 1 

            # # print(test_model_name)
            # # print("sc: ", total_sc/count * 100)
            # # print("acc: ", acc, "\n")

            # # f.write("sc: " + str(total_sc/count * 100) + '\n')
            # # f.write("acc: " + str(acc) + "\n")
            # f.close()

            ### stat
            # os.system("python get_acc_total.py {0} {1} {2}".format(iter_idx,part_idx,name))

            rlts = {}
            tmp_rlts = {}
            with open(rec_file, 'r', encoding="utf8") as f:
                rlt_lines = f.readlines()

            with open(test_scp, 'r') as f:
                scp_lines = f.readlines()

            name = None
            for line in rlt_lines:
                line = line.strip()
                
                if line.startswith("Sent#"):
                    sentid = line.replace("Sent#", '')
                    sentid = int(sentid)
                    name = scp_lines[sentid].strip()
                    if name not in rlts:
                        rlts[name]=[]
                        # for i in range(1500):
                        #     rlts[name].append("")


                if "preds" in line:
                    line = line.strip()
                    line = line.replace("preds:", '')
                    line = line.replace("</s>", '')
                    # line = line.replace("<unk>", '')
                    
                    rlts[name] = line


            rec = "{}.mlf".format(rec_file)
            out = "{}.out.2".format(rec_file)
            with open(rec, 'w') as f:
                f.write("#!MLF!#\n")
                for key in rlts:
                    f.write('"' + key + ".wav" + '"' + '\n')
                    value = rlts[key]
                    # val_sent = '<s>~'
                    # for word in value:
                    #     f.write(word + '\n')
                        # val_sent += word + '~'
                    f.write(value.replace(' ', '\n')+'\n')
                    f.write('.\n')
                    # f_sent.write(val_sent+'</s>\n')
                    ### print(val_sent);exit()

            cmd = "./HResults -t -f -d 32  -I {0} -e \"???\" \"<s>\" -e \"???\" \"</s>\" ./hmmlist {1} >{2}".format(test_mlf, rec, out)
            print(cmd)
            os.system(cmd)

            cmd = "grep 'Acc=' {}".format(out)
            os.system(cmd)

                #"""
