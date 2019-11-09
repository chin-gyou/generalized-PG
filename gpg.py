from basepara import *
from pytorch_pretrained_bert import BertTokenizer

class seqattn(base):
    def __init__(self, em, h_size, d_size, w_size, lr, bi=False):
        super().__init__()
        #word embedding
        self.embedding = nn.Embedding.from_pretrained(em, freeze = False)
        dc_size = 2*h_size if bi else h_size
        
        self.outproj = nn.Linear(dc_size + d_size, em.size(1))

        self.edit1 = torch.nn.Sequential(
            torch.nn.Linear(dc_size + d_size, d_size),
            torch.nn.ReLU(),
            torch.nn.Linear(d_size, dc_size + d_size),
            )
        self.edit = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(dc_size + d_size, em.size(1))
        )

        self.attnD = nn.Linear(d_size, dc_size)

        self.encoder = EncoderRNN(em.size(1), h_size, 0.3, bi)
        self.decoder = DecoderRNN(self.embedding, d_size, context_size=dc_size, drop_out = 0.3, i_size = dc_size)
        
        self.em_out = nn.Linear(em.size(1), em.size(0))
        self.em_out.weight = self.embedding.weight

        self.switch = torch.nn.Sequential(
            torch.nn.Linear(d_size + dc_size, w_size),
            torch.nn.ReLU(),
            torch.nn.Linear(w_size, 1),
            torch.nn.Sigmoid()
            )
        self.lr = lr

    def get_emd(self, text):
        emb = self.embedding(text)
        return emb#F.normalize(emb, p=2, dim=-1)

    #return the encoded vector of the context
    def encode(self, enc_text):
        mask = (enc_text!=PAD).float()
        slen = mask.sum(0)
        text_embed = self.get_emd(enc_text)
        seq_inputs = text_embed#enc_len*batch_size*inputs_len
        enc_states, ht = self.encoder.run(seq_inputs, slen, self.mode)
        return seq_inputs, enc_states, ht#seq_len*[batch_size*h_size]
        
    def cent(self, p1, p2):
        kl = -p1*torch.log((p1+1e-10))
        return kl
    
    #return the context vector and most focused position after attention
    def attnvec(self, encs, dec, umask, n_e, dcont, seq_inputs, target=None, batch=None):
        attnencs = encs
        attndec = self.attnD(dec).unsqueeze(0)
        dot_products = torch.sum(attnencs*attndec, -1)#seq_len*batch_size

        if target is None:
            topi = torch.range(0, encs.size(0) - 1).unsqueeze(-1).expand_as(dot_products).long().to(DEVICE)
        else:
            emd_dots = torch.sum(encs*dcont.unsqueeze(0), -1)
            _, topi = emd_dots.topk(n_e, 0)
        c_word = seq_inputs.gather(0, topi.unsqueeze(-1).expand(-1,-1, seq_inputs.size(2)))

        weights = softmax_mask(dot_products, 1-umask)
        
        topv = weights.gather(0, topi)#6*batch_size
        c_all = torch.sum(encs*(weights.unsqueeze(-1)),0)
        del dot_products
        topi = topi.unsqueeze(-1).expand(-1,-1, encs.size(2))
        c_vec = encs.gather(0, topi)#6*batch_size*h_size
        del encs
        return c_vec, topv, c_all, weights.max(0)[0], c_word#/(topv.sum(0, True) + 1e-10)

    """
    return the cost of one batch
    dec_text: dec_len*batch_size
    enc_text, enc_fd, enc_pos, enc_rpos: enc_len*batch_size
    """
    def forward(self, batch):
        dec_text = batch[0].detach()
        enc_text = batch[1].detach()
        seq_inputs, context, ht = self.encode(enc_text)
        _, dcontext, _ = self.encode(dec_text)
        umask = (batch[1]==PAD).float()#seq_len*batch_size
        last_h = torch.cat((ht[0], ht[1]), 1)
        d_hidden = nonlinear(self.decoder.initS(last_h))
        c_hidden = nonlinear(self.decoder.initC(last_h))
        pad_mask = (dec_text!=PAD).float()
        o_loss, p_loss = 0, 0
        t_len=torch.sum((dec_text!=PAD).float(), 0, True)
        n_e = min(kenum, context.size(0)) #if self.mode == 0 else context.size(0)
        t_sent, t_mask = 0, 0
        for i in range(dec_text.size(0)):
            c_vec, topv, c_all, mweight, c_word = self.attnvec(context, d_hidden, umask, n_e, dcontext[i].detach(), seq_inputs, 0, batch)
            sentinel = self.switch(torch.cat((d_hidden, c_all), 1))
            t_sent += sentinel.squeeze(-1)*pad_mask[i]
            allc = torch.cat((c_vec, c_all.unsqueeze(0)), 0)
            cprob = torch.cat((sentinel.t()*topv, 1-sentinel.t()), 0)
            c_in = torch.cat((d_hidden.unsqueeze(0).expand(n_e,-1, -1), c_vec), 2)
            p_output = self.em_out(c_word + self.edit(self.edit1(c_in) + c_in))
            n_output = self.em_out(self.outproj(torch.cat((d_hidden.unsqueeze(0), c_all.unsqueeze(0)), 2)))
            output = torch.cat((p_output, n_output), 0)

            gmask = (mweight > thresh).float()
            t_mask += gmask*pad_mask[i]
            t_prob = F.softmax(output, -1)
        
            tg_prob = torch.gather(t_prob, 2, dec_text[i].view(1, -1,1).expand(n_e+1, -1,-1)).squeeze(-1)
            marginal_l = torch.sum(tg_prob*cprob, 0)# + 1e-10
            o_loss -= torch.log(marginal_l)*pad_mask[i]
            p_loss -= torch.log(sentinel.squeeze(-1))*gmask*pad_mask[i]
            del t_prob
            del output
            d_hidden, c_hidden = self.decoder(dec_text[i], d_hidden, c_hidden, c_all, self.mode)
        return o_loss.unsqueeze(0), t_len, p_loss.unsqueeze(0)

    def cost(self, forwarded):
        oloss, tlen, ploss = forwarded
        return (oloss.sum() + ploss.sum())/tlen.sum(), oloss.sum(), tlen.sum(), oloss.sum()/tlen.sum(), ploss.sum()/tlen.sum()

    def decode(self, batch, decode_length = 50, beam_size = 5, k = 6):
        k = batch[1].size(0)
        n_e = k
        umask = (batch[1]==PAD).float()#seq_len*batch_size
        seq_inputs, context, ht = self.encode(batch[1])
        last_h = torch.cat((ht[0], ht[1]), 1)
        d_hidden = nonlinear(self.decoder.initS(last_h))
        d_cell = nonlinear(self.decoder.initC(last_h))
        
        decoder_outputs = torch.ones(beam_size, decode_length).long().to(DEVICE)
        decoder_score = torch.zeros(beam_size).float().to(DEVICE)
        continue_mask = beam_size#number of unfinished sentences
        finished = torch.ones(beam_size, decode_length).long().to(DEVICE)
        f_score = -1e8*torch.ones(beam_size).float().to(DEVICE)
        #decode first token
        c_vec, topv, c_all, _,  c_word = self.attnvec(context, d_hidden, umask, k, 0, seq_inputs)
        sentinel = self.switch(torch.cat((d_hidden, c_all), 1))
        #print(sentinel)
        allc = torch.cat((c_vec, c_all.unsqueeze(0)), 0)
        cprob = torch.cat((sentinel.t()*topv, 1-sentinel.t()), 0)
        c_in = torch.cat((d_hidden.unsqueeze(0).expand(n_e,-1, -1), c_vec), 2)
        p_output = self.em_out(c_word + self.edit(self.edit1(c_in) + c_in))
        n_output = self.em_out(self.outproj(torch.cat((d_hidden.unsqueeze(0), c_all.unsqueeze(0)), 2)))
        o = torch.cat((p_output, n_output), 0)
        #print(o.size())
        vocab_size = o.size(-1)
        joint_l = (F.softmax(o, -1)*cprob.unsqueeze(-1)).view(-1, vocab_size)
        tprob = joint_l.sum(0)
        topv, topi = tprob.topk(beam_size)
        decoder_outputs[:,0] = topi
        decoder_score = torch.log(topv)
        #p_l = joint_l[:k].index_select(1, topi)
        #p_l = p_l/p_l.sum(0, True)
        #print(p_l[:,0])
        #postv = joint_l.index_select(1,topi)/topv.unsqueeze(0)
        d_hidden = d_hidden.repeat(beam_size, 1)
        d_cell = d_cell.repeat(beam_size, 1)
        #c_vec = torch.sum(c_vec.repeat(1, beam_size, 1)*(p_l.unsqueeze(-1)), 0)
        #print('cvec1:', c_vec[0,:10])
        c_all = c_all.repeat(beam_size, 1)
        context = context.repeat(1, beam_size, 1)
        umask = umask.repeat(1, beam_size)
        dec_input = topi
        #print(topi)
        #print(decoder_score)
        for i in range(decode_length-1):
            nseq_inputs = seq_inputs.repeat(1, continue_mask, 1)
            #print(decoder_outputs)
            d_hidden, d_cell = self.decoder(dec_input, d_hidden, d_cell, c_all, 2)
            c_vec, topv, c_all,_, c_word = self.attnvec(context, d_hidden, umask, k, 0, nseq_inputs)
            sentinel = self.switch(torch.cat((d_hidden, c_all), 1))
            allc = torch.cat((c_vec, c_all.unsqueeze(0)), 0)
            cprob = torch.cat((sentinel.t()*topv, 1-sentinel.t()), 0)
            c_in = torch.cat((d_hidden.unsqueeze(0).expand(n_e,-1, -1), c_vec), 2)
            p_output = self.em_out(c_word + self.edit(self.edit1(c_in) + c_in))
            n_output = self.em_out(self.outproj(torch.cat((d_hidden.unsqueeze(0), c_all.unsqueeze(0)), 2)))
            o = torch.cat((p_output, n_output), 0)

            joint_l = (F.softmax(o, -1)*cprob.unsqueeze(-1))
            
            marginal_l = joint_l.sum(0)
            #if i > 2:
            #    zerois = find_trigram(decoder_outputs, i + 1)
            #    marginal_l[zerois] = 1e-10
            tprob = torch.log(marginal_l) + decoder_score.unsqueeze(-1)#continus_mask*V
            topv, topi = tprob.view(-1).topk(continue_mask)
            back_pointer = topi/vocab_size
            
            #print(back_pointer)
            decoder_outputs[:,:i+1] = decoder_outputs[back_pointer,:i+1]
            #print(decoder_outputs)
            decoder_outputs[:,i+1] = topi%vocab_size
            decoder_score = topv
            d_hidden = d_hidden[back_pointer]
            d_cell = d_cell[back_pointer]
            dec_input = topi%vocab_size
            #joint_l = joint_l[:k].index_select(1, back_pointer)
            #tg_prob = torch.gather(joint_l, 2, dec_input.view(1, -1,1).expand(k, -1,-1)).squeeze(-1)
            #postv = tg_prob/tg_prob.sum(0, True)
            #c_vec = torch.sum(c_vec.index_select(1,back_pointer)*postv.unsqueeze(-1),0)
            #print(topi%vocab_size)
            #print('cvec2:', c_vec[0,:10])
            #print('score2:', topv)
            #print(decoder_outputs)
            c_all = c_all[back_pointer]
            num_END = (dec_input==END).sum().long()
            if num_END > 0:
                END_mask = (dec_input==END).nonzero().squeeze(-1)
                other_mask = (dec_input!=END).nonzero().squeeze(-1)
                continue_mask-= num_END
                finished[continue_mask:continue_mask+num_END] = decoder_outputs[END_mask]
                f_score[continue_mask:continue_mask+num_END] = decoder_score[END_mask]/(i+2)
                if continue_mask == 0:
                    break
                decoder_outputs = decoder_outputs[other_mask]
                decoder_score = decoder_score[other_mask]
                d_hidden = d_hidden[other_mask]
                d_cell = d_cell[other_mask]
                dec_input = dec_input[other_mask]
                c_all = c_all[other_mask]
                context = context[:,:continue_mask, :]
                umask = umask[:,:continue_mask]

        if continue_mask > 0:
            finished[:continue_mask] = decoder_outputs
        topv, topi = f_score.topk(beam_size)
        #print(topv)
        #print(finished[topi[0]].unsqueeze(0))
        #sys.exit()
        return finished[topi[0]].unsqueeze(0)#batch_size*decode_length
        
if __name__ == '__main__':
    #em = torch.from_numpy(pickle.load(open(embed, 'rb')))
    #s=seqattn(em, h_size, d_size, w_size, lr, True)
    s=torch.load(open('./seq2seq/gpg_pre/best_epoch','rb'))
    #s.outproj = torch.nn.Sequential(
    #    torch.nn.Linear(2*h_size + d_size, d_size),
    #    torch.nn.Tanh(),
    #    torch.nn.Linear(d_size, em.size(0)),
    #    )
    print(s.lr)
    print(s.encoder.bi)
    s=s.to(DEVICE)
    parameters = filter(lambda p: p.requires_grad, s.parameters())
    s.optim = torch.optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)
    s.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(s.optim, 'min', factor = 0.1, patience = 1, min_lr = 1e-6)
    #s.optim = torch.optim.Adagrad(parameters, lr=s.lr, initial_accumulator_value= 0.1)
    #s.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(s.optim, 'min', factor = 0.5, patience = 1)
    #dloader = DataLoader(tabledata(data_dir, 'test'), batch_size = b_size, shuffle=False, collate_fn = merge_sample)
    
    #s.output_decode(dloader, './decoding/gigatest/sli2_95', BertTokenizer.from_pretrained('./pytorch-pretrained-BERT/examples/bert-base-uncased-vocab.txt', do_lower_case=True))
    #s.validate(dloader, './test.txt')
    s.run_train(data_dir, num_epochs=20, b_size = b_size, check_dir = check_dir, lazy_step = lazy_steps)
