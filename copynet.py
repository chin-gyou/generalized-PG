from basepara import *
from pytorch_pretrained_bert import BertTokenizer


class seqattn(base):
    def __init__(self, em, h_size, d_size, w_size, lr, bi=False):
        super().__init__()
        #word embedding
        self.embedding = nn.Embedding.from_pretrained(em, freeze = False)
        
        dc_size = 2*h_size if bi else h_size
        self.attnD = nn.Linear(d_size, dc_size)
        
        self.switch = torch.nn.Sequential(
            torch.nn.Linear(d_size + dc_size, w_size),
            torch.nn.ReLU(),
            torch.nn.Linear(w_size, 1),
            torch.nn.Sigmoid()
            )

        self.encoder = EncoderRNN(em.size(1), h_size, 0.2, bi)
        self.decoder = DecoderRNN(self.embedding, d_size, context_size=dc_size, drop_out = 0.2)
        self.outproj = nn.Linear(dc_size + d_size, em.size(1))
        self.em_out = nn.Linear(em.size(1), em.size(0))
        self.em_out.weight = self.embedding.weight
        #optimizer
        self.lr = lr
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1.2e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', factor = 0.1, patience = 1)

    #return the encoded vector of the context
    def encode(self, enc_text):
        mask = (enc_text!=PAD).float()
        slen = mask.sum(0)
        text_embed = self.embedding(enc_text)
        seq_inputs = text_embed#enc_len*batch_size*inputs_len
        enc_states, ht = self.encoder.run(seq_inputs, slen, self.mode)
        return seq_inputs, enc_states, ht#seq_len*[batch_size*h_size]
        
    #return the context vector and most focused position after attention
    def attnvec(self, encs, dec, umask):
        attnencs = encs
        attndec = self.attnD(dec).unsqueeze(0)
        dot_products = torch.sum(attnencs*attndec, -1)#seq_len*batch_size
        weights = softmax_mask(dot_products, 1-umask)
        c_vec = torch.sum(encs*(weights.unsqueeze(-1)),0)#batch_size*h_size
        
        #n_e = min(kenum, weights.size(0))
        #topv, topi = weights.topk(n_e,0)#6*batch_size
        #print(topi, topv, batch['enc_id'].gather(0, topi))
        
        #_, m_foc = torch.max(weights, 0)
        return c_vec, weights#, m_foc

    """
    return the cost of one batch
    dec_text: dec_len*batch_size
    enc_text, enc_fd, enc_pos, enc_rpos: enc_len*batch_size
    """
    def forward(self, batch):
        dec_text = batch[0]
        enc_text = batch[1]
        _, context, ht = self.encode(enc_text)
        umask = (batch[1]==PAD).float()#seq_len*batch_size
        last_h = torch.cat((ht[0], ht[1]), 1)
        d_hidden = nonlinear(self.decoder.initS(last_h))
        c_hidden = nonlinear(self.decoder.initC(last_h))
        o_loss = 0
        t_len=torch.sum((dec_text!=PAD).float(), 0, True)
        pad_mask = (dec_text!=PAD).float()
        t_sent = 0
        for i in range(dec_text.size(0)):
            c_vec, weights = self.attnvec(context, d_hidden, umask)
            #print(moc)
            #print(batch['enc_txt'][0][moc])
            output = self.em_out(self.outproj(torch.cat((d_hidden, c_vec), 1)))
            t_prob = F.softmax(output, -1)
        
            tg_prob = torch.gather(t_prob, 1, dec_text[i].unsqueeze(-1)).squeeze()
            del t_prob
            
            sentinel = self.switch(torch.cat((d_hidden, c_vec), 1)).squeeze(-1)
            #print(sentinel)
            #t_sent += sentinel.squeeze(-1)*pad_mask[i]
            w_mask = (enc_text == dec_text[i].unsqueeze(0)).float()
            point_l = torch.sum(weights*w_mask, 0)
            o_loss -= torch.log(1e-10 + (1-sentinel)*tg_prob + sentinel*point_l)*pad_mask[i]
            
            d_hidden, c_hidden = self.decoder(dec_text[i], d_hidden, c_hidden, c_vec, self.mode)
        #sys.exit()
        #print(t_sent.sum()/t_len.sum())
        return o_loss.unsqueeze(0), t_len
        #return t_sent.unsqueeze(0), t_len

    def cost(self, forwarded):
        oloss, tlen = forwarded
        return oloss.sum()/tlen.sum(), oloss.sum(), tlen.sum(), tlen.sum()
    
    def decode(self, batch, decode_length = 100, beam_size = 5):
        _, context, ht = self.encode(batch[1])
        umask = (batch[1]==PAD).float()#seq_len*batch_size
        last_h = torch.cat((ht[0], ht[1]), 1)
        d_hidden = nonlinear(self.decoder.initS(last_h))
        d_cell = nonlinear(self.decoder.initC(last_h))
        
        match = torch.eye(30522)[batch[1].squeeze(-1)].to(DEVICE)
     
        decoder_outputs = torch.ones(beam_size, decode_length).long().to(DEVICE)
        decoder_score = torch.zeros(beam_size).float().to(DEVICE)
        continue_mask = beam_size#number of unfinished sentences
        finished = torch.ones(beam_size, decode_length).long().to(DEVICE)
        f_score = -1e8*torch.ones(beam_size).float().to(DEVICE)
        #decode first token
        c_vec, weights = self.attnvec(context, d_hidden, umask)
        #print(d_hidden.size())
        #print(c_vec.size())
        o = self.em_out(self.outproj(torch.cat((d_hidden, c_vec), 1)))
        #o[:,0] -= 1e8
        o = F.softmax(o, -1)
        sentinel = self.switch(torch.cat((d_hidden, c_vec), 1)).squeeze(-1)
        #sentinel = self.switch(d_hidden).squeeze(-1)
        o *= (1-sentinel)
        o += torch.sum((sentinel*weights)*match, 0, True)
        topv, topi = o.topk(beam_size)
        decoder_outputs[:,0] = topi
        decoder_score = torch.log(topv.squeeze(0))

        d_hidden = d_hidden.repeat(beam_size, 1)
        d_cell = d_cell.repeat(beam_size, 1)
        c_vec = c_vec.repeat(beam_size, 1)
        context = context.repeat(1, beam_size, 1)
        umask = umask.repeat(1, beam_size)
        dec_input = topi.squeeze(0)
        for i in range(decode_length-1):
            d_hidden, d_cell = self.decoder(dec_input, d_hidden, d_cell, c_vec, 2)
            c_vec, weights = self.attnvec(context, d_hidden, umask)
            o = self.em_out(self.outproj(torch.cat((d_hidden, c_vec), 1)))
            o = F.softmax(o, -1)

            sentinel = self.switch(torch.cat((d_hidden, c_vec), 1))
            #sentinel = self.switch(d_hidden)
            o *= (1-sentinel)
            #print(continue_mask, context.size(0))
            #print(o[torch.range(0, continue_mask-1).unsqueeze(1).expand(continue_mask, context.size(0)).long(), encs.t()].size())
            o += torch.mm(sentinel*weights.t(),match)
            if i > 2:
                zerois = find_trigram(decoder_outputs, i + 1)
                o[zerois] = 1e-10
            #print(o.sum(-1))
            #sys.exit()
            #o[:,0] -= 1e8
            o = torch.log(o) + decoder_score.unsqueeze(1)#beam_size*vocab_size
            topv, topi = o.view(-1).topk(continue_mask)
            #topv, topi = topv.squeeze(0), topi.squeeze(0)
            back_pointer = topi/o.size(1)
            decoder_outputs[:,:i+1] = decoder_outputs[back_pointer,:i+1]
            decoder_outputs[:,i+1] = topi%o.size(1)
            decoder_score = topv
            d_hidden = d_hidden[back_pointer]
            d_cell = d_cell[back_pointer]
            c_vec = c_vec[back_pointer]
            dec_input = topi%o.size(1)
            num_END = (dec_input==END).sum().long()
            if num_END > 0:
                END_mask = (dec_input==END).nonzero().squeeze(-1)
                other_mask = (dec_input!=END).nonzero().squeeze(-1)
                continue_mask-= int(num_END.cpu().numpy())
                finished[continue_mask:continue_mask+num_END] = decoder_outputs[END_mask]
                f_score[continue_mask:continue_mask+num_END] = decoder_score[END_mask]/(i+2)
                if continue_mask == 0:
                    break
                decoder_outputs = decoder_outputs[other_mask]
                decoder_score = decoder_score[other_mask]
                d_hidden = d_hidden[other_mask]
                d_cell = d_cell[other_mask]
                dec_input = dec_input[other_mask]
                c_vec = c_vec[other_mask]
                context = context[:,:continue_mask, :]
                umask = umask[:,:continue_mask]
                if continue_mask == 0:
                    break
        
        if continue_mask > 0:
            finished[:continue_mask] = decoder_outputs
        topv, topi = f_score.topk(beam_size)
        return finished[topi[0]].unsqueeze(0)#batch_size*decode_length
        #return finished#batch_size*decode_length
        
if __name__ == '__main__':
    em = torch.from_numpy(pickle.load(open(embed, 'rb')))
    s=seqattn(em, h_size, d_size, w_size, lr, True)
    #s=torch.load(open('./seq2seq/cnn/copy2/best_epoch','rb'))
    print(s.encoder.bi)
    s=s.to(DEVICE)
    #dloader = DataLoader(tabledata(data_dir, 'test'), batch_size = b_size, shuffle=False, collate_fn = merge_sample)
    #s.output_decode(dloader, './decoding/cnn/copy2', BertTokenizer.from_pretrained('./pytorch-pretrained-BERT/examples/bert-base-uncased-vocab.txt', do_lower_case=True))
    #copy_replace('./processed_data/test/box.val','./outputs_mf','outputs','attn_outputs')
    #s.validate(dloader, './test.txt')
    s.run_train(data_dir, num_epochs=30, b_size = b_size, check_dir = check_dir)
