import torch
import torch.nn.functional as F
from config import device, d_model, d_ff, d_k, d_v, n_layers, n_heads, dropout
from config import GlobalVars

def beam_search_decoder(model, enc_input, start_symbol, beam_size=3):
    """
    Beam search decoder for the Transformer model.
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :param beam_size: The number of beams to maintain during search
    :return: The target input sequence with the highest score
    """
    g = GlobalVars.instance()
    enc_outputs, _ = model.encoder(enc_input)
    # Initialize the beam with the start symbol
    beams = [(torch.tensor([[start_symbol]], dtype=enc_input.dtype).cuda(), 0.0)]

    for _ in range(g.tgt_len - 1):  # tgt_len is the maximum length of the target sequence
        new_beams = []
        for beam_seq, beam_score in beams:
            if beam_seq.size(1) >= g.tgt_len or beam_seq[0, -1].item() == g.tgt_vocab['<E>']:
                new_beams.append((beam_seq, beam_score))
                continue
            # Get the decoder output for the current beam sequence
            dec_outputs, _, _ = model.decoder(beam_seq, enc_input, enc_outputs)
            projected = model.projection(dec_outputs)
            # Get the probabilities for the next word
            probs = torch.nn.functional.log_softmax(projected[:, -1, :], dim=-1)
            # Get the top k candidates
            topk_probs, topk_indices = probs.topk(beam_size)
            for i in range(beam_size):
                next_word = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([beam_seq, next_word], dim=-1)
                new_score = beam_score + topk_probs[0, i].item()
                # 若新词是结束符，标记为完成路径
                if next_word.item() == g.tgt_vocab['<E>']:
                    new_beams.append((new_seq, new_score))
                else:
                    new_beams.append((new_seq, new_score))
        # Keep only the top beam_size sequences
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # Return the sequence with the highest score
    return beams[0][0]
