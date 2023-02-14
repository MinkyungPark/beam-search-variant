# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import copy
from tqdm import tqdm
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F

import sacrebleu

# from fairseq import search
from examples.srd_beam_q.modules import search
from fairseq.models import FairseqIncrementalDecoder
from fairseq.ngram_repeat_block import NGramRepeatBlock

from examples.srd_beam_q.modules.latency import length_adaptive_average_lagging

class BeamPolicyGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
        tokens_to_suppress=(),
    ):
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos, self.pad})
            if symbols_to_strip_from_output is not None
            else {self.eos, self.pad}
        )
        
        self.token_indices_to_suppress: Optional[Tensor] = None
        token_indices_to_suppress = []
        for token_string in tokens_to_suppress:
            token_index = tgt_dict.index(token_string)
            assert token_index != self.unk
            token_indices_to_suppress.append(token_index)
        if len(token_indices_to_suppress) > 0:
            self.token_indices_to_suppress = torch.Tensor(
                token_indices_to_suppress
            ).long()

        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.model.set_decoder_beam_size(self.beam_size)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len or self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.beam_search = search.BeamSearch(tgt_dict)
        # self.beam_search = search.PrefixConstrainedBeamSearch(tgt_dict)
        self.beam_search.stop_on_max_len = True

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.enc_embed_dim = self.model.single_model.encoder.embed_tokens.embedding_dim
        self.value_search = search.ValueSearch(tgt_dict)
        
    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin with these tokens
            constraints (torch.LongTensor, optional): force decoder to include the list of constraints
            bos_token (int, optional): beginning of sentence token (default: self.eos)
        """
        result = self._generate(sample, **kwargs)
        return result

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        read_size: int,
        search_size: int,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
        post_process=None,
        decode_fn=None,
    ):
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]

        bsz, src_len = src_tokens.size()[:2]

        self.max_len = 200
        # self.max_len = min(
        #     int(self.max_len_a * src_len + self.max_len_b),
        #     self.max_len - 1,
        # )
        # assert (
        #     self.min_len <= self.max_len
        # ), "min_len cannot be larger than max_len, please adjust these!"
        
        self.bos_token = bos_token
        self.device = src_tokens.device

# ------------------------------------------------ Prepare Encoder Out ------------------------------------------------ #
        enc_incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )
        device = src_tokens.device
        total_enc_out_info = []
        total_enc_out = torch.ones(src_len, src_len+1, bsz, self.enc_embed_dim)
        total_enc_out = total_enc_out.to(device).float()
        total_enc_pad_mask = torch.ones(src_len, bsz, src_len+1)
        total_enc_pad_mask = total_enc_pad_mask.to(device).bool()
        
        for i in range(src_len):
            stream_src = src_tokens[:, :i+1]
            eos_or_pad = torch.all(
                            (stream_src.ne(self.eos) & stream_src.ne(self.pad))
                            , dim=1
                        ).unsqueeze(1)
            padding = torch.full((bsz,1), self.eos).to(src_tokens)
            padding = padding.masked_fill_(~eos_or_pad, self.pad)
            stream_src_eos = torch.concat([stream_src, padding], dim=1)

            # without eos
            stream_net_input = {'src_tokens': stream_src,
                                'incremental_state': enc_incremental_state}
            
            with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
                self.model.forward_encoder(stream_net_input)[0]
                
            without_eos_incremental = copy.deepcopy(enc_incremental_state)
            
            # with eos
            net_input_eos = {'src_tokens': stream_src_eos,
                             'incremental_state': enc_incremental_state}
            
            with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
                eos_enc_out = self.model.forward_encoder(net_input_eos)[0]
                
                origin_src_lenghts = (
                    (stream_src.ne(self.eos) & stream_src.ne(self.pad))
                    .long()
                    .sum(dim=1)
                ).unsqueeze(1)
                
                out = {k:[] for k in eos_enc_out.keys()}
                out['origin_src_lenghts'] = origin_src_lenghts
                out['src_lengths'] = eos_enc_out['src_lengths']
                out['origin_src_tokens'] = stream_src
                out['padded_src_tokens'] = stream_src_eos
                total_enc_out_info.append(out)
                
                total_enc_out[i, :i+2, :, :] = eos_enc_out['encoder_out'][0].unsqueeze(0)
                total_enc_pad_mask[i, :, :i+2] = eos_enc_out['encoder_padding_mask'][0].unsqueeze(0)
                
            enc_incremental_state = without_eos_incremental
    
        total_enc_out = total_enc_out.cpu()
        total_enc_pad_mask = total_enc_pad_mask.cpu()

        # dec_out = self.get_inital_dec_states(src_tokens[:, 0])

        # Reshape enc states
        # L, L+1, B, D -> B, L, L+1 * D -> B * SZ, L, L+1 * D
        l, ll, _, d = total_enc_out.size()
        total_enc_out = total_enc_out.permute(2,0,1,3).contiguous()
        total_enc_out = total_enc_out.view(bsz, l, -1)
        total_enc_out = torch.repeat_interleave(
            total_enc_out, search_size, dim=0
        )
        # L, B, L+1 -> B, L, L+1 -> B * SZ, L, L+1
        total_enc_pad_mask = total_enc_pad_mask.permute(1, 0, 2).contiguous()
        total_enc_pad_mask = torch.repeat_interleave(
            total_enc_pad_mask, search_size, dim=0   
        )
        
        # # Test
        # encoder_outs = [total_enc_out_info[-1]]
        # # sb, l, ll*d -> sb, ll*d -> sb, ll, d -> ll, sb, d
        # encoder_outs[0]['encoder_out'] = [total_enc_out[:, -1].view(-1, ll, d).permute(1,0,2).contiguous().to(device)]
        # # sb, l, ll -> sb, ll
        # encoder_outs[0]['encoder_padding_mask'] = [total_enc_pad_mask[:, -1].contiguous().to(device)]
        # encoder_outs[0]['src_lengths'] = []
        # result = self.beam_decoding(encoder_outs)
        # result = result[::search_size]
        # return result

# ------------------------------------------------ Repeat for search, read size ------------------------------------------------ #
        search_bsz = bsz * search_size
        rs_bsz = search_bsz * read_size
        max_step = (src_lengths[0].item() + sample['target'].size(1)) * 2
        # max_step = 10

        # Origin : bsz
        original_ids: Tensor = sample["id"].cpu()
        original_batch_idxs = torch.arange(bsz).long().cpu()
        targets = {
            idx.item(): target.tolist() for idx, target in zip(original_ids, sample['target'])
        }
        # Final : bsz * search_size * read_size
        aug_size = search_size * read_size
        final_ids = original_ids.repeat_interleave(aug_size)
        final_batch_idxs = original_batch_idxs.repeat_interleave(aug_size)
        # final_src_lens = src_lengths.repeat_interleave(aug_size)
        src_lens = src_lengths.cpu().repeat_interleave(search_size)
        
        encoder_outs = [{k:[] for k in total_enc_out_info[-1].keys()}]

        final = {
            k: torch.zeros(search_bsz, max_step + 1).long()
            for k in ["src_idxs", "action_seqs", "hypo_tokens",
                      "scores", "read_finish", "finish"]
        }
        final["action_seqs"][:, 0] = 1 # for first read token
        final["action_seqs"][:, 1:] = 2 # action pad
        final["hypo_tokens"] += self.pad
        final["scores"] = final["scores"].float()

# ------------------------------------------------ Node Search ------------------------------------------------ #
        for step in tqdm(range(max_step)):
            t = final['finish'].bool()[:, step]
            if torch.all(final['finish'].bool()[:, step]):
                break

            rs_enc_outs = torch.ones(rs_bsz, ll*d).to(total_enc_out)
            rs_pad_masks = torch.ones(rs_bsz, ll).to(total_enc_pad_mask)

            cur_idxs = final["src_idxs"][:, step] # (sb,)
            rs_cur_idxs = cur_idxs.repeat_interleave(read_size) # (rsb,)

            for read_num in range(read_size): # r * sb
                # sb,
                idxs = torch.min((src_lens - 1), cur_idxs + read_num)
                # sb, 1, 1 -> sb, 1, ll*d
                enc_idxs = idxs.view(-1,1,1).repeat(1,1,ll*d)
                # sb, l, ll*d -> sb, 1, ll*d -> sb, ll*d
                tmp = total_enc_out.gather(1, enc_idxs).squeeze()
                rs_enc_outs[read_num::read_size, :] = tmp

                pad_idxs = idxs.view(-1,1,1).repeat(1,1,ll)
                # sb, l, ll -> sb, ll
                tmp = total_enc_pad_mask.gather(1, pad_idxs).squeeze()
                rs_pad_masks[read_num::read_size, :] = tmp
            
            # rsb, ll*d -> rsb, ll, d -> ll, rsb, d
            e = rs_enc_outs.view(-1, ll, d).permute(1, 0, 2).contiguous()
            encoder_outs[0]['encoder_out'] = [e.to(self.device)]
            encoder_outs[0]['encoder_padding_mask'] = [rs_pad_masks.to(self.device)]
            
            prev_tokens = final['hypo_tokens'][:, :step+1]
            prev_mask = (prev_tokens != self.pad)
            mask_sum = torch.cumsum(prev_mask, dim=1)
            prev_idxs = torch.masked_fill(mask_sum, ~prev_mask, 0)
            prev_tokens = torch.scatter(
                torch.ones_like(prev_tokens), dim=1,
                index=prev_idxs, src=prev_tokens
            )[:, 1:]
            prev_tokens = F.pad(prev_tokens, (0,1), value=1)

            prefix_tokens = torch.repeat_interleave(
                prefix_tokens, read_size, dim=0
            )
                
            finalized = self.beam_decoding(
                encoder_outs,
                prefix_tokens=prefix_tokens.to(device),
            )
            torch.cuda.empty_cache()
            
            #### Score
            score_result = {
                sent_id: {k: [] for k in ["hypo_tokens", "bleu", "latency", "scores"]}
                for sent_id in original_ids.tolist()
            }
            for i in range(0, rs_bsz, read_size):
                # [0]00 [0]00 [0]00 [0]00 [1]11 [1]11 [1]11 ..
                sent_id = final_ids[i].item()

                target_str = decode_fn(
                    self.tgt_dict.string(
                        targets[sent_id],
                        post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore={self.eos, self.pad}
                    ))

                tmp = torch.jit.annotate(
                    List[List],
                    [torch.jit.annotate(List, []) for i in range(4)],)
                for j, elem in enumerate(finalized[i:i + read_size]):
                    prev_tks = prefix_tokens[i+j, :step+1]
                    prev_tks = prev_tks[prev_tks.ne(self.pad)]
                    hyp_step = prev_tks.size(0)

                    if elem:
                        beam_tokens = elem[0]['tokens'].cpu()
                        assert torch.all(
                            prev_tks[prev_tks.ne(self.eos)] == 
                            beam_tokens[:hyp_step][beam_tokens[:hyp_step].ne(self.eos)]
                        )
                    else:
                        beam_tokens = torch.concat(
                            (prev_tks, torch.LongTensor([self.eos]), -1)
                        )
                    
                    hyp_strs = self.tgt_dict.string(
                        beam_tokens.tolist(),
                        post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore={self.eos, self.pad}
                    )                    
                    b = sacrebleu.sentence_bleu(decode_fn(hyp_strs), [target_str])
                    bleu_score = round(b.score, 1)
                    latency = rs_cur_idxs[i].item() + 1 + j
                    # score = (bleu_score - (5 * latency)œ)
                    score = (bleu_score - latency)
                    # score = (bleu_score - (0.5 * latency))

                    if elem and j > 0: # read
                        hyp_token = torch.LongTensor([self.pad])
                    else: # write
                        hyp_token = beam_tokens[hyp_step : hyp_step + 1]
                        if hyp_token.numel() == 0:
                            hyp_token = torch.LongTensor([self.eos])
                    
                    for k, item in enumerate([[hyp_token], [bleu_score], [latency], [score]]):
                        tmp[k].extend(item)

                for key, item in zip(['hypo_tokens', 'bleu', 'latency', 'scores'], tmp):
                    score_result[sent_id][key].append(item)    
            
            final = self.value_search.step(
                final=final, 
                final_ids=final_ids,
                score_result=score_result, 
                step=step,
                read_size=read_size,
                keep_size=search_size,
                src_lens=src_lens,
                targets=targets
            )
        # Finish Search
        
        result = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )
        search_batch_idxs = final_batch_idxs[::read_size] # sb
        for s_idx in range(0, search_bsz, search_size):
            hypo_tokens = final['hypo_tokens'][s_idx: s_idx + search_size]
            actions = final['action_seqs'][s_idx: s_idx + search_size]
            
            finish_steps = torch.sum((actions != 2), dim=1) - 1
            scores = final['scores'][s_idx: s_idx + search_size]
            scores = torch.gather(
                scores, dim=1,
                index=finish_steps.unsqueeze(1)
            ).squeeze()
            _, sorted_scores_indices = torch.sort(scores, descending=True)

            result[search_batch_idxs[s_idx]] = [
                {
                    "tokens": torch.LongTensor(hypo_tokens[ssi]),
                    "score": scores[ssi],
                    # "actions": actions[ssi],
                    "alignment": torch.empty(0),
                } for ssi in sorted_scores_indices
            ]

        s = src_lengths.tolist()
        t = [target[target.ne(self.pad)].size(-1) for target in sample['target']]
        path = '/workspace/fairseq/examples/srd_beam_q/scripts/latency.txt'
        for i, final_result in enumerate(zip(final['src_idxs'][::search_size],final['action_seqs'][::search_size])):
            final_src, final_act = final_result
            d = final_src[final_act == 0].unsqueeze(0)
            ss = torch.LongTensor([[s[i]]])
            tt = torch.LongTensor([[d.size(-1)]])
            ref = torch.LongTensor([[t[i]]])
            al = length_adaptive_average_lagging(d, ss, tt, ref)
            with open(path, 'a') as f:
                f.write(str(al.item()) + '\n')

        return result
    
# ------------------------------------------------ Decode Beam Search ------------------------------------------------ #
    def beam_decoding(
        self,
        encoder_outs: List[Dict[str, List]],
        prefix_tokens: Optional[Tensor] = None,
    ):
        beam_size = self.beam_size
        max_len = self.max_len
        bos_token = self.bos_token
        device = self.device

        bsz = encoder_outs[0]['encoder_out'][0].size(1)

        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(device).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(device)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(device).long().eq(-1)
        )  # forward and backward-compatible False mask

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            with torch.autograd.profiler.record_function(
                "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores, inners = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            else:
                if step < self.min_len:
                    # minimum length constraint (does not apply if using prefix_tokens)
                    lprobs[:, self.eos] = -math.inf

                if self.token_indices_to_suppress is not None:
                    lprobs[:, self.token_indices_to_suppress] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)


            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.beam_search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    # src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.beam_search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # finished hypotheses
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.beam_search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                # src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            # Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.beam_search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        # src_lengths,
        max_len: int,
        has_prev: bool=False,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}

        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min(
            [
                m.max_decoder_positions()
                for m in self.models
                if hasattr(m, "max_decoder_positions")
            ]
            + [sys.maxsize]
        )

    def set_decoder_beam_size(self, beam_size):
        """Set beam size for efficient beamable enc-dec attention."""
        if beam_size > 1:
            for model in self.models:
                if hasattr(model, "set_beam_size"):
                    model.set_beam_size(beam_size)

    @torch.jit.export
    def forward_encoder(
        self, 
        net_input: Dict[str, Tensor],
    ):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn, decoder_out[1]['inner_states'][-1]

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)

        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )
