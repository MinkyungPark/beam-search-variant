import math

import numpy as np
import torch
from torch import Tensor
from typing import List, Dict, Optional

from fairseq.search import Search


class ValueSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None
        self.policy = [0.5, 0.5]

    @torch.jit.export
    def step(
        self,
        final: Dict[str, Tensor],
        final_ids: Tensor, # bsz * search_size * read_size
        score_result: Dict[int, Dict[str, List]],
        step: int,
        read_size: int,
        keep_size: int,
        src_lens: Tensor, # bsz * search_size
        targets: Dict,
    ):
        search_ids = final_ids[::read_size]
        for sent_id, result in score_result.items():
            target_check = targets[sent_id]
            # Set Mask & Index
            id_mask = (search_ids == sent_id)
            src_len = src_lens[id_mask][:1]
            
            prev_src_idxs = final['src_idxs'][id_mask, step] # (id_bsz,)
            prev_finish = final['finish'][id_mask, step]
            prev_read_finish = final['read_finish'][id_mask, step]
            
            match_idxs = torch.repeat_interleave(
                torch.arange(keep_size),
                read_size, dim=0
            ).long()
            arange_idxs = torch.arange(match_idxs.size(0)).long()

            scores = torch.FloatTensor(result['scores']).flatten()
            hypos = torch.LongTensor(result['hypo_tokens']).flatten()
            
            # 0=False=Write, 1=True=Read 
            actions = torch.arange(read_size).repeat(keep_size).long()
            actions[actions != 0] = 1

            # Top K
            top_prediction = torch.topk(
                scores,
                k=keep_size
            )
            top_scores, top_indexs = top_prediction
            # ---------------------------------------------------- policy
            same_scores = torch.full((keep_size,), -1)
            for i, scr in enumerate(top_scores.tolist()):
                not_visited = (same_scores == -1)
                if not_visited[i]:
                    same_mask = (top_scores == scr)
                    same_mask = (same_mask & not_visited)
                    same_scores[same_mask] = i
            
            for pos in set(same_scores.tolist()):
                pos_mask = (same_scores == pos)
                num = torch.sum(pos_mask).item()
                if num > 1:
                    pos_idxs = match_idxs[top_indexs[pos_mask]]
                    wr_sample = torch.index_select(
                        arange_idxs.view(keep_size, read_size),
                        dim=0, index=pos_idxs
                    )
                    wr_sample = wr_sample[:, ::read_size-1]
                    
                    sampled_act = torch.from_numpy(
                        np.random.choice(2, num, p=self.policy)
                    )
                    # sampled_act = torch.ones(num).long()
                    # write_mask = prev_read_finish[pos_idxs].bool()
                    # sampled_act[write_mask] = 0
                    
                    top_indexs[pos_mask] = torch.gather(
                        wr_sample, 
                        dim=1, 
                        index=sampled_act.unsqueeze(1)
                    ).squeeze()

                    top_scores[pos_mask] = torch.index_select(
                        scores, 0, top_indexs[pos_mask]
                    )
            # ---------------------------------------------------- policy
            match_top_idxs = match_idxs[top_indexs]

            # Reorder
            actions = actions[top_indexs]
            _prev_finish_mask = prev_finish[match_top_idxs].bool()
            # prev_finish : action -> 2
            actions[_prev_finish_mask] = 2

            # Reorder
            _prev_src_idxs = prev_src_idxs[match_top_idxs]
            src_idxs = torch.min(
                (src_len - 1), # cannot be larger than src token length
                _prev_src_idxs + actions,
            )
            
            max_src_idx = (src_len.item() - 1)
            read_finish = (src_idxs == max_src_idx).long()
            
            # not read finish : eos -> pad
            top_hypos = hypos[top_indexs]
            hyp_mask = ~(read_finish.bool()) & (top_hypos == self.eos)
            top_hypos[hyp_mask] = self.pad

            eos_mask = (top_hypos == self.eos)
            finish = (read_finish.bool() & eos_mask).long()

            # Reorder final[:, :step+1]
            key_list = list(final.keys())
            for key in key_list:
                final[key][id_mask] = torch.index_select(
                    final[key][id_mask],
                    dim=0,
                    index=match_top_idxs
                )
                            
            # Update            
            final['src_idxs'][id_mask, step+1] = src_idxs
            final['action_seqs'][id_mask, step+1] = actions
            final['scores'][id_mask, step+1] = top_scores
            final['hypo_tokens'][id_mask, step+1] = top_hypos
            final['read_finish'][id_mask, step+1] = read_finish
            final['finish'][id_mask, step+1] = finish

        return final
    

class BeamSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        # lprobs (1, B, detokens)
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf


class PrefixConstrainedBeamSearch(Search):
    def __init__(self, tgt_dict, prefix_allowed_tokens_fn):
        super().__init__(tgt_dict)
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.stop_on_max_len = True

    @torch.jit.export
    def apply_mask(self, x, prev_output_tokens, original_batch_idxs):
        beam_size = x.shape[0] // original_batch_idxs.shape[0]
        original_batch_idxs = (
            original_batch_idxs.unsqueeze(-1).repeat((1, beam_size)).flatten().tolist()
        )

        mask = torch.full_like(x, -math.inf)
        for sent_i, (sent, batch_i) in enumerate(
            zip(prev_output_tokens, original_batch_idxs)
        ):
            mask[sent_i, :, self.prefix_allowed_tokens_fn(batch_i, sent)] = 0

        return mask

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs: Tensor,
        scores: Tensor,
        prev_output_tokens: Tensor,
        original_batch_idxs: Tensor,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        lprobs += self.apply_mask(
            lprobs.view(bsz * beam_size, 1, vocab_size),
            prev_output_tokens,
            original_batch_idxs,
        ).view(bsz, beam_size, vocab_size)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf