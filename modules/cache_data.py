# from typing import Dict, List, Optional

# import torch
# from torch import Tensor 
# import numpy as np
# import msgpack_numpy as m
# import redis
# m.patch() 

# class RedisConnection():
#     def __init__(self):
#         pool = redis.ConnectionPool(host='srd-redis', port=6379, db=0)
#         self.conn = redis.Redis(connection_pool=pool)

#     def make_key(
#             self, 
#             sample_id: int, 
#             src_toks: Tensor, 
#             hyp_toks: Tensor, 
#             state_type: str
#         ):
#         """
#         sample_id : 99
#         src: 3 4 6 7
#         hyp: 99 8 3 6
#         -> 99_3:4:6:7_99:8:3:6
#         """
#         return_keys = []
#         for src_token, hyp_token in zip(src_toks, hyp_toks):
#             src = ":".join([str(st) for st in src_token.tolist()])
#             tgt = ":".join([str(ht) for ht in hyp_token.tolist()])
#             redis_key = "_".join([str(sample_id), src, tgt, state_type])
#             return_keys.append(redis_key)
#         return return_keys

#     def get_dec_state(
#             self, 
#             sample_id, 
#             src_toks, 
#             hyp_toks, 
#             state_type='inner',
#             device=''
#         ):
#         redis_keys = self.make_key(sample_id, src_toks, hyp_toks, state_type)

#         if state_type == 'inner': # 패딩필요
#             # inner (num_tok, bsz, dim)
#             xx = []
#             tok_lens = []
#             for redis_key in redis_keys:
#                 x = m.unpackb(self.conn.get(redis_key))
#                 x = torch.from_numpy(x)
#                 xx.append(x)
#                 tok_lens.append(x.size(0))
#             max_len = np.max()
#             return 
        
#         else: # incrementa state # 패딩 노필요 아니다 필요..
#             for redis_key in redis_keys:
#                 if self.conn.exists(redis_key):
#                     pass
#                 else:
#                     pass

        
#             else: # incremental state
#                 layer_keys = self.conn.get(redis_key)
#                 layer_keys = layer_keys.split(':')
                
#                 incremental_states = torch.jit.annotate(
#                     List[Dict[str, Dict[str, Optional[Tensor]]]],
#                     [
#                         torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
#                         for _ in range(1)
#                     ],
#                 )
                
#                 for key in layer_keys:
#                     for inner_key in ['prev_key', 'prev_value', 'prev_key_padding_mask']:
#                         incremental_key = ':'.join([redis_key, key, inner_key])
#                         x = m.unpackb(self.conn.get(incremental_key))
#                         x = torch.from_numpy(x).to(device)
#                         incremental_states[0][key][inner_key] = x

#                 return incremental_states

#     def set_dec_state(
#             self, 
#             sample_id, 
#             src_toks, 
#             hyp_toks, 
#             x, 
#             state_type='inner'
#         ):
#         redis_keys = self.make_key(sample_id, src_toks, hyp_toks, state_type)
#         if state_type == 'inner':
#             dec_state = m.packb(x.detach().cpu().numpy())
#             self.conn.set(redis_key, dec_state)
#         else:  # incremental state
#             """
#             incremental_state = {
#                 '901920-390290....att_state': {
#                     'prev_key': torch,
#                     'prev_value': torch,
#                     'prev_key_padding_mask': None
#                 }, ...
#             }
#             """
#             layer_keys = list(x.keys())
#             self.conn.set(redis_key, ':'.join(layer_keys))
#             for key in range(layer_keys):
#                 for inner_key in ['prev_key', 'prev_value', 'prev_key_padding_mask']:
#                     incremental_key = ':'.join([redis_key, key, inner_key])
#                     xx = x[key][inner_key].detach().cpu().numpy()
#                     xx = m.packb(xx)
#                     self.conn.set(incremental_key, xx)

