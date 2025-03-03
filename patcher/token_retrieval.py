from contextlib import contextmanager
from math import ceil
from typing import Optional

import sglang
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from vllm.config import (
    DeviceConfig,
    ModelConfig,
    LoRAConfig,
    MultiModalConfig,
    ParallelConfig,
    SchedulerConfig,
    CacheConfig,
)
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.model_loader.loader import (
    DefaultModelLoader,
    _initialize_model,
    device_loading_context,
)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

ROPE_BASE = -1
ROPE_SCALE = -1
ROPE_MODE = ""
MAX_N_TOKENS = -1
TOP_K = -1
N_INIT = -1
N_Local = -1
PREFILL_CHUNK_SIZE = -1
QUERY_ROTATE = False
QUERY_CACHE = False
KERNEL_SIZE=-1

@contextmanager
def cuda_timer(timer_name="Operation"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    yield

    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    print(f"{timer_name} time (ms): {elapsed_time:.4f}")


class RotaryEmbedding(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            base: float,
            distance_scale: float = 1.0,
            device: torch.device = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.distance_scale = distance_scale
        self.device = device
        self._seq_len_cached = -1
        self._cos_table = None
        self._sin_table = None

    def _init_inv_freq(self):
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
                self.base
                ** (
                        torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
                        / self.dim
                )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _update_cos_sin_tables(self, seq_len: int):
        """
        _cos_table and _sin_table are 2-D tensors, but adapts to 2-4D input via broadcast.
        """
        if not hasattr(self, "inv_freq"):
            self._init_inv_freq()
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.device)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_table = emb.cos().unsqueeze(1)
            self._sin_table = emb.sin().unsqueeze(1)

    def apply_rotary_pos_emb(
            self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        self._update_cos_sin_tables(int(position_ids.max()) + 1)
        cos = self._cos_table[position_ids, :]
        sin = self._sin_table[position_ids, :]
        rotated_x = ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(
            x.dtype
        )
        return rotated_x

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_rotary_pos_emb(x, position_ids)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class ReqToTokenRetriever:
    def __init__(
            self,
            num_layers,
            head_dim,
            num_heads,
            num_kv_heads,
            fingerprint_dim,
            max_num_tokens,
            token_to_kv_pool,
            dtype,
            device,
    ):
        # now we only support one running request
        self.rope_embedding = RotaryEmbedding(head_dim, ROPE_BASE, ROPE_SCALE, device)
        self.kwargs = {
            "num_layers": num_layers,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "fingerprint_dim": fingerprint_dim,
            "max_num_tokens": max_num_tokens,
            "dtype": dtype,
            "device": device,
            "token_to_kv_pool": token_to_kv_pool,
            "rotary_embedding": self.rope_embedding,
        }
        self.current_req_id = None
        self.current_token_retriever = None

    def get_token_retriever(self, req_id):
        if self.current_req_id != req_id:
            self.current_req_id = req_id
            if (
                    self.current_token_retriever is not None
                    and QUERY_CACHE
                    and self.current_token_retriever.retrieval_count > 0
            ):
                print("skip_count:", self.current_token_retriever.skip_count)
                print("retrieval_count:", self.current_token_retriever.retrieval_count)
                print(
                    "skip_rate:",
                    self.current_token_retriever.skip_count
                    / self.current_token_retriever.retrieval_count,
                )
            self.current_token_retriever = TokenRetriever(**self.kwargs)
        return self.current_token_retriever


@triton.jit
def paged_matmul_kernel(
        # Pointers to input and output tensors
        query_ptr,  # [num_heads, head_dim]
        token_ptr,  # [max_num_tokens, num_kv_heads, head_dim]
        indices_ptr,  # [num_relevant_tokens]
        scores_ptr,  # [num_heads, num_relevant_tokens]
        # Variables
        num_relevant_tokens,
        # Constants
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_TOKENS: tl.constexpr = 128,
):
    """
    Triton kernel to compute softmax scores for queries against keys per head with GQA.

    Parameters:
    - query_ptr: Pointer to query_fingerprints [num_heads, head_dim]
    - token_ptr: Pointer to token_fingerprints [max_num_tokens, num_kv_heads, head_dim]
    - indices_ptr: Pointer to relevant_indices [num_relevant_tokens]
    - scores_ptr: Pointer to output scores [num_heads, num_relevant_tokens]
    - num_heads: Number of attention heads
    - num_relevant_tokens: Number of relevant tokens
    - num_kv_heads: Number of KV heads (num_heads % num_kv_heads == 0)
    - head_dim: Dimension of each head
    """

    head_id = tl.program_id(0)  # Head index
    token_block_id = tl.program_id(1)  # Token block index

    token_start = token_block_id * BLOCK_SIZE_TOKENS
    token_indices = token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    mask_tokens = token_indices < num_relevant_tokens

    kv_head = head_id % NUM_KV_HEADS  # KV head index for the current head

    # Shape: [head_dim]
    query_offset = head_id * HEAD_DIM
    query = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_DIM),
        mask=head_id < NUM_HEADS,
        other=0.0,
    )

    # Shape: [BLOCK_SIZE_TOKENS]
    indices = tl.load(indices_ptr + token_indices, mask=mask_tokens, other=0)

    # Shape: [BLOCK_SIZE_TOKENS, head_dim]
    token_offsets = (
            (indices[:, None] * NUM_KV_HEADS * HEAD_DIM)  # [num_relevant_tokens, 1]
            + (kv_head * HEAD_DIM)  # scalar
            + tl.arange(0, HEAD_DIM)  # [head_dim]
    )
    tokens = tl.load(token_ptr + token_offsets, mask=mask_tokens[:, None], other=0.0)

    # Shape: [BLOCK_SIZE_TOKENS]
    scores = tl.sum(query[None, :] * tokens, axis=1)

    # Shape: [num_heads, num_relevant_tokens]
    scores_offset = (
            head_id * num_relevant_tokens + token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    )

    tl.store(scores_ptr + scores_offset, scores, mask=mask_tokens)


def paged_matmul(
        query,  # torch.Tensor of shape [num_heads, head_dim]
        token,  # torch.Tensor of shape [max_num_tokens, num_kv_heads, head_dim]
        indices,  # torch.Tensor of shape [num_relevant_tokens]
        scores,  # torch.Tensor of shape [num_heads, num_relevant_tokens]
        num_relevant_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        BLOCK_SIZE_TOKENS=128,
):
    num_token_blocks = (
                               num_relevant_tokens + BLOCK_SIZE_TOKENS - 1
                       ) // BLOCK_SIZE_TOKENS
    grid = (num_heads, num_token_blocks)

    paged_matmul_kernel[grid](
        query_ptr=query,
        token_ptr=token,
        indices_ptr=indices,
        scores_ptr=scores,
        num_relevant_tokens=num_relevant_tokens,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        num_warps=4,
    )


class TokenRetriever:
    def __init__(
            self,
            num_layers,
            head_dim,
            num_heads,
            num_kv_heads,
            fingerprint_dim,
            max_num_tokens,
            dtype,
            device,
            token_to_kv_pool=None,
            rotary_embedding: Optional["RotaryEmbedding"] = None,
    ):

        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.fingerprint_dim = fingerprint_dim
        self.max_num_tokens = max_num_tokens
        self.dtype = dtype
        self.device = device

        self.token_fingerprints = [
            token_to_kv_pool.get_key_buffer(layer_id)
            for layer_id in range(self.num_layers)
        ]

        self.token_indices = torch.empty(
            (self.num_layers, self.max_num_tokens),
            device=self.device,
            dtype=torch.int32,
        )

        self.rope_embedding = rotary_embedding

        self.query_fingerprints_cache = torch.empty(
            (self.num_layers, self.num_heads * self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )

        self.topk_indices_cache = torch.empty(
            (self.num_layers, TOP_K),
            device=self.device,
            dtype=torch.int64,
        )

        self.similarity_threshold = torch.tensor(
            [0.9 for _ in range(self.num_layers)], device=self.device, dtype=self.dtype
        )

        self.skip_count = 0
        self.retrieval_count = 0

        self.is_first_query = [True for _ in range(self.num_layers)]

        self.clear()

    def clear(self):
        self.num_tokens = [0 for _ in range(self.num_layers)]
        self.is_first_query = [True for _ in range(self.num_layers)]
        self.skip_count = 0

    def get_all_tokens(self, layer_id):
        return self.token_indices[layer_id, : self.num_tokens[layer_id]]

    def add_k_cache(self, indices, layer_id):
        tail_idx = self.num_tokens[layer_id] + indices.shape[0]
        assert tail_idx <= self.max_num_tokens
        self.token_indices[layer_id, self.num_tokens[layer_id]: tail_idx] = indices
        self.num_tokens[layer_id] = tail_idx

    def get_topk_tokens(self, query_fingerprints, token_fingerprints, topk, indices):
        num_q_heads = query_fingerprints.shape[-1] // self.head_dim
        query_fingerprints = query_fingerprints.view(num_q_heads, self.head_dim)

        num_heads = num_q_heads
        num_tokens = indices.shape[0]

        scores = torch.empty(
            (num_heads, num_tokens), device=self.device, dtype=torch.bfloat16
        )

        # Launch the Triton kernel
        paged_matmul(
            query_fingerprints,
            token_fingerprints,
            indices,
            scores,
            num_tokens,
            num_heads,
            self.num_kv_heads,
            self.head_dim,
        )

        scores = torch.softmax(scores, dim=-1).sum(dim=0)
        
        if dist.is_initialized():  # TP
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
        
        if KERNEL_SIZE > 0:
            scores = torch.nn.functional.max_pool1d(
                scores.unsqueeze(0), kernel_size=KERNEL_SIZE, padding=(KERNEL_SIZE-1)//2, stride=1
            ).squeeze(0)
        
        topk_indices = torch.topk(scores, topk, dim=-1).indices
        sorted_topk_tokens = torch.sort(topk_indices).values
        return sorted_topk_tokens

    def retrieval_indices(self, query, layer_id, n_init, n_local, topk):
        current_num_tokens = self.num_tokens[layer_id]
        if n_init + topk + n_local >= current_num_tokens:
            return None

        if QUERY_ROTATE:
            position_ids = torch.arange(
                n_local,
                n_local + query.shape[0],
                device=self.device,
            )
            query = self.rope_embedding(
                query.view(query.shape[0], -1, self.head_dim), position_ids
            ).view(query.shape[0], -1)
        else:
            query = query.view(query.shape[0], -1)

        query_fingerprints = torch.mean(query, dim=0)

        if QUERY_CACHE:
            if (
                    self.is_first_query[layer_id]
                    or torch.cosine_similarity(
                self.query_fingerprints_cache[layer_id], query_fingerprints
            )
                    < self.similarity_threshold[layer_id]
            ):
                token_fingerprints = self.token_fingerprints[layer_id][
                    self.token_indices[
                    layer_id, n_init: self.num_tokens[layer_id] - n_local
                    ]
                ]
                token_fingerprints = token_fingerprints.view(
                    token_fingerprints.shape[0], -1
                )

                topk_tokens = (
                        self.get_topk_tokens(query_fingerprints, token_fingerprints, topk)
                        + n_init
                )
                self.topk_indices_cache[layer_id] = topk_tokens
                self.query_fingerprints_cache[layer_id] = query_fingerprints
                self.is_first_query[layer_id] = False
            else:
                topk_tokens = self.topk_indices_cache[layer_id]
                self.skip_count += 1
            self.retrieval_count += 1

        else:
            relevant_indices = self.token_indices[
                               layer_id, n_init: current_num_tokens - n_local
                               ]

            token_fingerprints = self.token_fingerprints[layer_id]
            topk_tokens = (
                    self.get_topk_tokens(
                        query_fingerprints, token_fingerprints, topk, relevant_indices
                    )
                    + n_init
            )

        retrieved_tokens = torch.cat(
            [
                torch.arange(0, n_init, device=self.device),
                topk_tokens,
                torch.arange(
                    current_num_tokens - n_local,
                    current_num_tokens,
                    device=self.device,
                ),
            ]
        )

        final_indices = self.token_indices[layer_id, retrieved_tokens]
        return final_indices


def patch_model_runner():
    from sglang.srt.model_executor.model_runner import ModelRunner

    class PatchedModelRunner(ModelRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.req_to_token_retriever = ReqToTokenRetriever(
                self.model_config.num_hidden_layers,
                self.model_config.head_dim,
                self.model_config.num_attention_heads,
                self.model_config.get_num_kv_heads(self.tp_size),
                self.model_config.get_num_kv_heads(self.tp_size)
                * self.model_config.head_dim,
                MAX_N_TOKENS,
                self.token_to_kv_pool,
                self.dtype,
                "cuda",
            )
            print("max_total_num_tokens:", self.max_total_num_tokens)

    sglang.srt.model_executor.model_runner.ModelRunner = PatchedModelRunner


def patch_input_metadata():
    class PatchedInputMetadata(InputMetadata):

        token_retriever: TokenRetriever

        def __init__(self, **kwargs):
            self.token_retriever = kwargs.pop("token_retriever")
            super().__init__(**kwargs)

        def init_flashinfer_handlers(
                self,
                model_runner,
                prefix_lens,
                flashinfer_use_ragged,
        ):
            flashinfer_use_ragged = False
            patched_forward_batch_info_update_flashinfer_indices(
                self.forward_mode,
                model_runner,
                self.req_pool_indices,
                self.seq_lens,
                prefix_lens,
                flashinfer_use_ragged=flashinfer_use_ragged,
            )

            (
                self.flashinfer_prefill_wrapper_ragged,
                self.flashinfer_prefill_wrapper_paged,
                self.flashinfer_decode_wrapper,
                self.flashinfer_use_ragged,
            ) = (
                model_runner.flashinfer_prefill_wrapper_ragged,
                model_runner.flashinfer_prefill_wrapper_paged,
                model_runner.flashinfer_decode_wrapper,
                flashinfer_use_ragged,
            )

        @classmethod
        def from_schedule_batch(
                cls,
                model_runner,
                batch: ScheduleBatch,
                forward_mode: ForwardMode,
        ):
            ret = cls(
                forward_mode=forward_mode,
                batch_size=batch.batch_size(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                req_to_token_pool=model_runner.req_to_token_pool,
                token_to_kv_pool=model_runner.token_to_kv_pool,
                token_retriever=model_runner.req_to_token_retriever.get_token_retriever(
                    batch.reqs[0].rid
                ),
                out_cache_loc=batch.out_cache_loc,
                return_logprob=batch.return_logprob,
                top_logprobs_nums=batch.top_logprobs_nums,
            )

            ret.compute_positions(batch)

            ret.compute_extend_infos(batch)

            if (
                    forward_mode != ForwardMode.DECODE
                    or model_runner.server_args.disable_flashinfer
            ):
                ret.total_num_tokens = int(torch.sum(ret.seq_lens))

            if forward_mode != ForwardMode.DECODE:
                ret.init_multimuldal_info(batch)

            prefix_lens = None
            if forward_mode != ForwardMode.DECODE:
                prefix_lens = torch.tensor(
                    [len(r.prefix_indices) for r in batch.reqs], device="cuda"
                )

            if model_runner.server_args.disable_flashinfer:
                ret.init_triton_args(batch, prefix_lens)

            flashinfer_use_ragged = False
            if not model_runner.server_args.disable_flashinfer:
                if (
                        forward_mode != ForwardMode.DECODE
                        and int(torch.sum(ret.seq_lens)) > 4096
                        and model_runner.sliding_window_size is None
                ):
                    flashinfer_use_ragged = True
                ret.init_flashinfer_handlers(
                    model_runner, prefix_lens, flashinfer_use_ragged
                )

            return ret

    def patched_forward_batch_info_update_flashinfer_indices(
            forward_mode,
            model_runner,
            req_pool_indices,
            seq_lens,
            prefix_lens,
            flashinfer_decode_wrapper=None,
            flashinfer_use_ragged=False,
    ):
        """Init auxiliary variables for FlashInfer attention backend."""
        num_qo_heads = (
                model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
        head_dim = model_runner.model_config.head_dim
        batch_size = len(req_pool_indices)

        if model_runner.sliding_window_size is None:
            if flashinfer_use_ragged:
                raise NotImplementedError("Ragged attention not supported yet")
            else:
                paged_kernel_lens = seq_lens

            kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
            req_pool_indices_cpu = req_pool_indices.cpu().numpy()
            paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
            kv_indices = torch.cat(
                [
                    model_runner.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            ).contiguous()

            kv_last_page_len = torch.ones(
                (batch_size,), dtype=torch.int32, device="cuda"
            )

            if forward_mode == ForwardMode.DECODE:
                # CUDA graph uses different flashinfer_decode_wrapper
                if flashinfer_decode_wrapper is None:
                    flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

                flashinfer_decode_wrapper.end_forward()
                flashinfer_decode_wrapper.begin_forward(
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
            else:

                # extend part
                qo_indptr = torch.zeros(
                    (batch_size + 1,), dtype=torch.int32, device="cuda"
                )
                qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

                if flashinfer_use_ragged:
                    raise NotImplementedError("Ragged attention not supported yet")

                # cached part
                model_runner.flashinfer_prefill_wrapper_paged.end_forward()
                model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
        else:
            raise NotImplementedError("Sliding window not supported yet")

    sglang.srt.model_executor.forward_batch_info.InputMetadata = PatchedInputMetadata
    sglang.srt.model_executor.forward_batch_info.update_flashinfer_indices = (
        patched_forward_batch_info_update_flashinfer_indices
    )


def patch_model():
    def patched_default_model_loader_load_model(
            self,
            *,
            model_config: ModelConfig,
            device_config: DeviceConfig,
            lora_config: Optional[LoRAConfig],
            multimodal_config: Optional[MultiModalConfig],
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
    ) -> torch.nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    lora_config,
                    multimodal_config,
                    cache_config,
                    scheduler_config,
                )
            model.load_weights(
                self._get_weights_iterator(
                    model_config.model,
                    model_config.revision,
                    fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                ),
            )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return patch_attention(model.eval())

    def patch_attention(model):

        def patched_radix_attention_extend_forward_flashinfer(
                self, q, k, v, input_metadata: InputMetadata
        ):
            # using two wrappers is unnecessary in the current PR, but are prepared for future PRs
            prefill_wrapper_paged = input_metadata.flashinfer_prefill_wrapper_paged
            if self.sliding_window_size != -1:
                prefill_wrapper_paged = prefill_wrapper_paged[0]
            else:
                if isinstance(prefill_wrapper_paged, list):
                    prefill_wrapper_paged = prefill_wrapper_paged[1]

            assert not input_metadata.flashinfer_use_ragged  # not supported yet

            num_chunks = ceil(q.shape[0] / PREFILL_CHUNK_SIZE)
            outputs = torch.empty_like(q)

            kv_last_page_len = prefill_wrapper_paged._paged_kv_last_page_len_buf.clone()
            qo_indptr = prefill_wrapper_paged._qo_indptr_buf.clone()

            for chunk_idx in range(num_chunks):
                start = chunk_idx * PREFILL_CHUNK_SIZE
                end = min((chunk_idx + 1) * PREFILL_CHUNK_SIZE, q.shape[0])

                if k is not None:
                    assert v is not None
                    self.store_kv_cache(
                        k,
                        v,
                        input_metadata,
                        start=start,
                        end=end,
                    )

                retrieved_indices = input_metadata.token_retriever.retrieval_indices(q[start:end].contiguous(),
                                                                                     self.layer_id, N_INIT, N_Local,
                                                                                     TOP_K)
                retrieved_indptr = prefill_wrapper_paged._paged_kv_indptr_buf.clone()

                if retrieved_indices is None:
                    retrieved_indices = input_metadata.token_retriever.get_all_tokens(
                        self.layer_id
                    )

                retrieved_indptr[1] = len(retrieved_indices)
                qo_indptr[1] = end - start
                prefill_wrapper_paged.end_forward()
                prefill_wrapper_paged.begin_forward(
                    qo_indptr,
                    retrieved_indptr,
                    retrieved_indices,
                    kv_last_page_len,
                    self.tp_q_head_num,
                    self.tp_k_head_num,
                    self.head_dim,
                    1,
                )

                o = prefill_wrapper_paged.forward(
                    q[start:end]
                    .contiguous()
                    .view(-1, self.tp_q_head_num, self.head_dim),
                    input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                    causal=True,
                    sm_scale=self.scaling,
                    window_left=self.sliding_window_size,
                    logits_soft_cap=self.logit_cap,
                    rope_scale=ROPE_SCALE,
                    rope_theta=ROPE_BASE,
                    pos_encoding_mode=ROPE_MODE,
                )

                outputs[start:end] = o.view(-1, self.tp_q_head_num * self.head_dim)
            return outputs

        def patched_radix_attention_decode_forward_flashinfer(
                self, q, k, v, input_metadata: InputMetadata
        ):
            decode_wrapper = input_metadata.flashinfer_decode_wrapper
            if self.sliding_window_size != -1:
                decode_wrapper = decode_wrapper[0]
            else:
                if isinstance(decode_wrapper, list):
                    decode_wrapper = decode_wrapper[1]

            if k is not None:
                assert v is not None
                self.store_kv_cache(k, v, input_metadata)

            retrieved_indices = input_metadata.token_retriever.retrieval_indices(q.contiguous(), self.layer_id, N_INIT,
                                                                                 N_Local, TOP_K)
            if retrieved_indices is not None:
                if (
                        self.layer_id == 0
                ):  # we only need to estimate the length of indices once
                    retrieved_indptr = decode_wrapper._paged_kv_indptr_buf.clone()
                    retrieved_indptr[1] = len(retrieved_indices)
                    kv_last_page_len = (
                        decode_wrapper._paged_kv_last_page_len_buf.clone()
                    )
                    decode_wrapper.end_forward()
                    decode_wrapper.begin_forward(
                        retrieved_indptr,
                        retrieved_indices,
                        kv_last_page_len,
                        self.tp_q_head_num,
                        self.tp_k_head_num,
                        self.head_dim,
                        1,
                    )
                else:
                    decode_wrapper._paged_kv_indices_buf.copy_(retrieved_indices)

            o = decode_wrapper.forward(
                q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                sm_scale=self.scaling,
                logits_soft_cap=self.logit_cap,
                rope_scale=ROPE_SCALE,
                rope_theta=ROPE_BASE,
                pos_encoding_mode=ROPE_MODE,
            )

            return o.view(-1, self.tp_q_head_num * self.head_dim)

        def patched_radix_attention_forward(
                self, q, k, v, input_metadata: InputMetadata
        ):
            if k is not None:
                assert v is not None
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

            if input_metadata.forward_mode == ForwardMode.EXTEND:
                return patched_radix_attention_extend_forward_flashinfer(
                    self, q, k, v, input_metadata
                )
            elif input_metadata.forward_mode == ForwardMode.DECODE:
                return patched_radix_attention_decode_forward_flashinfer(
                    self, q, k, v, input_metadata
                )

        def patched_radix_attention_store_kv_cache(
                self, cache_k, cache_v, input_metadata: InputMetadata, start=0, end=None
        ):
            k_cache = input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id)
            v_cache = input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id)
            k_cache[input_metadata.out_cache_loc[start:end]] = cache_k[start:end]
            v_cache[input_metadata.out_cache_loc[start:end]] = cache_v[start:end]
            input_metadata.token_retriever.add_k_cache(
                input_metadata.out_cache_loc[start:end],
                self.layer_id,
            )

        def patched_meta_attention_forward(
                self,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                input_metadata: InputMetadata,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            attn_output = self.attn(q, k, v, input_metadata)
            output, _ = self.o_proj(attn_output)
            return output

        for layer in model.model.layers:
            layer.self_attn.__class__.forward = (
                patched_meta_attention_forward  # support for Llama, Qwen2, Mistral
            )
            layer.self_attn.attn.__class__.forward = patched_radix_attention_forward
            layer.self_attn.attn.__class__.store_kv_cache = (
                patched_radix_attention_store_kv_cache
            )
        return model

    def patched_rotary_embedding_get_rope(*args, **kwargs) -> None:
        return None

    rotary_embedding.get_rope = patched_rotary_embedding_get_rope
    DefaultModelLoader.load_model = patched_default_model_loader_load_model


def patch(
        rope_base=1e6,
        rope_scale=1,
        rope_model="ROPE_LLAMA",
        max_n_tokens=1024,
        top_k=16,
        n_init=1,
        n_local=16,
        kernel_size=-1,
):
    global ROPE_BASE
    global ROPE_SCALE
    global ROPE_MODE
    global MAX_N_TOKENS
    global TOP_K
    global N_INIT
    global N_Local
    global QUERY_ROTATE
    global QUERY_CACHE
    global PREFILL_CHUNK_SIZE
    global KERNEL_SIZE

    ROPE_BASE = rope_base
    ROPE_SCALE = rope_scale
    ROPE_MODE = rope_model
    MAX_N_TOKENS = max_n_tokens
    TOP_K = top_k
    N_INIT = n_init
    N_Local = n_local
    KERNEL_SIZE=kernel_size

    QUERY_ROTATE = True
    PREFILL_CHUNK_SIZE = 512
    QUERY_CACHE = False

    patch_input_metadata()
    patch_model_runner()
    patch_model()