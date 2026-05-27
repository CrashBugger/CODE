import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []

def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    activated_key = act_fn(key)
    activated_query = act_fn(query)

    # Reuse the activated tensor when possible so we do not keep an extra
    # full-sized difference tensor alive on top of the two activations.
    if activated_key.data_ptr() != key.data_ptr():
        activated_key.sub_(activated_query)
        diff = activated_key
    else:
        diff = activated_key - activated_query

    l2_norm = torch.norm(diff, dim=-1)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        mask = act_mask.to(device=l2_norm.device, dtype=l2_norm.dtype)
        return torch.sum(l2_norm * mask, dim=1) / torch.sum(mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)

class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(WISE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        if type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            print(f"New weights successfully inserted into {layer}")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):
        if not self.config.retrieve:
            if hasattr(self.get_adapter_layer(), 'editing') and not self.get_adapter_layer().editing:
                # final merge
                if not self.get_adapter_layer().original_layer.weight.equal(self.get_adapter_layer().new_weight) and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    self.get_adapter_layer().memory_weight.append(self.get_adapter_layer().new_weight)
                if len(self.get_adapter_layer().memory_weight) > 0 and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    print('length of memory is ', len(self.get_adapter_layer().memory_weight), '!!!!!!')
                    self.get_adapter_layer().merge_weight()
        return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer.to(self.model.device)

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def _forward_for_edit(self, model_inputs):
        # WISE computes its own objective from logits/activations, so avoid the
        # model-internal LM loss path and KV-cache allocation during edit-time.
        forward_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
        return self.model(**forward_inputs, use_cache=False)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            optimizer.zero_grad()
            ft_loss, act_loss = self._backward_edit_losses_by_request(
                tokens,
                last_prompt_token_loc,
                act_mask=act_mask,
                deact_mask=deact_mask,
            )
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                optimizer.zero_grad()
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                self._clear_cached_layer_outputs()
                break
            if i == config.n_iter - 1:
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            neg_memo_loss = ft_loss.new_zeros(())
            pos_memo_loss = ft_loss.new_zeros(())
            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self._forward_for_edit(memo_input)

                    memory_act_loss = self._cal_memory_neg_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_act_loss = memory_act_loss.to(ft_loss.device)
                    memory_act_loss.backward()
                    memory_loss.append(memory_act_loss.detach())
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self._forward_for_edit(memo_input)

                    pos_memo_loss = self._cal_memory_pos_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    pos_memo_loss = pos_memo_loss.to(ft_loss.device)
                    pos_memo_loss.backward()
                    del memo_input
                    loss += pos_memo_loss
            # for replay Appendix B.3
            self.get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self._norm_constraint(self.config.norm_constraint)

            self._clear_cached_layer_outputs()
            del ft_loss, act_loss, loss
            torch.cuda.empty_cache()

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        #
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def _get_forward_chunk_size(self, total_rows):
        chunk_size = getattr(self.config, "forward_chunk_size", 0)
        if chunk_size is None or chunk_size <= 0:
            return total_rows
        return min(chunk_size, total_rows)

    def _slice_batch_tensors(self, batch_tensors, start, end):
        return {key: value[start:end] for key, value in batch_tensors.items()}

    def _clear_cached_layer_outputs(self, adapter_layer=None):
        adapter_layer = adapter_layer or self.get_adapter_layer()
        adapter_layer.original_layer_output = None
        adapter_layer.new_weight_layer_output = None
        torch.cuda.empty_cache()

    def _positive_mean(self, values):
        positive_values = values[values > 0]
        if positive_values.numel() == 0:
            return values.new_zeros(())
        return torch.mean(positive_values)

    def _combine_activation_terms(self, in_scope_dist, out_scope_dist, config, device):
        loss = out_scope_dist.view(-1, 1) - in_scope_dist + config.gamma
        loss2 = out_scope_dist - config.alpha
        loss3 = config.beta - in_scope_dist
        return (
            self._positive_mean(loss3).to(device)
            + self._positive_mean(loss2).to(device)
            + self._positive_mean(loss).to(device)
        )

    def _compute_sample_activation_loss(
        self,
        prompt_original_output,
        prompt_new_output,
        config,
        act_mask,
        deact_mask,
        locality_original_output=None,
        locality_new_output=None,
    ):
        if act_mask is not None:
            in_scope_dist = euc(
                prompt_original_output,
                prompt_new_output,
                config,
                act_mask=act_mask,
            )
            out_scope_dist = euc(
                prompt_original_output,
                prompt_new_output,
                config,
                act_mask=deact_mask,
            )
        else:
            in_scope_dist = euc(prompt_original_output, prompt_new_output, config)
            out_scope_dist = euc(locality_original_output, locality_new_output, config)

        return self._combine_activation_terms(
            in_scope_dist,
            out_scope_dist,
            config,
            prompt_original_output.device,
        )

    def _get_edit_layout(self, tokens):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1

        total_rows = tokens["input_ids"].shape[0]
        bs = total_rows - k
        if bs % k != 0:
            raise ValueError("WISE expects the prompt rows to be evenly divisible by batch size.")

        len_temp = bs // k
        return k, total_rows, bs, len_temp

    def _compute_request_edit_loss(
        self,
        tokens,
        last_prompt_token_loc,
        request_idx,
        act_mask=None,
        deact_mask=None,
    ):
        k, total_rows, bs, len_temp = self._get_edit_layout(tokens)
        chunk_size = self._get_forward_chunk_size(total_rows)
        adapter_layer = self.get_adapter_layer()

        prompt_start = request_idx * len_temp
        prompt_end = (request_idx + 1) * len_temp
        locality_row_idx = bs + request_idx

        prompt_loss_sum = None
        prompt_original_outputs = []
        prompt_new_outputs = []
        editing_activation_sum = 0.0
        editing_activation_count = 0

        for start in range(prompt_start, prompt_end, chunk_size):
            end = min(start + chunk_size, prompt_end)
            chunk_tokens = self._slice_batch_tensors(tokens, start, end)
            logits = self._forward_for_edit(chunk_tokens).logits

            original_output = adapter_layer.original_layer_output
            new_output = adapter_layer.new_weight_layer_output
            prompt_original_outputs.append(original_output)
            prompt_new_outputs.append(new_output)

            chunk_row_losses = self._compute_masked_ft_loss(
                logits,
                chunk_tokens['labels'],
                last_prompt_token_loc[start:end],
            )
            prompt_loss_sum = (
                chunk_row_losses.sum()
                if prompt_loss_sum is None
                else prompt_loss_sum + chunk_row_losses.sum()
            )

            activation_dist = euc(original_output, new_output, self.config).detach()
            editing_activation_sum += activation_dist.sum().item()
            editing_activation_count += activation_dist.numel()

        locality_tokens = self._slice_batch_tensors(tokens, locality_row_idx, locality_row_idx + 1)
        self._forward_for_edit(locality_tokens)
        locality_original_output = adapter_layer.original_layer_output
        locality_new_output = adapter_layer.new_weight_layer_output

        if locality_row_idx < total_rows - 1:
            locality_activation_dist = euc(locality_original_output, locality_new_output, self.config).detach()
            editing_activation_sum += locality_activation_dist.sum().item()
            editing_activation_count += locality_activation_dist.numel()

        prompt_original_output = torch.cat(prompt_original_outputs, dim=0)
        prompt_new_output = torch.cat(prompt_new_outputs, dim=0)
        sample_act_mask = act_mask[request_idx] if act_mask is not None else None
        sample_deact_mask = deact_mask[request_idx] if deact_mask is not None else None
        sample_act_loss = self._compute_sample_activation_loss(
            prompt_original_output,
            prompt_new_output,
            config=self.config,
            act_mask=sample_act_mask,
            deact_mask=sample_deact_mask,
            locality_original_output=locality_original_output,
            locality_new_output=locality_new_output,
        )

        scaled_ft_loss = prompt_loss_sum / bs
        scaled_act_loss = sample_act_loss / k
        request_loss = scaled_ft_loss + scaled_act_loss

        return (
            request_loss,
            scaled_ft_loss.detach(),
            scaled_act_loss.detach(),
            editing_activation_sum,
            editing_activation_count,
        )

    def _backward_edit_losses_by_request(self, tokens, last_prompt_token_loc, act_mask=None, deact_mask=None):
        k, _, _, _ = self._get_edit_layout(tokens)
        adapter_layer = self.get_adapter_layer()

        ft_loss_total = None
        act_loss_total = None
        editing_activation_sum = 0.0
        editing_activation_count = 0

        for request_idx in range(k):
            request_loss, ft_loss_part, act_loss_part, act_sum, act_count = self._compute_request_edit_loss(
                tokens,
                last_prompt_token_loc,
                request_idx,
                act_mask=act_mask,
                deact_mask=deact_mask,
            )
            request_loss.backward()

            ft_loss_total = ft_loss_part if ft_loss_total is None else ft_loss_total + ft_loss_part
            act_loss_total = act_loss_part if act_loss_total is None else act_loss_total + act_loss_part
            editing_activation_sum += act_sum
            editing_activation_count += act_count

            self._clear_cached_layer_outputs(adapter_layer)
            del request_loss

        adapter_layer.last_editing_activation_value = (
            editing_activation_sum / editing_activation_count if editing_activation_count > 0 else None
        )

        return ft_loss_total, act_loss_total

    def _compute_chunk_edit_losses(self, tokens, last_prompt_token_loc, act_mask=None, deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1

        total_rows = tokens["input_ids"].shape[0]
        bs = total_rows - k
        if bs % k != 0:
            raise ValueError("WISE expects the prompt rows to be evenly divisible by batch size.")

        len_temp = bs // k
        chunk_size = self._get_forward_chunk_size(total_rows)
        adapter_layer = self.get_adapter_layer()

        ft_losses = []
        total_act_loss = []
        locality_outputs = [None] * k
        editing_activation_sum = 0.0
        editing_activation_count = 0

        for start in range(bs, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk_tokens = self._slice_batch_tensors(tokens, start, end)
            self._forward_for_edit(chunk_tokens)

            original_output = adapter_layer.original_layer_output
            new_output = adapter_layer.new_weight_layer_output
            activation_dist = euc(original_output, new_output, self.config).detach()

            for row_offset in range(end - start):
                global_row_idx = start + row_offset
                locality_idx = global_row_idx - bs
                locality_outputs[locality_idx] = (
                    original_output[row_offset:row_offset + 1],
                    new_output[row_offset:row_offset + 1],
                )
                if global_row_idx < total_rows - 1:
                    editing_activation_sum += activation_dist[row_offset].item()
                    editing_activation_count += 1

        prompt_original_buffers = [[] for _ in range(k)]
        prompt_new_buffers = [[] for _ in range(k)]

        for start in range(0, bs, chunk_size):
            end = min(start + chunk_size, bs)
            chunk_tokens = self._slice_batch_tensors(tokens, start, end)
            logits = self._forward_for_edit(chunk_tokens).logits

            original_output = adapter_layer.original_layer_output
            new_output = adapter_layer.new_weight_layer_output
            activation_dist = euc(original_output, new_output, self.config).detach()
            editing_activation_sum += activation_dist.sum().item()
            editing_activation_count += activation_dist.numel()

            ft_losses.append(
                self._compute_masked_ft_loss(
                    logits,
                    chunk_tokens['labels'],
                    last_prompt_token_loc[start:end],
                )
            )

            cursor = start
            while cursor < end:
                sample_idx = cursor // len_temp
                sample_end = min(end, (sample_idx + 1) * len_temp)
                rel_start = cursor - start
                rel_end = sample_end - start

                prompt_original_buffers[sample_idx].append(original_output[rel_start:rel_end])
                prompt_new_buffers[sample_idx].append(new_output[rel_start:rel_end])

                if sample_end == (sample_idx + 1) * len_temp:
                    prompt_original_output = torch.cat(prompt_original_buffers[sample_idx], dim=0)
                    prompt_new_output = torch.cat(prompt_new_buffers[sample_idx], dim=0)
                    locality_original_output, locality_new_output = locality_outputs[sample_idx]
                    sample_act_mask = act_mask[sample_idx] if act_mask is not None else None
                    sample_deact_mask = deact_mask[sample_idx] if deact_mask is not None else None
                    total_act_loss.append(
                        self._compute_sample_activation_loss(
                            prompt_original_output,
                            prompt_new_output,
                            config=self.config,
                            act_mask=sample_act_mask,
                            deact_mask=sample_deact_mask,
                            locality_original_output=locality_original_output,
                            locality_new_output=locality_new_output,
                        )
                    )
                    prompt_original_buffers[sample_idx].clear()
                    prompt_new_buffers[sample_idx].clear()

                cursor = sample_end

        if any(buffer for buffer in prompt_original_buffers) or any(buffer for buffer in prompt_new_buffers):
            raise RuntimeError("WISE activation buffers were not fully consumed during chunked loss computation.")

        adapter_layer.last_editing_activation_value = (
            editing_activation_sum / editing_activation_count if editing_activation_count > 0 else None
        )

        return torch.cat(ft_losses, dim=0).mean(), sum(total_act_loss) / len(total_act_loss)

    def _compute_masked_ft_loss(self, logits, labels, last_prompt_token_loc):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)
        for i, col_index in enumerate(last_prompt_token_loc):
            label_mask[i, col_index - 1:] = True

        return (loss * label_mask).sum(1) / label_mask.sum(1)

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        total_rows = tokens["input_ids"].shape[0]
        chunk_size = self._get_forward_chunk_size(total_rows)
        adapter_layer = self.get_adapter_layer()

        if chunk_size >= total_rows:
            logits = self._forward_for_edit(tokens).logits
            ft_loss = self._compute_masked_ft_loss(
                logits[:-k],
                tokens['labels'][:-k],
                last_prompt_token_loc[:-k],
            ).mean()
            return ft_loss

        ft_losses = []
        original_layer_outputs = []
        new_weight_layer_outputs = []

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk_tokens = self._slice_batch_tensors(tokens, start, end)
            logits = self._forward_for_edit(chunk_tokens).logits

            original_layer_outputs.append(adapter_layer.original_layer_output)
            new_weight_layer_outputs.append(adapter_layer.new_weight_layer_output)

            ft_end = min(end, bs)
            if start < ft_end:
                relative_end = ft_end - start
                chunk_ft_loss = self._compute_masked_ft_loss(
                    logits[:relative_end],
                    chunk_tokens['labels'][:relative_end],
                    last_prompt_token_loc[start:ft_end],
                )
                ft_losses.append(chunk_ft_loss)

        adapter_layer.original_layer_output = torch.cat(original_layer_outputs, dim=0)
        adapter_layer.new_weight_layer_output = torch.cat(new_weight_layer_outputs, dim=0)

        return torch.cat(ft_losses, dim=0).mean()

    def _cal_edit_losses(self, tokens, last_prompt_token_loc, act_mask=None, deact_mask=None):
        total_rows = tokens["input_ids"].shape[0]
        chunk_size = self._get_forward_chunk_size(total_rows)

        if chunk_size >= total_rows:
            ft_loss = self._cal_ft_loss(tokens, last_prompt_token_loc)
            act_loss = self._cal_activation_loss(
                self.get_adapter_layer().original_layer_output,
                self.get_adapter_layer().new_weight_layer_output,
                config=self.config,
                act_mask=act_mask,
                deact_mask=deact_mask,
            )
            return ft_loss, act_loss

        return self._compute_chunk_edit_losses(
            tokens,
            last_prompt_token_loc,
            act_mask=act_mask,
            deact_mask=deact_mask,
        )

    def _cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        len_temp = original_layer_output.shape[0] / k - 1
        for i,act_mk in enumerate(act_mask):
            if act_mk is not None:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=act_mk)
                out_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config)
                if (i==k-1):
                    out_scope_dist = euc(original_layer_output[int(i-k):, ...], new_weight_layer_output[int(i-k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i-k):int(i+1-k), ...], new_weight_layer_output[int(i-k):int(i+1-k), ...], config)

            loss = out_scope_dist.view(-1,1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
        return sum(total_loss) / len(total_loss)

    def _cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def _cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {
            'memory_weight': self.get_adapter_layer().memory_weight,
            'memory_mean_act': self.get_adapter_layer().memory_mean_act,
            'merge_cnt': self.get_adapter_layer().merge_cnt,
            'editing_mean_act': self.get_adapter_layer().editing_mean_act,
            'editing_total_cnt': self.get_adapter_layer().editing_total_cnt,
            'weight_mask': self.get_adapter_layer().weight_mask,
            # Add other variables that need to be saved
        }
        if hasattr(self.get_adapter_layer(), 'key_id') and self.get_adapter_layer().key_id is not None:
            additional_info['key_id'] = self.get_adapter_layer().key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': self.get_adapter_layer().state_dict(),
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded WISE config is different from the original config")

        # Restore the state dictionary of the WISE Adapter instance
        self.get_adapter_layer().load_state_dict(saved_data['adapter_state_dict'])
        # Restore additional information
        adapter_layer = self.get_adapter_layer()
        for key, value in saved_data['additional_info'].items():
            setattr(adapter_layer, key, value)
        
        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and WISE state loaded from {load_path}")



class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.original_layer.to('cpu')
        self.memory_weight = []
        self.memory_weight_device = 'cpu'
        self.memory_mean_act = []
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias # For Conv1D
        else:
            self.bias = None
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None 

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0
        self.last_editing_activation_value = None

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        weight_to_save = copy.deepcopy(self.new_weight).to('cpu')
        self.memory_weight.append(weight_to_save)
        self.new_weight = copy.deepcopy(self.original_layer.weight).to(self.device)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                original_weight_gpu = self.original_layer.weight.to(self.device)
                memory_weights_gpu = [w.to(self.device) for w in self.memory_weight]
                if original_weight_gpu.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute([self.config.weights / len(memory_weights_gpu) for _ in range(len(memory_weights_gpu))], original_weight_gpu, memory_weights_gpu, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute([0.4 / len(memory_weights_gpu) for _ in range(len(memory_weights_gpu))] + [0.6], original_weight_gpu, memory_weights_gpu + [self.layer.weight], densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(original_weight_gpu)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                original_weight_gpu = self.original_layer.weight.to(self.device)
                memory_weights_gpu = [w.to(self.device) for w in self.memory_weight[-merge_num:]]
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)], original_weight_gpu, memory_weights_gpu, densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(original_weight_gpu)
                self.memory_weight.append(new_merge_weight.to('cpu'))
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            original_weight_gpu = self.original_layer.weight.to(self.device)
            self.new_weight = copy.deepcopy(original_weight_gpu)

    def save_editing_activation(self):
        if self.last_editing_activation_value is not None:
            self.editing_mean_act.update(self.last_editing_activation_value)
            self.last_editing_activation_value = None
            return

        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # Get indices of unmasked elements
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # Update mask state
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias, input.view(-1, input.size(-1)), self.new_weight).view(input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            with torch.no_grad():
                self.original_layer.to(self.device)
                self.original_layer_output = self.original_layer(*args)
                self.original_layer.to('cpu')
        else:
            if not self.config.retrieve:
                self.original_layer.to(self.device)
                original_layer_output = self.original_layer(*args)
                self.original_layer.to('cpu')
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                if dist1.item() < threshold and dist2.item() < threshold:
                    layer_out = original_layer_output
                elif dist1.item() > dist2.item():
                    layer_out = layer_output
                else:
                    layer_out = new_weight_layer_output
            else:
                self.original_layer.to(self.device)
                original_layer_output = self.original_layer(*args)
                self.original_layer.to('cpu')
                new_weight_layer_output = self.new_weight_forward(*args)
                dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                min_dist = dist1
                if min_dist.dim() > 0:  
                    min_dist = min_dist.mean()
                if min_dist.item() < threshold:
                    layer_out = original_layer_output
                else:
                    layer_out = new_weight_layer_output

                for i in range(len(self.memory_weight)):
                    memory_retrieve_weight = self.memory_weight[i].to(self.device)
                    memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                    dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True)
                    if dist > min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:
                        layer_out = memory_weight_layer_output
                        min_dist = dist
        return layer_out


class WISEMultimodal(WISE):
    def edit(self, config, multimodal_inputs, text_tokens, ans_token_len, act_mask=None, deact_mask=None):
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in text_tokens.items()}, False])
        last_prompt_token_loc = (text_tokens["labels"] == -100).sum(dim=-1) - 1
        
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)        
        
        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([super().get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len)

            act_loss = super()._cal_activation_loss(super().get_adapter_layer().original_layer_output, super().get_adapter_layer().new_weight_layer_output,
                                                  config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                super().get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                super().get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self._forward_for_edit(memo_input)

                    memory_act_loss = super()._cal_memory_neg_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self._forward_for_edit(memo_input)

                    pos_memo_loss = super()._cal_memory_pos_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            super().get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                super()._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            super().get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            super().get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _cal_ft_loss(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        
        if k != 1:
            raise AssertionError("Not support Batch Edit")
        
        bs = text_tokens["input_ids"].shape[0] - k
        logits = self._forward_for_edit(multimodal_inputs).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = multimodal_inputs['input_ids'][:-k, 1:].contiguous()
        # only cal loss of target text tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        a = shift_logits.view(-1, shift_logits.size(-1))
        b = shift_labels.view(-1)[-ans_token_len:]
        a = a[-b.size(0):,:]
        loss = loss_fct(a, b)
        loss = loss.view(bs, -1)
        label_mask = torch.ones_like(loss, dtype=torch.bool)        
        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss
    
