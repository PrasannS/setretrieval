import torch
from torch import Tensor
from typing import Literal
from numpy import ndarray
import math
import copy
from tqdm.autonotebook import trange
import numpy as np
import logging

from scipy.cluster import hierarchy
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense as DenseSentenceTransformer
from sentence_transformers.models import Transformer
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.util import batch_to_device, load_file_path
from torch import nn
from tqdm.autonotebook import trange
from transformers.utils import cached_file

logger = logging.getLogger(__name__)

# given normal output with vector shape [B, T, D], zero out all vectors that aren't in the first token for each item in batch
def newforward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
    outs = self.old_forward(input, **kwargs)
    if self._first_module().tokenizer.query_vectors > 0:
        mask = (input["input_ids"] == self._first_module().tokenizer.cls_token_id) & (input["attention_mask"] == 1)  # Shape: [batch_size, seq_len]
        # ignore cls token
        mask[:, 0] = False
        # breakpoint()
        # only select token_embeddings from outs that are masked
        outs['token_embeddings'] = outs['token_embeddings'][mask].view(outs['token_embeddings'].size(0), -1, outs['token_embeddings'].size(-1))
    # print(outs['token_embeddings'].shape)
    return outs

# use to monkey patch ColBERT to tokenize with pad tokens
# cases to handle
# 1. default, in which case this function shouldn't get called
# 2. pad, in which case we add pad tokens and only these are used
# 3. pad + passive, in which case we add passive vectors, then pad tokens (only these are used for vectors)
# 4. default + passive, we add passive tokens. Both these and the pad tokens are used for vectors
def padded_tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        is_query: bool = True,
        pad: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenizes the input texts.

        Args:
            texts (Union[list[str], list[dict], list[tuple[str, str]]]): A list of texts to be tokenized.
            is_query (bool): Flag to indicate if the texts are queries. Defaults to True.
            pad (bool): Flag to indicate if elements should be padded to max length. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary of tensors with the tokenized texts, including "input_ids",
                "attention_mask", and optionally "token_type_ids".
        """
        # Set max sequence length based on whether the input is a query or document
        # max_length = self.query_length if is_query else self.document_length
        # # TODO this should be tokenizer max length
        if self._first_module().max_seq_length in [512, 8192, 32768]: # this is to handle weirdness from prefix token
            self._first_module().max_seq_length = self._first_module().max_seq_length - 1
        
        # print(f"WARNING: HARD-CODED MAX SEQ LENGTH TO 511")
        # HACK ok I think this was usually padding to max length which was killing memory for qwen I'm assuming
        # need to fix this up a bit
        if "||||" in texts[0]:

            # for the set case (to circumvent annoying dataloader stuff)
            spltexts = [text.split("||||") for text in texts]
            texts = [item for sublist in spltexts for item in sublist]

        # breakpoint()
        if self._first_module().tokenizer.cls_token is None:
            cls_token = '[EMB]'
            # breakpoint()
            # check if [EMB] is in added tokens
            if self._first_module().tokenizer.convert_tokens_to_ids(cls_token) is None:
                # in this case, add a special '[CLS]' token to the tokenizer, set it to be the cls_token_id
                self._first_module().tokenizer.cls_token_id = self._first_module().tokenizer.add_tokens(cls_token)
                self._first_module().tokenizer.cls_token = cls_token
                self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
            else:
                self._first_module().tokenizer.cls_token_id = self._first_module().tokenizer.convert_tokens_to_ids(cls_token)
                self._first_module().tokenizer.cls_token = cls_token

        # breakpoint()
        addvecs = self._first_module().tokenizer.query_vectors if is_query else self._first_module().tokenizer.doc_vectors
        passivevecs = self._first_module().tokenizer.qpass_vecs if is_query else self._first_module().tokenizer.dpass_vecs

        if type(texts[0]) == dict:
            texts = [text['text'] for text in texts]

        # filter stuff longer than 10k chars
        overcnt = sum([len(text) > 10000 for text in texts])
        if overcnt > 0:
            # print([len(text) for text in texts])
            # breakpoint()
            print(f"WARNING: {overcnt} texts were longer than 10k chars, filtering them out")
            texts = [text[:10000] for text in texts]
        

        # breakpoint()
        if passivevecs > 0:
            # add passive vectors to the end of the text, these should just allow for more computation
            texts = [text + " *" * passivevecs for text in texts]
        
        # breakpoint()

        # breakpoint()

        if addvecs > 0:
            ct = self._first_module().tokenizer.cls_token
            ct = ct if ct == "[EMB]" else " "+ct
            texts = [text + ct * addvecs for text in texts]

        # Tokenize the texts, let's generally not pad to max length here
        tokenized_outputs = self._first_module().tokenize(texts, padding="longest")

        tokenized_outputs['input_ids'] = tokenized_outputs['input_ids'][:, :self._first_module().max_seq_length]
        tokenized_outputs['attention_mask'] = tokenized_outputs['attention_mask'][:, :self._first_module().max_seq_length]
        if "token_type_ids" in tokenized_outputs:
            tokenized_outputs['token_type_ids'] = tokenized_outputs['token_type_ids'][:, :self._first_module().max_seq_length]

        # in rows that contain no padding, then replace last addvecs tokens with cls_tokens 
        if "bert-large-uncased" in self.tokenizer.name_or_path:
            nprows = [row[-1] != self._first_module().tokenizer.pad_token_id and row[-1] != self._first_module().tokenizer.sep_token_id for row in tokenized_outputs['input_ids']]
            for i in range(len(nprows)):
                if nprows[i]:
                    tokenized_outputs['input_ids'][i, -addvecs:] = self._first_module().tokenizer.cls_token_id
            if sum(nprows) > 0:
                print(f"{sum(nprows)} rows were truncated, adjusted their cls tokens")

        # Determine prefix ID based on input type
        prefix_id = self.query_prefix_id if is_query else self.document_prefix_id

        # add prefix id to every tokenized tensor in tokenized_outputs
        tokenized_outputs['input_ids'] = self.insert_prefix_token(tokenized_outputs['input_ids'], prefix_id)
        tokenized_outputs['attention_mask'] = self.insert_prefix_token(tokenized_outputs['attention_mask'], 1)

        # Update token type IDs if they exist
        if "token_type_ids" in tokenized_outputs:
            tokenized_outputs["token_type_ids"] = self.insert_prefix_token(tokenized_outputs['token_type_ids'], 0)

        # use this in loss function when needed
        # tokenized_outputs['splitquant'] = torch.tensor(splitquant)
        return tokenized_outputs

# for model-specific monkey patching
def modadj_tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], is_query: bool = True, pad: bool = False) -> dict[str, torch.Tensor]:
    tokenized_outputs = self.old_tokenize(texts, is_query=is_query, pad=pad)
    if "qwen" in self.tokenizer.name_or_path.lower():
        # swap indices 0 and 1 in all sequences
        tmp = tokenized_outputs['input_ids'][:, 0].clone()
        tokenized_outputs['input_ids'][:, 0] = tokenized_outputs['input_ids'][:, 1].clone()
        tokenized_outputs['input_ids'][:, 1] = tmp
    return tokenized_outputs

# def mod_encode(self, sentences: str | list[str], prompt_name: str | None = None, prompt: str | None = None, batch_size: int = 32, show_progress_bar: bool = None, precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32", convert_to_numpy: bool = True, convert_to_tensor: bool = False, padding: bool = False, device: str = None, normalize_embeddings: bool = True, is_query: bool = True, pool_factor: int = 1, protected_tokens: int = 1) -> list[torch.Tensor] | ndarray | torch.Tensor:
#     embeddings = self.old_encode(sentences, prompt_name, prompt, batch_size, show_progress_bar, precision, convert_to_numpy, convert_to_tensor, padding, device, normalize_embeddings, is_query, pool_factor, protected_tokens)
#     # identify tokens with zero vectors and get rid of them to get a new list of vectors for each item
#     new_embeddings = []
#     for embedding in embeddings:
#         temb = torch.tensor(embedding)
#         # identify tokens with zero vectors (all numbers in last dimension are zeros)
#         zero_mask = temb.abs().sum(dim=-1) == 0
#         # breakpoint()
#         # get rid of them
#         temb = temb[~zero_mask]
#         new_embeddings.append(temb)
#     # breakpoint()
#     print(f"Embedding count: {len(new_embeddings[0])}")
#     return new_embeddings

def mod_encode(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        padding: bool = False,
        device: str = None,
        normalize_embeddings: bool = True,
        is_query: bool = True,
        pool_factor: int = 1,
        protected_tokens: int = 1,
        usemask: bool = False,
    ) -> list[torch.Tensor] | ndarray | torch.Tensor:
        """
        Computes sentence embeddings.

        Parameters
        ----------
        sentences
            The sentences to embed.
        prompt_name
            The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary, which is either set in
            the constructor or loaded from the model configuration. For example, if `prompt_name` is "query" and the
            `prompts` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as
            "query: What is the capital of France?" because the sentence is appended to the prompt. If `prompt` is also
            set, this argument is ignored. Defaults to None.
        prompt
            The prompt to use for encoding. For example, if the prompt is "query: ", then the sentence "What is the capital
            of France?" will be encoded as "query: What is the capital of France?" because the sentence is appended to the
            prompt. If `prompt` is set, `prompt_name` is ignored. Defaults to None.
        batch_size
            The batch size used for the computation. Defaults to 32.
        show_progress_bar
            Whether to output a progress bar when encoding sentences. Defaults to None.
        output_value
            The type of embeddings to return: "sentence_embedding" to get sentence embeddings, "token_embeddings" to get
            wordpiece token embeddings, and `None` to get all output values. Defaults to "sentence_embedding".
        precision
            The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". All
            non-float32 precisions are quantized embeddings. Quantized embeddings are smaller in size and faster to compute,
            but may have lower accuracy. They are useful for reducing the size of the embeddings of a corpus for semantic
            search, among other tasks. Defaults to "float32".
        convert_to_numpy
            Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors. Defaults to True.
        convert_to_tensor
            Whether the output should be one large tensor. Overwrites `convert_to_numpy`. Defaults to False.
        device
            Which :class:`torch.device` to use for the computation. Defaults to None.
        normalize_embeddings
            Whether to normalize returned vectors to have length 1. In that case, the faster dot-product (util.dot_score)
            instead of cosine similarity can be used. Defaults to False.
        is_query
            Whether the input sentences are queries. If True, the query prefix is added to the input sentences and the
            sequence is padded; otherwise, the document prefix is added and the sequence is not padded. Defaults to True.
        pool_factor
            The factor by which to pool the document embeddings, resulting in 1/pool_factor of the original tokens. If set
            to 1, no pooling is done; if set to 2, 50% of the tokens are kept; if set to 3, 33%, and so on. Defaults to 1.
        protected_tokens
            The number of tokens at the beginning of the sequence that should not be pooled. Defaults to 1 (CLS token).

        """
        if isinstance(sentences, list):
            # If we have a list of list of sentences, we encode each list separately.
            if isinstance(sentences[0], list):
                embeddings = []

                for batch in sentences:
                    batch_embedings = self.encode(
                        sentences=batch,
                        prompt_name=prompt_name,
                        prompt=prompt,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                        precision=precision,
                        convert_to_numpy=convert_to_numpy,
                        convert_to_tensor=convert_to_tensor,
                        padding=padding,
                        device=device,
                        normalize_embeddings=normalize_embeddings,
                        is_query=is_query,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                    )

                    batch_embedings = (
                        torch.stack(batch_embedings)
                        if convert_to_tensor
                        else batch_embedings
                    )

                    embeddings.append(batch_embedings)

                return embeddings


        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        # Convert to tensor takes precedence over convert to numpy
        if not convert_to_numpy:
            convert_to_tensor = True
        convert_to_numpy = not convert_to_tensor

        # TODO: We cannot convert to tensor/numpy for token embeddings as they are not the same size
        # if output_value != "sentence_embedding":
        # convert_to_tensor = False
        # convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if prompt is not None and prompt_name is not None:
            logger.warning(
                "Provide either a `prompt` or a `prompt_name`, not both. "
                "Ignoring the `prompt_name` in favor of the provided `prompt`."
            )

        elif prompt is None:
            if prompt_name is not None:
                prompt = self.prompts.get(prompt_name)
                if prompt is None:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary. "
                        f"Available keys are: {list(self.prompts.keys())!r}."
                    )
            else:
                prompt = self.prompts.get(self.default_prompt_name)

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models require removing the prompt before pooling (e.g. Instructor, Grit).
            # Tracking the prompt length allow us to remove the prompt during pooling.
            tokenized_prompt = self.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = (
                    tokenized_prompt["input_ids"].shape[-1] - 1
                )

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc=f"Encoding queries (bs={batch_size})"
            if is_query
            else f"Encoding documents (bs={batch_size})",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(texts=sentences_batch, is_query=is_query)

            if self.device.type == "hpu":
                if "input_ids" in features:
                    curr_tokenize_len = features["input_ids"].shape

                    additional_pad_len = (
                        2 ** math.ceil(math.log2(curr_tokenize_len[1]))
                        - curr_tokenize_len[1]
                    )

                    features["input_ids"] = torch.cat(
                        tensors=(
                            features["input_ids"],
                            torch.ones(
                                size=(curr_tokenize_len[0], additional_pad_len),
                                dtype=torch.int8,
                            ),
                        ),
                        dim=-1,
                    )

                    features["attention_mask"] = torch.cat(
                        tensors=(
                            features["attention_mask"],
                            torch.zeros(
                                size=(curr_tokenize_len[0], additional_pad_len),
                                dtype=torch.int8,
                            ),
                        ),
                        dim=-1,
                    )

                    if "token_type_ids" in features:
                        features["token_type_ids"] = torch.cat(
                            tensors=(
                                features["token_type_ids"],
                                torch.zeros(
                                    size=(curr_tokenize_len[0], additional_pad_len),
                                    dtype=torch.int8,
                                ),
                            ),
                            dim=-1,
                        )

            features = batch_to_device(batch=features, target_device=device)
            features.update(extra_features)

            with torch.no_grad():
                # TODO: add the truncate/sliding window logic here
                out_features = self.forward(input=features)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                if not is_query:
                    # Compute the mask for the skiplist (punctuation symbols)
                    skiplist_mask = self.skiplist_mask(
                        input_ids=features["input_ids"], skiplist=self.skiplist
                    )
                    masks = torch.logical_and(
                        input=skiplist_mask, other=out_features["attention_mask"]
                    )
                else:
                    if self.do_query_expansion:
                        # We keep all tokens in the query (no skiplist) and we do not want to prune expansion tokens in queries even if we do not attend to them in attention layers
                        masks = torch.ones_like(
                            input=out_features["input_ids"], dtype=torch.bool
                        )
                    else:
                        # We only keep the original tokens and prune padding tokens
                        masks = out_features["attention_mask"].bool()

                embeddings = []
                for (
                    token_embedding,
                    mask,
                ) in zip(out_features["token_embeddings"], masks):
                    if usemask:
                        token_embedding = (
                            torch.nn.functional.normalize(
                                input=token_embedding[mask], p=2, dim=1
                            )
                            if normalize_embeddings
                            else token_embedding[mask]
                        )
                    else:
                        token_embedding = (
                            torch.nn.functional.normalize(
                                input=token_embedding, p=2, dim=1
                            )
                            if normalize_embeddings
                            else token_embedding
                        )
                    embeddings.append(token_embedding)

                # Pool factor must be greater than 1: keeping 1 over pool_factor tokens embeddings
                if pool_factor > 1 and not is_query:
                    embeddings = self.pool_embeddings_hierarchical(
                        documents_embeddings=embeddings,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                    )

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = [embedding.cpu() for embedding in embeddings]

                all_embeddings.extend(embeddings)

        # Pad the embeddings to the same length. Documents can have different lengths while queries are already padded (when using query expansion, else requires padding as well).
        if padding:
            all_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences=all_embeddings, batch_first=True, padding_value=0
            )

            # Create a list of tensors.
            all_embeddings = torch.split(
                tensor=all_embeddings, split_size_or_sections=1, dim=0
            )

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(
                embeddings=all_embeddings, precision=precision
            )

        # Return a list of arrays instead of single contiguous array since documents can have different lengths.
        if convert_to_tensor:
            if not len(all_embeddings):
                return torch.tensor()

            if isinstance(all_embeddings, np.ndarray):
                all_embeddings = [
                    torch.from_numpy(ndarray=embedding) for embedding in all_embeddings
                ]

        elif convert_to_numpy:
            bloat = all_embeddings[0].dtype == torch.bfloat16
            all_embeddings = [
                embedding.float().numpy() if bloat else embedding.numpy()
                for embedding in all_embeddings
            ]

        return all_embeddings[0] if input_was_string else all_embeddings