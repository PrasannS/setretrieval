import torch
from torch import Tensor
from typing import Literal
from numpy import ndarray

# given normal output with vector shape [B, T, D], zero out all vectors that aren't in the first token for each item in batch
def newforward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
    outs = self.old_forward(input, **kwargs)
    mask = (input["input_ids"] == self._first_module().tokenizer.cls_token_id) & (input["attention_mask"] == 1)  # Shape: [batch_size, seq_len]
    # ignore cls token
    mask[:, 0] = False
    # breakpoint()
    mask = mask.unsqueeze(-1)
    # mask = torch.zeros_like(outs['token_embeddings'])
    # mask[:, 1, :] = 1   

    outs['token_embeddings'] = outs['token_embeddings'] * mask

    return outs

# use to monkey patch ColBERT to tokenize with pad tokens
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
        if self._first_module().max_seq_length in [512, 32768]: # this is to handle weirdness from prefix token
            self._first_module().max_seq_length = self._first_module().max_seq_length - 1
        # print(f"WARNING: HARD-CODED MAX SEQ LENGTH TO 511")
        # HACK ok I think this was usually padding to max length which was killing memory for qwen I'm assuming
        # need to fix this up a bit
        splitquant=0
        if "||||" in texts[0]:

            # for the set case (to circumvent annoying dataloader stuff)
            spltexts = [text.split("||||") for text in texts]
            splitquant = len(spltexts[0])
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

        if type(texts[0]) == dict:
            texts = [text['text'] for text in texts]


        ct = self._first_module().tokenizer.cls_token
        ct = ct if ct == "[EMB]" else " "+ct
        texts = [text + ct * addvecs for text in texts]


        # Tokenize the texts, let's generally not pad to max length here
        tokenized_outputs = self._first_module().tokenize(texts, padding="longest")

        # breakpoint()

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
        tokenized_outputs['splitquant'] = torch.tensor(splitquant)
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

def mod_encode(self, sentences: str | list[str], prompt_name: str | None = None, prompt: str | None = None, batch_size: int = 32, show_progress_bar: bool = None, precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32", convert_to_numpy: bool = True, convert_to_tensor: bool = False, padding: bool = False, device: str = None, normalize_embeddings: bool = True, is_query: bool = True, pool_factor: int = 1, protected_tokens: int = 1) -> list[torch.Tensor] | ndarray | torch.Tensor:
    embeddings = self.old_encode(sentences, prompt_name, prompt, batch_size, show_progress_bar, precision, convert_to_numpy, convert_to_tensor, padding, device, normalize_embeddings, is_query, pool_factor, protected_tokens)
    # identify tokens with zero vectors and get rid of them to get a new list of vectors for each item
    new_embeddings = []
    for embedding in embeddings:
        temb = torch.tensor(embedding)
        # identify tokens with zero vectors (all numbers in last dimension are zeros)
        zero_mask = temb.abs().sum(dim=-1) == 0
        # breakpoint()
        # get rid of them
        temb = temb[~zero_mask]
        new_embeddings.append(temb)
    # breakpoint()
    print(f"Embedding count: {len(new_embeddings[0])}")
    return new_embeddings