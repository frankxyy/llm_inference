import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase

from text_generation_server.models.types import Batch, GeneratedText
from text_generation_server.pb.generate_pb2 import InfoResponse

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        requires_padding: bool,
        dtype: torch.dtype,
        device: torch.device,
        decode_buffer: int = 3,
    ):
        if decode_buffer < 1:
            raise ValueError("decode_buffer must be >= 1")

        self.tokenizer = tokenizer
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.requires_padding = requires_padding
        self.dtype = dtype
        self.device = device
        self.decode_buffer = decode_buffer

    @property
    def info(self) -> InfoResponse:
        return InfoResponse(
            requires_padding=self.requires_padding,
            dtype=str(self.dtype),
            device_type=self.device.type,
        )

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError

    def decode_token(
        self,
        all_input_ids: List[int],
        offset: Optional[int] = None,
        token_offset: Optional[int] = None,
    ) -> Tuple[str, Optional[int], Optional[int]]:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        if all_input_ids[-1] in self.all_special_ids:
            return (
                self.tokenizer.decode(all_input_ids[-1], skip_special_tokens=False),
                None,
                None,
            )

        if token_offset is None:
            token_offset = len(all_input_ids) - self.decode_buffer
            # left token buffer
            if self.decode_buffer > 1:
                # Decode token_offset token minus last one and token_offset tokens
                raw_texts = self.tokenizer.batch_decode(
                    [all_input_ids[token_offset:-1], all_input_ids[token_offset:]],
                    skip_special_tokens=False,
                )

                # default offset is only the last token
                offset = len(raw_texts[0])
                sequence_text = raw_texts[1]
            else:
                # Only decode the last token without using a token buffer
                sequence_text = self.tokenizer.decode(
                    all_input_ids[-1], skip_special_tokens=False
                )
                # no offset in this case
                offset = 0
        else:
            assert offset is not None
            sequence_text = self.tokenizer.decode(
                all_input_ids[token_offset:],
                skip_special_tokens=False,
            )

        # get text
        token_text = sequence_text[offset:]

        # if text is utf-8
        if token_text and token_text[-1] != "�":
            return token_text, None, None
        else:
            return "", offset, token_offset
