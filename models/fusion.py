from cgitb import text
import torch
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput


class Fusion(nn.Module):
    """Fusion Model"""

    def __init__(
        self, 
        text_model: nn.Module,
        amr_model: nn.Module,
        concat_emb_dim: int,
        classifier_dropout: float,
        amr_eos_token_id: str
    ):
        super().__init__()
        self.amr_eos_token_id = amr_eos_token_id
        self.text_model = text_model
        self.amr_model = amr_model
        
        self.classification_head = BartClassificationHead(
            concat_emb_dim,
            concat_emb_dim,
            1,
            classifier_dropout,
        )

    def forward(
        self, 
        text_input_ids=None,
        text_attention_mask=None,
        text_token_type_ids=None,
        amr_input_ids=None,
        amr_attention_mask=None,
        labels=None
    ):
        num_choices = text_input_ids.shape[1] if text_input_ids is not None else None
        
        text_input_ids = text_input_ids.view(-1, text_input_ids.size(-1)) if text_input_ids is not None else None
        text_attention_mask = text_attention_mask.view(-1, text_attention_mask.size(-1)) if text_attention_mask is not None else None
        text_token_type_ids = text_token_type_ids.view(-1, text_token_type_ids.size(-1)) if text_token_type_ids is not None else None

        amr_input_ids = amr_input_ids.view(-1, amr_input_ids.size(-1)) if amr_input_ids is not None else None
        amr_attention_mask = amr_attention_mask.view(-1, amr_attention_mask.size(-1)) if amr_attention_mask is not None else None


        text_emb = self.text_model(
            text_input_ids, 
            token_type_ids = text_token_type_ids,
            attention_mask = text_attention_mask
        )[1] # pooled output

        amr_last_hidden_state = self.amr_model(
            amr_input_ids, 
            attention_mask = amr_attention_mask
        )[0] # last hidden state

        eos_mask = amr_input_ids.eq(self.amr_eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        amr_emb = amr_last_hidden_state[eos_mask, :].view(amr_last_hidden_state.size(0), -1, amr_last_hidden_state.size(-1))[
            :, -1, :
        ]

        concat_emb = torch.cat((text_emb, amr_emb), dim=1)
        logits = self.classification_head(concat_emb)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return loss, reshaped_logits

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
