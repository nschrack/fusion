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
        text=None,
        amr=None,
        labels=None,
    ):

        num_choices = text.input_ids.shape[1] if text.input_ids is not None else None
        
        text.input_ids = text.input_ids.view(-1, text.input_ids.size(-1)) if text.input_ids is not None else None
        text.attention_mask = text.attention_mask.view(-1, text.attention_mask.size(-1)) if text.attention_mask is not None else None
        text.token_type_ids = text.token_type_ids.view(-1, text.token_type_ids.size(-1)) if text.token_type_ids is not None else None

        amr.input_ids = amr.input_ids.view(-1, amr.input_ids.size(-1)) if amr.input_ids is not None else None
        amr.attention_mask = amr.attention_mask.view(-1, amr.attention_mask.size(-1)) if amr.attention_mask is not None else None
        amr.token_type_ids = amr.token_type_ids.view(-1, amr.token_type_ids.size(-1)) if amr.token_type_ids is not None else None


        text_emb = self.text_model(
            text.input_ids, 
            token_type_ids = text.token_type_ids,
            attention_mask = text.attention_mask
        )[1] # pooled output

        amr_last_hidden_state = self.amr_model(
            amr.input_ids, 
            token_type_ids = amr.token_type_ids,
            attention_mask = amr.attention_mask
        )[0] # last hidden state

        eos_mask = amr.input_ids.eq(self.amr_eos_token_id)

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

        
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

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


 # load a local model 
        #path = '/some/local/path/pytorch/vision'
        #>>> model = torch.hub.load(path, 'resnet50', pretrained=True)
