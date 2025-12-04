import torch
import torch.nn as nn
from transformers import AutoModel


class FinbertHAN(nn.Module):
    """
    Leg 2: HAN + Recency Decay + Auxiliary Features
    Architecture adapted for speed using distilroberta
    1. Frozen distilroberta -> sentence embeddings
    2. Bi-GRU -> document embeddings
    3. Time-aware attention -> weighted document vector
    4. Concat(document vector, aux features) -> regressor -> return prediction
    """

    def __init__(self, 
                 gru_hidden_dim=100, 
                 dropout_rate=0.1, 
                 bert_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                 aux_dim=4): # Novelty + velocity + 2 flags
        super().__init__()

        # Layer 1: Transfer learning
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name)
        except OSError:
            print(f"Could not load {bert_model_name}. Using fallback.")
            self.bert = AutoModel.from_pretrained('distilroberta-base')

        # Optimization: freeze transformer weights
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_output_dim = self.bert.config.hidden_size

        # Layer 2: Bi-GRU is document encoding
        # This will get the narrative flow and context between sentences
        self.doc_encoder = nn.GRU(
            input_size=bert_output_dim,
            hidden_size=gru_hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # Layer 3: Time aware attention (Pooling)
        # Attention mechanism to filter noise
        rnn_output_dim = gru_hidden_dim * 2
        self.attention_layer = nn.Linear(rnn_output_dim, rnn_output_dim)
        self.u_context = nn.Parameter(torch.randn(rnn_output_dim))
        
        # Recency-weighted average or light time-decay
        self.time_decay_tau = nn.Parameter(torch.tensor([0.1]))

        # Layer 4: Regressor prediction
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_output_dim + aux_dim, 1)
        )

    def forward(self, input_ids, attention_mask, doc_lengths, time_gaps, aux_features, news_mask):
        # 1. distilroberta pass
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooling strategy: distilroberta does not have a pooler_output
        # We take the s token at index 0 which acts as the sentence embedding
        sent_embeds = bert_out.last_hidden_state[:, 0, :]
        
        # 2. Reconstruct document structure
        # Turns flat list of sentences back into [Batch, Max_Sentences, Dim]
        batch_docs, mask_padding = self._unflatten_docs(sent_embeds, doc_lengths)
        
        # 3. Bi-GRU pass: Understand narrative flow
        doc_flow, _ = self.doc_encoder(batch_docs)
        
        # 4. Attention mechanism
        # Calculate raw importance scores based on content u_it * u_context
        u_it = torch.tanh(self.attention_layer(doc_flow))
        scores = torch.matmul(u_it, self.u_context)
        
        # Apply time decay: subtract (tau * hours_old) from score
        # Older news gets a lower attention score
        decay_term = torch.abs(self.time_decay_tau) * time_gaps
        scores = scores - decay_term
        
        # Mask out padding sentences so they don't affect softmax
        scores = scores.masked_fill(mask_padding == 0, -1e9)
    
        # Handle fully masked rows aka empty docs to avoid NaN in softmax
        is_all_masked = (mask_padding == 0).all(dim=1, keepdim=True)
        alpha = torch.softmax(scores, dim=1)
        alpha = alpha.masked_fill(is_all_masked.expand_as(alpha), 0.0)
    
        # 5. Weighted sum: Create final document vector
        doc_summary = torch.matmul(alpha.unsqueeze(1), doc_flow).squeeze(1)
        
        # 6. Feature fusion: Add velocity, novelty, and flags
        combined_features = torch.cat([doc_summary, aux_features], dim=1)
        
        # 7. Regress to scalar alpha
        prediction = self.regressor(combined_features)
        
        # If no headlines: set z_news = 0
        prediction = prediction * news_mask
        
        return prediction, alpha, news_mask

    def _unflatten_docs(self, flat_embeds, lengths):
        """
        Helper to reshape flat sentence embeddings into batch format
        Handles variable length documents via padding
        """
        batch_size = len(lengths)
        max_len = max(1, max(lengths)) if lengths else 1
        embed_dim = flat_embeds.shape[1] if flat_embeds.shape[0] > 0 else self.bert.config.hidden_size
        device = flat_embeds.device if flat_embeds.shape[0] > 0 else torch.device('cpu')
        
        padded_docs = torch.zeros(batch_size, max_len, embed_dim, device=device)
        mask = torch.zeros(batch_size, max_len, device=device)
        
        start_idx = 0
        for i, length in enumerate(lengths):
            end_idx = start_idx + length
            if length > 0:
                padded_docs[i, :length] = flat_embeds[start_idx:end_idx]
                mask[i, :length] = 1 
            start_idx = end_idx
            
        return padded_docs, mask