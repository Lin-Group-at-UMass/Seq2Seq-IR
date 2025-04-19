from src.models import  LSTM_autoregressive, GPT, GRU_test, Transformer
from pytorch_lightning import seed_everything
from torch import nn
import torch
import torch.nn.functional as F

class LSTM_autoregressive_topk(LSTM_autoregressive):
    def decode_step(self, token, hidden, cell):
        """
        Single step decoding:
         - Input: token: Tensor of shape (1, 1) (current token, index is int)
                 hidden, cell: Current LSTM state, both with shape (num_layers, 1, hidden_dim)
         - Processing: First pass the token through embedding and dropout to get embedded, then call forward_step to get the output at the current time step,
                   then pass through dropout, linear mapping, log_softmax to get the log probability distribution
         - Return: logits (shape (1, num_classes)), new_hidden, new_cell
        """
        # token: (1, 1)
        embedded = self.embedding(token)         # shape: (1, 1, hidden_dim)
        output, new_hidden, new_cell = self.forward_step(embedded, hidden, cell)
        output = self.linear2(output)              # shape: (1, 1, num_classes)
        logits = F.log_softmax(output, dim=-1)       # shape: (1, 1, num_classes)
        # Call detach() on the new hidden and cell to cut off gradient propagation
        new_hidden = new_hidden.detach()
        new_cell = new_cell.detach()
        return logits[:, -1, :], new_hidden, new_cell  # Return shape: (1, num_classes)

    def predict_step(self, batch, batch_idx, beam_width=5):
        seed = 78438379
        deterministic = True
        seed_everything(seed, workers=deterministic)
        """
        For each input spectrum sample, use beam search to generate top-k candidate token sequences.
        Returns a list, where each element corresponds to the beam search result of a sample,
        each result is a list of candidates, where each candidate is a tuple: (sequence, score)
          - sequence: token sequence (list), the first token defaults to the start token (set to 0 here)
          - score: accumulated log probability
        """
        # Use no_grad environment to prevent generating unnecessary computation graphs
        with torch.no_grad():
            spectra, inputs, _ = batch
            batch_size = spectra.size(0)

            # Define start and end tokens (adjust according to the actual task)
            start_token = 0  # start token
            eos_token = 30    # end token

            # Encode the spectra of the entire batch to obtain the hidden and cell states of the LSTM encoder
            num_layers = self.encoder.num_layers * 2 if self.bidirectional else self.encoder.num_layers
            hc = torch.empty((num_layers, batch_size, self.encoder.hidden_size), device=spectra.device)
            hc = nn.init.xavier_uniform_(hc).type_as(spectra)
            _, (hidden, cell) = self.encoder(spectra, (hc, hc))
            # Note: hidden and cell shapes are (num_layers, batch_size, hidden_dim)

            all_beam_results = []  # Store beam search results for each sample

            # Perform beam search separately for each sample in the batch
            for i in range(batch_size):
                # Extract the state of the i-th sample, keeping the batch dimension as 1
                hidden_i = hidden[:, i:i+1, :].contiguous()
                cell_i = cell[:, i:i+1, :].contiguous()

                # Initialize beam, each candidate is (sequence, score, hidden, cell)
                beam = [([start_token], 0.0, hidden_i, cell_i)]

                # Generate at most max_word_len-1 tokens (excluding the start token)
                for t in range(self.max_word_len - 1):
                    new_beam = []
                    for seq, score, h, c in beam:
                        # If the current candidate sequence already ends with the end token, do not expand further
                        if seq[-1] == eos_token:
                            new_beam.append((seq, score, h, c))
                            continue
                        # Convert the last token of the current candidate sequence to a tensor, shape (1, 1)
                        token_tensor = torch.tensor([[seq[-1]]], dtype=torch.long, device=spectra.device)
                        # Call decode_step to get the log probability distribution at the current time step
                        logits, new_h, new_c = self.decode_step(token_tensor, h, c)  # logits: (1, num_classes)
                        log_probs = logits.squeeze(0)  # shape: (num_classes,)
                        # Select the beam_width candidate tokens with the highest probability
                        top_log_probs, top_indices = log_probs.topk(beam_width)
                        for j in range(beam_width):
                            next_token = top_indices[j].item()
                            new_score = score + top_log_probs[j].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, new_score, new_h, new_c))
                    # Keep the beam_width candidate sequences with the highest accumulated log probability
                    beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
                all_beam_results.append(beam)
            return all_beam_results

class GPT_topk(GPT):
    def decode_step(self, seq, encoder_outputs):
        """
        Single step decoding:
          - Input: Currently generated token sequence seq (shape (1, seq_len)) and corresponding encoder_outputs (shape (1, mem_len, hidden_dim)).
          - Output: After calculation by the transformer layer, return the log probability logits of the last time step, shape (1, num_classes).
        """
        # offset is usually the sequence length of the encoder output (e.g., the duration of the spectra)
        offset = encoder_outputs.size(1)
        # Construct the mask, the generation of the mask depends on the current decoder input seq and offset (consistent with forward)
        mask = self.pad_masking(seq, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(seq, offset)

        # Add token embedding and positional embedding
        x = self.token_embedding(seq) + self.positional_embedding(seq)

        # Sequentially pass through each transformer layer (each layer receives encoder_outputs and mask)
        for transformer in self.transformers:
            x = transformer(x, encoder_outputs, mask)
        x = self.ln_head(x)
        logits = F.log_softmax(self.outlinear(x), dim=-1)
        return logits

    def predict_step(self, batch, batch_idx, beam_width=5):
        """
        Use beam search for each input sample to generate top-k candidate token sequences.
        Each candidate consists of (sequence, score), where sequence is the generated token sequence,
        and score is the accumulated log probability.

        Different from LSTM generation, the GPT model needs to pass the entire decoder input each time,
        its input shape is (1, max_word_len) (the current candidate sequence fills the front part, the remaining part is filled with 0),
        and extract the prediction distribution at the current position (i.e., the position corresponding to the current candidate length).
        """
        # Fix the random seed to ensure the generation process is repeatable (can be omitted according to actual needs)
        seed = 78438379
        deterministic = True
        seed_everything(seed, workers=deterministic)

        with torch.no_grad():
            spectra, inputs, _ = batch
            batch_size = spectra.size(0)

            # Define start and end tokens (adjust according to the specific task)
            start_token = 0  # start token
            eos_token = 30   # end token

            # Encode the spectra of the entire batch to get the encoder output memory
            batch_size = spectra.size(0)
            num_layers = self.LSTM.num_layers * 2 if self.bidirectional else self.LSTM.num_layers
            hc = torch.empty((num_layers, batch_size, self.LSTM.hidden_size), device=spectra.device)
            hc = torch.nn.init.xavier_uniform_(hc).type_as(spectra)
            encoder_outputs, (_, _) = self.LSTM(spectra, (hc, hc))

            all_beam_results = []  # Store beam search results for each sample

            # Perform beam search separately for each sample in the batch
            for i in range(batch_size):
                # Take out the spectral data of a single sample (keep the batch dimension as 1)
                encoder_outputs_i = encoder_outputs[i:i + 1]  # shape: (1, mem_len, hidden_dim)
                # Initialize beam, the initial candidate sequence only contains the start token, the accumulated score is 0.0
                beam = [([start_token], 0.0)]
                # Generate at most max_word_len - 1 tokens (excluding the start token)
                for t in range(self.max_word_len - 1):
                    new_beam = []
                    for seq, score in beam:
                        # If the current candidate has ended with the end token, do not expand further
                        if seq[-1] == eos_token:
                            new_beam.append((seq, score))
                            continue

                        # Construct decoder input, shape (1, max_word_len)
                        # Fill the generated tokens into the front part, fill the remaining positions with 0
                        inp = torch.zeros((1, self.max_word_len), dtype=torch.int, device=spectra.device)
                        seq_len = len(seq)
                        inp[0, :seq_len] = torch.tensor(seq, dtype=torch.int, device=spectra.device)
                        # Call the GPT model for prediction, return the log probability distribution of shape (1, max_word_len, num_classes)
                        #outputs = self(sample_spectra, inp)
                        outputs = self.decode_step(inp, encoder_outputs_i)
                        # The current prediction position is t (e.g., if the initial candidate length is 1, the prediction position is 1)
                        logits = outputs[0, t, :]  # shape: (num_classes,)

                        # Select the beam_width candidate tokens with the highest probability
                        top_log_probs, top_indices = logits.topk(beam_width)
                        for j in range(beam_width):
                            next_token = top_indices[j].item()
                            new_score = score + top_log_probs[j].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, new_score))
                    # Keep the beam_width candidate sequences with the highest accumulated score (can be normalized, e.g., score/sequence_length)
                    beam = sorted(new_beam, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
                all_beam_results.append(beam)
            return all_beam_results

class GRU_topk(GRU_test):
    def decode_step(self, embedded, hidden):
        """
        Single step decoding:
         - Input: embedded: Tensor of shape (1, 1, hidden_dim) (embedded representation of the current token)
                  hidden: Current GRU state, shape (num_layers, 1, hidden_dim)
         - Processing: Pass the embedded input through the forward_step to get the output at the current time step,
                  then pass through linear mapping and log_softmax to get the log probability distribution
         - Return: logits (shape (1, 1, num_classes)) and the new hidden state (call detach() to cut off gradients)
        """
        # embedded: (1, 1, hidden_dim)
        output, new_hidden = self.forward_step(embedded, hidden)
        output = self.linear2(output)       # (1, 1, num_classes)
        logits = F.log_softmax(output, dim=-1)  # (1, 1, num_classes)
        new_hidden = new_hidden.detach()
        # Return the log probability of the last time step (shape (1, 1, num_classes))
        return logits, new_hidden

    def predict_step(self, batch, batch_idx, beam_width=5):
        """
        Use beam search to generate top-k candidate token sequences, modified to save the embedding cache at each step,
        thereby avoiding embedding the entire sequence each time.

        Generation process:
          - For each sample, initialize the candidate beam to contain only the start token (and its embedding)
          - At each step, only calculate the embedding of the newly generated token and append it to the candidate's embedding cache
          - Use the recursive nature of GRU, relying only on the current token's embedding and hidden state for the next step generation
        """
        # Fix the random seed to ensure the generation process is repeatable (can be omitted according to actual needs)
        seed = 78438379
        deterministic = True
        seed_everything(seed, workers=deterministic)

        with torch.no_grad():
            spectra, inputs, _ = batch
            batch_size = spectra.size(0)

            # Define start and end tokens (adjust according to the specific task)
            start_token = 0  # start token
            eos_token = 30   # end token

            # Encode the spectra of the entire batch to get the encoder output
            num_layers = self.encoder.num_layers * 2 if self.bidirectional else self.encoder.num_layers
            hc = torch.empty((num_layers, batch_size, self.encoder.hidden_size), device=spectra.device)
            hc = torch.nn.init.xavier_uniform_(hc).type_as(spectra)
            encoder_outputs, hidden = self.encoder(spectra, hc)

            all_beam_results = []  # Store beam search results for each sample
            inp = torch.zeros((batch_size, self.max_word_len)).type_as(spectra)
            inp = inp.int()
            embedded = self.embedding(inp)
            # Perform beam search separately for each sample in the batch
            for i in range(batch_size):
                # Take out the hidden state of a single sample, shape (num_layers, 1, hidden_dim)
                hidden_i = hidden[:, i:i + 1, :]
                embedded_i = embedded[i:i + 1, :, :]
                # Initialize beam: each candidate contains (token sequence, accumulated score, hidden state, embedding cache)
                beam = [([start_token], 0.0, hidden_i, embedded_i)]
                # Generate at most max_word_len - 1 tokens (excluding the start token)
                for t in range(self.max_word_len - 1):
                    new_beam = []
                    for seq, score, h, emb in beam:
                        # If the candidate has ended with the end token, do not expand
                        if seq[-1] == eos_token:
                            new_beam.append((seq, score, h, emb))
                            continue

                        # Only embed the new token (no need to construct the token id tensor for the entire candidate sequence)
                        logits, new_h = self.decode_step(emb, h)
                        # logits: shape (1, 1, num_classes); take logits[0, t, :] to get the log probability of each token at the current time step
                        top_log_probs, top_indices = logits[0, t, :].topk(beam_width)
                        # Expand the current candidate for beam_width candidates
                        for j in range(beam_width):
                            next_token = top_indices[j].item()
                            new_score = score + top_log_probs[j].item()
                            new_seq = seq + [next_token]
                            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=spectra.device)
                            token_tensor = self.embedding(token_tensor)
                            # Append the embedding of the new token to the existing embedding cache
                            emb = torch.cat([emb[:, :t, :], token_tensor, emb[:, t + 1:, :]], dim=1)
                            new_beam.append((new_seq, new_score, new_h, emb))
                    # Keep the beam_width candidates with the highest score (can be normalized based on actual needs)
                    beam = sorted(new_beam, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
                all_beam_results.append(beam)
            return all_beam_results

class Transformer_topk(Transformer):
    # Added helper function: encode the spectrum, return the initial hidden and cell for the decoder
    def encode_spectra(self, spectra):
        """
        Encode the spectrum, return the encoder output (memory), used for the decoder's attention mechanism.
        This process is consistent with the encoder part in forward.
        """
        # Preprocess the spectrum, positional encoding and BN
        encoder_input = self.mlp(spectra)
        encoder_input = self.positional_encoding(encoder_input)
        encoder_input = self.bn_head_encoder(encoder_input)
        # Get the output memory of the encoder
        encoder_output = self.encoder(encoder_input)
        return encoder_output

    # Added helper function: single step decoding, input a single token and hidden/cell state, output logits at the current time step and new hidden/cell
    def decode_step(self, seq, memory):
        """
        Single step decoding:
          - Input: Currently generated token sequence seq (shape (1, seq_len)) and corresponding memory (shape (1, mem_len, hidden_dim))
          - Generation: Decoder output logits for the current sequence, return the logits of the last time step (shape (1, num_classes))
        """
        # Construct mask based on the current input sequence (the sequence length of tgt will be calculated internally)
        tgt_mask, tgt_key_padding_mask = self.create_mask(seq)
        # Get token embedding, plus positional encoding
        decoder_in = self.positional_encoding(self.tgt_tok_emb(seq))

        decoder_in = self.ln_head_decoder(decoder_in)
        # Call Transformer decoder (Note: memory has already been obtained by the encoder)
        output = self.decoder(
            tgt=decoder_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Map the decoder output to the vocabulary size and take the log probability
        logits = F.log_softmax(self.linear(output), dim=-1)
        # Only return the logits of the last time step, used to predict the next token
        return logits  # shape: (1, seq_len, num_classes) -> return logits[:, -1, :] for shape (1, num_classes) is more standard


    # Modified predict_step, using beam search to generate top-k molecules
    # Here beam_width is k, default is 5, can be adjusted as needed
    def predict_step(self, batch, batch_idx, beam_width=5):
        """
        For each input spectrum sample, use beam search to generate top-k candidate sequences.
        Returns a list, where each element corresponds to the beam search result of a sample,
        each result is a list of candidates, where each candidate is a tuple: (sequence, score)
          - sequence: token sequence (list), the first token is the start token 
          - score: accumulated log probability
        """
        spectra, inputs, _ = batch
        batch_size = spectra.shape[0]

        # Define start and end tokens (adjust according to the actual task situation)
        start_token = 0  # Start token
        eos_token = 30    # End token

        # Encode the spectra of the entire batch to obtain the encoder output memory
        encoder_output = self.encode_spectra(spectra)

        all_beam_results = []  # Store beam search results for each sample

        # Perform beam search separately for each sample in the batch
        for i in range(batch_size):
            # Take out the spectral data of a single sample (keep the batch dimension as 1)
            encoder_outputs_i = encoder_output[i:i + 1]  # shape: (1, mem_len, hidden_dim)
            # Initialize beam, the initial candidate sequence only contains the start token, the accumulated score is 0.0
            beam = [([start_token], 0.0)]
            # Generate at most max_word_len - 1 tokens (excluding the start token)
            for t in range(self.max_word_len - 1):
                new_beam = []
                for seq, score in beam:
                    # If the current candidate has ended with the end token, do not expand further
                    if seq[-1] == eos_token:
                        new_beam.append((seq, score))
                        continue

                    # Construct decoder input, shape (1, max_word_len)
                    # Fill the generated tokens into the front part, fill the remaining positions with 0 
                    inp = torch.zeros((1, self.max_word_len), dtype=torch.int, device=spectra.device)
                    seq_len = len(seq)
                    inp[0, :seq_len] = torch.tensor(seq, dtype=torch.int, device=spectra.device)
                    # Call the decode_step function for prediction, return the log probability distribution shape (1, max_word_len, num_classes)
                    outputs = self.decode_step(inp, encoder_outputs_i)
                    # The current prediction position is t (e.g., if the initial candidate length is 1, the prediction position is 1)
                    # Take the logits for the *next* token to predict, which corresponds to the *last* generated token's position (seq_len - 1) in the output.
                    # But since we index from 0 and t goes from 0 to max_word_len-2, the index t correctly corresponds to the position of the *next* token to predict.
                    logits = outputs[0, t, :]  # shape: (num_classes,)

                    # Select the beam_width candidate tokens with the highest probability
                    top_log_probs, top_indices = logits.topk(beam_width)
                    for j in range(beam_width):
                        next_token = top_indices[j].item()
                        new_score = score + top_log_probs[j].item()
                        new_seq = seq + [next_token]
                        new_beam.append((new_seq, new_score))
                # Keep the beam_width candidate sequences with the highest accumulated score (can be normalized, e.g., score/sequence_length)
                beam = sorted(new_beam, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
            all_beam_results.append(beam)
        return all_beam_results
