from src.models import  LSTM_autoregressive, GPT, GRU_test, Transformer
from pytorch_lightning import seed_everything
from torch import nn
import torch
import torch.nn.functional as F

class LSTM_autoregressive_topk(LSTM_autoregressive):
    def decode_step(self, token, hidden, cell):
        """
        单步解码：
         - 输入：token: 形状为 (1, 1) 的张量（当前 token，下标为 int）
                 hidden, cell：LSTM 当前状态，形状均为 (num_layers, 1, hidden_dim)
         - 处理：先将 token 通过 embedding 与 dropout 得到 embedded，再调用 forward_step 得到当前时刻输出，
                   然后经过 dropout、线性映射、log_softmax 得到对数概率分布
         - 返回：logits（形状为 (1, num_classes)）、new_hidden、new_cell
        """
        # token: (1, 1)
        embedded = self.embedding(token)         # shape: (1, 1, hidden_dim)
        output, new_hidden, new_cell = self.forward_step(embedded, hidden, cell)
        output = self.linear2(output)              # shape: (1, 1, num_classes)
        logits = F.log_softmax(output, dim=-1)       # shape: (1, 1, num_classes)
        # 对新的 hidden 和 cell 调用 detach() 切断梯度传递
        new_hidden = new_hidden.detach()
        new_cell = new_cell.detach()
        return logits[:, -1, :], new_hidden, new_cell  # 返回 shape: (1, num_classes)

    def predict_step(self, batch, batch_idx, beam_width=5):
        seed = 78438379
        deterministic = True
        seed_everything(seed, workers=deterministic)
        """
        对每个输入的光谱样本，使用 beam search 生成 top-k 的候选 token 序列。
        返回一个列表，每个元素对应一个样本的 beam search 结果，
        每个结果为一个候选列表，其中每个候选为元组：(sequence, score)
          - sequence: token 序列（列表），第一个 token 默认为 start token（此处设为 0）
          - score: 累积的对数概率
        """
        # 使用 no_grad 环境，防止生成不必要的计算图
        with torch.no_grad():
            spectra, inputs, _ = batch
            batch_size = spectra.size(0)

            # 定义起始符和终止符（根据实际任务调整）
            start_token = 0  # 假设 0 为起始符
            eos_token = 30    # 假设 1 为终止符

            # 对整个 batch 的 spectra 进行编码，获得 LSTM encoder 的 hidden 和 cell
            num_layers = self.encoder.num_layers * 2 if self.bidirectional else self.encoder.num_layers
            hc = torch.empty((num_layers, batch_size, self.encoder.hidden_size), device=spectra.device)
            hc = nn.init.xavier_uniform_(hc).type_as(spectra)
            _, (hidden, cell) = self.encoder(spectra, (hc, hc))
            # 注意：hidden 和 cell 的形状为 (num_layers, batch_size, hidden_dim)

            all_beam_results = []  # 存储每个样本的 beam search 结果

            # 对 batch 中每个样本分别进行 beam search
            for i in range(batch_size):
                # 提取第 i 个样本的状态，保持 batch 维度为 1
                hidden_i = hidden[:, i:i+1, :].contiguous()
                cell_i = cell[:, i:i+1, :].contiguous()

                # 初始化 beam，每个候选为 (sequence, score, hidden, cell)
                beam = [([start_token], 0.0, hidden_i, cell_i)]

                # 生成最多 max_word_len-1 个 token（不计起始符）
                for t in range(self.max_word_len - 1):
                    new_beam = []
                    for seq, score, h, c in beam:
                        # 如果当前候选序列已以终止符结尾，则不再扩展
                        if seq[-1] == eos_token:
                            new_beam.append((seq, score, h, c))
                            continue
                        # 将当前候选序列最后一个 token 转为张量，形状 (1, 1)
                        token_tensor = torch.tensor([[seq[-1]]], dtype=torch.long, device=spectra.device)
                        # 调用 decode_step 得到当前时刻对数概率分布
                        logits, new_h, new_c = self.decode_step(token_tensor, h, c)  # logits: (1, num_classes)
                        log_probs = logits.squeeze(0)  # shape: (num_classes,)
                        # 选择概率最高的 beam_width 个候选 token
                        top_log_probs, top_indices = log_probs.topk(beam_width)
                        for j in range(beam_width):
                            next_token = top_indices[j].item()
                            new_score = score + top_log_probs[j].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, new_score, new_h, new_c))
                    # 保留累计对数概率最高的 beam_width 个候选序列
                    beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
                all_beam_results.append(beam)
            return all_beam_results

class GPT_topk(GPT):
    def decode_step(self, seq, encoder_outputs):
        """
        单步解码：
          - 输入：当前已生成的 token 序列 seq（形状为 (1, seq_len)）以及对应的 encoder_outputs（形状为 (1, mem_len, hidden_dim)）。
          - 输出：经过 transformer 层计算后，返回最后时刻的对数概率 logits，形状为 (1, num_classes)。
        """
        # offset 通常取 encoder 输出的序列长度（例如 spectra 的时长）
        offset = encoder_outputs.size(1)
        # 构造 mask，mask 的生成依赖于 decoder 当前输入 seq 和 offset（与 forward 中一致）
        mask = self.pad_masking(seq, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(seq, offset)

        # Token embedding 与位置 embedding 相加
        x = self.token_embedding(seq) + self.positional_embedding(seq)

        # 顺序通过各 transformer 层（每层均接收 encoder_outputs 以及 mask）
        for transformer in self.transformers:
            x = transformer(x, encoder_outputs, mask)
        x = self.ln_head(x)
        logits = F.log_softmax(self.outlinear(x), dim=-1)
        return logits

    def predict_step(self, batch, batch_idx, beam_width=5):
        """
        对每个输入样本使用 beam search 生成 top‑k 候选 token 序列。
        每个候选由 (sequence, score) 构成，其中 sequence 为生成的 token 序列，
        score 为累计的对数概率。

        与 LSTM 的生成不同，GPT 模型每次需将整个 decoder 输入传入，
        其输入形状为 (1, max_word_len)（当前候选序列填入前部，剩余部分填充0），
        并取出当前位置（即当前候选长度对应的位置）的预测分布。
        """
        # 固定随机种子，确保生成过程可重复（根据实际需求可省略）
        seed = 78438379
        deterministic = True
        seed_everything(seed, workers=deterministic)

        with torch.no_grad():
            spectra, inputs, _ = batch
            batch_size = spectra.size(0)

            # 定义起始符和终止符（根据具体任务调整）
            start_token = 0  # 假设 0 为起始符
            eos_token = 30   # 假设 30 为终止符

            # 对整个 batch 的 spectra 进行编码，得到 encoder 输出 memory
            batch_size = spectra.size(0)
            num_layers = self.LSTM.num_layers * 2 if self.bidirectional else self.LSTM.num_layers
            hc = torch.empty((num_layers, batch_size, self.LSTM.hidden_size), device=spectra.device)
            hc = torch.nn.init.xavier_uniform_(hc).type_as(spectra)
            encoder_outputs, (_, _) = self.LSTM(spectra, (hc, hc))

            all_beam_results = []  # 存储每个样本的 beam search 结果

            # 对 batch 中的每个样本单独进行 beam search
            for i in range(batch_size):
                # 取出单个样本的谱数据（保持 batch 维度为 1）
                encoder_outputs_i = encoder_outputs[i:i + 1]  # shape: (1, mem_len, hidden_dim)
                # 初始化 beam，初始候选序列仅包含起始符，累计得分为 0.0
                beam = [([start_token], 0.0)]
                # 生成最多 max_word_len - 1 个 token（不计起始符）
                for t in range(self.max_word_len - 1):
                    new_beam = []
                    for seq, score in beam:
                        # 如果当前候选已以终止符结束，则不再扩展
                        if seq[-1] == eos_token:
                            new_beam.append((seq, score))
                            continue

                        # 构造 decoder 输入，形状为 (1, max_word_len)
                        # 将已生成的 token 填入前部，其余位置填0（假设 0 为 padding token）
                        inp = torch.zeros((1, self.max_word_len), dtype=torch.int, device=spectra.device)
                        seq_len = len(seq)
                        inp[0, :seq_len] = torch.tensor(seq, dtype=torch.int, device=spectra.device)
                        # 调用 GPT 模型进行预测，返回形状为 (1, max_word_len, num_classes) 的 log 概率分布
                        #outputs = self(sample_spectra, inp)
                        outputs = self.decode_step(inp, encoder_outputs_i)
                        # 当前预测的位置为 t（例如初始候选长度为1，则预测位置为1）
                        logits = outputs[0, t, :]  # shape: (num_classes,)

                        # 选择概率最高的 beam_width 个候选 token
                        top_log_probs, top_indices = logits.topk(beam_width)
                        for j in range(beam_width):
                            next_token = top_indices[j].item()
                            new_score = score + top_log_probs[j].item()
                            new_seq = seq + [next_token]
                            new_beam.append((new_seq, new_score))
                    # 保留累计得分最高（可归一化处理，如 score/sequence_length）的 beam_width 个候选序列
                    beam = sorted(new_beam, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
                all_beam_results.append(beam)
            return all_beam_results

class GRU_topk(GRU_test):
    def decode_step(self, embedded, hidden):
        """
        单步解码：
         - 输入：token: 形状为 (1, 1) 的张量（当前 token，下标为 int）
                  hidden: GRU 当前状态，形状为 (num_layers, 1, hidden_dim)
         - 处理：将 token 通过 embedding 得到 embedded，再调用 forward_step 得到当前时刻输出，
                  然后经过线性映射和 log_softmax 得到对数概率分布
         - 返回：logits（形状为 (1, num_classes)）以及新的 hidden 状态（调用 detach() 切断梯度）
        """
        # token: (1, 1)
        #embedded = self.embedding(token)  # (1, 1, hidden_dim)
        output, new_hidden = self.forward_step(embedded, hidden)
        output = self.linear2(output)       # (1, 1, num_classes)
        logits = F.log_softmax(output, dim=-1)  # (1, 1, num_classes)
        new_hidden = new_hidden.detach()
        # 返回最后一个时间步的对数概率（形状为 (1, num_classes)）
        return logits, new_hidden

    def predict_step(self, batch, batch_idx, beam_width=5):
        """
        使用 beam search 生成 top‑k 候选 token 序列，修改后在每步生成时保存 embedding 缓存，
        从而避免每次对整个序列进行 embedding 操作。

        生成过程：
          - 对每个样本，初始化候选 beam 为仅含起始符（以及其 embedding）
          - 每步仅计算新生成 token 的 embedding，并将其拼接到候选的 embedding 缓存上
          - 使用 GRU 的递归特性，仅依赖当前 token 的 embedding 和 hidden 状态进行下一步生成
        """
        # 固定随机种子，确保生成过程可重复（根据实际需求可省略）
        seed = 78438379
        deterministic = True
        seed_everything(seed, workers=deterministic)

        with torch.no_grad():
            spectra, inputs, _ = batch
            batch_size = spectra.size(0)

            # 定义起始符和终止符（根据具体任务调整）
            start_token = 0  # 假设 0 为起始符
            eos_token = 30   # 假设 30 为终止符

            # 对整个 batch 的 spectra 进行编码，得到 encoder 输出
            num_layers = self.encoder.num_layers * 2 if self.bidirectional else self.encoder.num_layers
            hc = torch.empty((num_layers, batch_size, self.encoder.hidden_size), device=spectra.device)
            hc = torch.nn.init.xavier_uniform_(hc).type_as(spectra)
            encoder_outputs, hidden = self.encoder(spectra, hc)

            all_beam_results = []  # 存储每个样本的 beam search 结果
            inp = torch.zeros((batch_size, self.max_word_len)).type_as(spectra)
            inp = inp.int()
            embedded = self.embedding(inp)
            # 对 batch 中的每个样本分别进行 beam search
            for i in range(batch_size):
                # 取出单个样本的 hidden 状态，形状为 (num_layers, 1, hidden_dim)
                hidden_i = hidden[:, i:i + 1, :]
                embedded_i = embedded[i:i + 1, :, :]
                # 初始化 beam：每个候选包含 (token 序列, 累计得分, hidden 状态, embedding 缓存)
                beam = [([start_token], 0.0, hidden_i, embedded_i)]
                # 生成最多 max_word_len - 1 个 token（不计起始符）
                for t in range(self.max_word_len - 1):
                    new_beam = []
                    for seq, score, h, emb in beam:
                        # 若该候选已以终止符结束，则不扩展
                        if seq[-1] == eos_token:
                            new_beam.append((seq, score, h, emb))
                            continue

                        # 只对新 token 进行 embedding（无需构造整个候选序列的 token id tensor）
                        #last_token_tensor = torch.tensor([[seq[-1]]], device=spectra.device, dtype=torch.long)
                        logits, new_h = self.decode_step(emb, h)
                        # logits: 形状 (1, 1, num_classes)；取 logits[0, 0, :] 得到当前时刻各 token 的对数概率
                        top_log_probs, top_indices = logits[0, t, :].topk(beam_width)
                        # 针对 beam_width 个候选扩展当前候选
                        for j in range(beam_width):
                            next_token = top_indices[j].item()
                            new_score = score + top_log_probs[j].item()
                            new_seq = seq + [next_token]
                            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=spectra.device)
                            token_tensor = self.embedding(token_tensor)
                            # 将新 token 的 embedding 拼接到已有的 embedding 缓存上
                            emb = torch.cat([emb[:, :t, :], token_tensor, emb[:, t + 1:, :]], dim=1)
                            new_beam.append((new_seq, new_score, new_h, emb))
                    # 保留得分最高的 beam_width 个候选（此处可以根据实际需要归一化分数）
                    beam = sorted(new_beam, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
                all_beam_results.append(beam)
            return all_beam_results

class Transformer_topk(Transformer):
    # 新增辅助函数：对光谱进行编码，返回 decoder 初始 hidden 和 cell
    def encode_spectra(self, spectra):
        """
        对光谱进行编码，返回 encoder 输出（memory），用于 decoder 的注意力机制。
        该过程与 forward 中的 encoder 部分一致。
        """
        # 对光谱进行前处理、位置编码与 BN
        encoder_input = self.mlp(spectra)
        encoder_input = self.positional_encoding(encoder_input)
        encoder_input = self.bn_head_encoder(encoder_input)
        # 得到 encoder 的输出 memory
        encoder_output = self.encoder(encoder_input)
        return encoder_output

    # 新增辅助函数：单步解码，输入单个 token 以及 hidden 和 cell 状态，输出当前时刻的 logits 及新的 hidden 和 cell
    def decode_step(self, seq, memory):
        """
        单步解码：
          - 输入：当前已生成的 token 序列 seq（形状为 (1, seq_len)）和对应的 memory（形状为 (1, mem_len, hidden_dim)）
          - 生成：decoder 对当前序列的输出 logits，返回最后一个时刻的 logits（形状为 (1, num_classes)）
        """
        # 根据当前输入序列构造 mask（内部会计算 tgt 的序列长度）
        tgt_mask, tgt_key_padding_mask = self.create_mask(seq)
        # 得到 token embedding，再加上位置编码
        decoder_in = self.positional_encoding(self.tgt_tok_emb(seq))

        decoder_in = self.ln_head_decoder(decoder_in)
        # 调用 Transformer decoder（注意：memory 已经由 encoder 得到）
        output = self.decoder(
            tgt=decoder_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # 将 decoder 输出映射到词表大小，并取对数概率
        logits = F.log_softmax(self.linear(output), dim=-1)
        # 只返回最后时刻的 logits，用于预测下一个 token
        return logits  # shape: (1, num_classes)

    # 修改后的 predict_step，使用 beam search 生成 top-k 分子
    # 这里 beam_width 就是 k，默认设为 5，可根据需要调整
    def predict_step(self, batch, batch_idx, beam_width=5):
        """
        对每个输入的光谱样本，使用 beam search 生成 top‑k 的候选序列。
        返回一个列表，每个元素对应一个样本的 beam search 结果，
        每个结果为一个候选列表，其中每个候选为元组：(sequence, score)
          - sequence: token 序列（列表），第一个 token 为 start token（此处假设为 0）
          - score: 累积的对数概率
        """
        spectra, inputs, _ = batch
        batch_size = spectra.shape[0]

        # 定义起始符和终止符（根据任务实际情况调整）
        start_token = 0  # 起始符
        eos_token = 30    # 终止符

        # 对整个 batch 的光谱进行编码，获得 encoder 的输出 memory
        encoder_output = self.encode_spectra(spectra)

        all_beam_results = []  # 存储每个样本的 beam search 结果

        # 对 batch 中的每个样本单独进行 beam search
        for i in range(batch_size):
            # 取出单个样本的谱数据（保持 batch 维度为 1）
            encoder_outputs_i = encoder_output[i:i + 1]  # shape: (1, mem_len, hidden_dim)
            # 初始化 beam，初始候选序列仅包含起始符，累计得分为 0.0
            beam = [([start_token], 0.0)]
            # 生成最多 max_word_len - 1 个 token（不计起始符）
            for t in range(self.max_word_len - 1):
                new_beam = []
                for seq, score in beam:
                    # 如果当前候选已以终止符结束，则不再扩展
                    if seq[-1] == eos_token:
                        new_beam.append((seq, score))
                        continue

                    # 构造 decoder 输入，形状为 (1, max_word_len)
                    # 将已生成的 token 填入前部，其余位置填0（假设 0 为 padding token）
                    inp = torch.zeros((1, self.max_word_len), dtype=torch.int, device=spectra.device)
                    seq_len = len(seq)
                    inp[0, :seq_len] = torch.tensor(seq, dtype=torch.int, device=spectra.device)
                    # 调用 GPT 模型进行预测，返回形状为 (1, max_word_len, num_classes) 的 log 概率分布
                    # outputs = self(sample_spectra, inp)
                    outputs = self.decode_step(inp, encoder_outputs_i)
                    # 当前预测的位置为 t（例如初始候选长度为1，则预测位置为1）
                    logits = outputs[0, t, :]  # shape: (num_classes,)

                    # 选择概率最高的 beam_width 个候选 token
                    top_log_probs, top_indices = logits.topk(beam_width)
                    for j in range(beam_width):
                        next_token = top_indices[j].item()
                        new_score = score + top_log_probs[j].item()
                        new_seq = seq + [next_token]
                        new_beam.append((new_seq, new_score))
                # 保留累计得分最高（可归一化处理，如 score/sequence_length）的 beam_width 个候选序列
                beam = sorted(new_beam, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
            all_beam_results.append(beam)
        return all_beam_results