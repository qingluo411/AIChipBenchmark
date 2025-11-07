import torch
import torch.nn as nn
import time
from blocks.encoder_layer import EncoderLayer
from blocks.decoder_layer import DecoderLayer


def test_transformer_encoder_block(d_model, n_head, ffn_hidden, batch_size, seq_len, num_iterations):

    transformer_encoder_block = EncoderLayer (d_model, ffn_hidden, n_head, drop_prob = 0.1).dlc()
    transformer_encoder_block = transformer_encoder_block.train()
    
    input_tensor = torch.rand(batch_size, seq_len, d_model).dlc()

    # warm up
    for i in range(20):
        output_tensor = transformer_encoder_block(input_tensor, None)

    # test 
    torch.dlc.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = transformer_encoder_block(input_tensor, None)
    torch.dlc.synchronize()
    end_time = time.time()
    print("Time per iteration of encoder: {:.6f} ms".format((end_time - start_time) / num_iterations * 1000))


def test_transformer_decoder_block(d_model, n_head, ffn_hidden, batch_size, tgt_len, memory_len, num_iterations):
    transformer_decoder_block = DecoderLayer(d_model, ffn_hidden, n_head, drop_prob = 0.1).dlc()
    tgt_tensor = torch.rand(batch_size, tgt_len, d_model).dlc()
    memory_tensor = torch.rand(batch_size, memory_len, d_model).dlc()

    transformer_decoder_block = transformer_decoder_block.train()
    # warm up
    for i in range(20):
        output_tensor = transformer_decoder_block(tgt_tensor, memory_tensor, None, None)

   # test 
    torch.dlc.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = transformer_decoder_block(tgt_tensor, memory_tensor, None, None)
    torch.dlc.synchronize()
    end_time = time.time()

    print("Time per iteration of decoder: {:.6f} ms".format((end_time - start_time) / num_iterations * 1000))

# test
test_transformer_encoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, seq_len=512, num_iterations=1000)
test_transformer_decoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, tgt_len=512, memory_len=512, num_iterations=1000)


