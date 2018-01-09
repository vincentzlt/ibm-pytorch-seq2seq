import argparse
import logging
import os
import sys
sys.path.append(os.getcwd())

import torch
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_path',
    action='store',
    dest='train_path',
    help='Path to train data. data should be tab segged')
parser.add_argument(
    '--dev_path',
    action='store',
    dest='dev_path',
    help='Path to dev data. data should be tab segged ')
parser.add_argument(
    '--expt_dir',
    action='store',
    dest='expt_dir',
    default='./experiment',
    help='Path to experiment directory. If load_checkpoint is True,'
    ' then path to checkpoint directory has to be provided')
parser.add_argument(
    '--load_checkpoint',
    action='store',
    dest='load_checkpoint',
    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument(
    '--resume',
    action='store_true',
    dest='resume',
    default=False,
    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument(
    '--log-level', dest='log_level', default='debug', help='Logging level.')

# make a virtual command
TRAIN_PATH = './data/toy_reverse/train/data.txt'
DEV_PATH = './data/toy_reverse/dev/data.txt'

opt = parser.parse_args(
    '--train_path {TRAIN_PATH} --dev_path {DEV_PATH} '.format(
        TRAIN_PATH=TRAIN_PATH, DEV_PATH=DEV_PATH).split())

# define logger
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# Prepare dataset
src = SourceField()
tgt = TargetField()
max_len = 50


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len


train = torchtext.data.TabularDataset(
    path=opt.train_path,
    format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter)
dev = torchtext.data.TabularDataset(
    path=opt.dev_path,
    format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter)
src.build_vocab(train, max_size=50000)
tgt.build_vocab(train, max_size=50000)
input_vocab = src.vocab
output_vocab = tgt.vocab

# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight=weight, mask=pad)
if torch.cuda.is_available():
    loss.cuda()

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME,
                     opt.load_checkpoint)))
    checkpoint_path = os.path.join(
        opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        bidirectional = True
        encoder = EncoderRNN(
            vocab_size=len(src.vocab),
            max_len=max_len,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            rnn_cell='lstm',
            variable_lengths=True)
        decoder = DecoderRNN(
            vocab_size=len(tgt.vocab),
            max_len=max_len,
            hidden_size=hidden_size * 2,
            dropout_p=0.2,
            use_attention=True,
            bidirectional=bidirectional,
            rnn_cell='lstm',
            eos_id=tgt.eos_id,
            sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder=encoder, decoder=decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(
        expt_dir=opt.expt_dir,
        loss=loss,
        batch_size=32,
        random_seed=None,
        checkpoint_every=50,
        print_every=10,
    )

    seq2seq = t.train(
        model=seq2seq,
        data=train,
        num_epochs=6,
        resume=opt.resume,
        dev_data=dev,
        optimizer=optimizer,
        teacher_forcing_ratio=0.5)

evaluator = Evaluator(loss=loss, batch_size=32)
dev_loss, accuracy = evaluator.evaluate(model=seq2seq, data=dev)
assert dev_loss < 1.5

beam_search = Seq2seq(
    encoder=seq2seq.encoder,
    decoder=TopKDecoder(decoder_rnn=seq2seq.decoder, k=3))

predictor = Predictor(
    model=beam_search, src_vocab=input_vocab, tgt_vocab=output_vocab)
inp_seq = "1 3 5 7 9"
seq = predictor.predict(list(inp_seq.split()))
print(" ".join(seq[:-1]))
print(inp_seq[::-1])
