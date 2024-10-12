import torch
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *

# modified WAVENET file implementation from: https://github.com/Vichoko/pytorch-wavenet/tree/master

dtype = torch.FloatTensor  # data type
ltype = torch.LongTensor  # label type

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = WaveNetModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     dtype=dtype,
                     bias=True)
model.cuda()
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='example.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='../../../notebooks/reproduce_training_ekaterina/gen_model_test/unpacked_data',
                      test_stride=500)
print('the dataset has ' + str(len(data)) + ' items')


def generate_and_log_samples(step):
    sample_length = 32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    # logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    # logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated")


trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.001,
                         snapshot_path='snapshots',
                         snapshot_name='birdset_model',
                         snapshot_interval=1000,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=16,
              epochs=2)

start_data = data[250000][0] # use start data from the data set
start_data = torch.max(start_data, 0)[1] # convert one hot vectors to integers

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

generated = model.generate_fast(num_samples=160000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=1000,
                                 temperature=1.0,
                                 regularize=0.)
