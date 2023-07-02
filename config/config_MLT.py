class Config(object):
    save_dir = 'Himia_wave_MLTmodel_1m_AAMsoftmax_256'
    train_dir = '../data/'
    val_dir = '../data/'
    
    workers = 0
    batch_size = 1
    max_frames = 200
    
    data_wavaug = True
    
    fs = 16000

    conv_type = '1D' #1D, 2D
    model = 'MLTmodel' # ResNet34StatsPool,TDNN,ECAPA_TDNN
    in_planes = 768 # conv_type:1D, in_planes=n_mels; 2D, in_planes=32, 64
    embd_dim = 256
    hidden_dim = 1024
    classifier = 'AAMSoftmax' # AAMSoftmax,AMSoftmax, ASoftmax, Softmax
    angular_m = 0.2
    angular_s = 32
    
    warm_up_epoch = 2
    lr = 0.001

    epochs = 50
    start_epoch = 0
    load_classifier = False
    
    seed = 3007
    gpu = '0'
    
    utt2wav = [line.split() for line in open('%s/train.1m.scp' % train_dir)]
    spk2int = {line.split()[0]:i for i, line in enumerate(open('%s/train_1m_spk2utt' % train_dir, encoding='utf-8'))}
    spk2utt = {line.split()[0]:line.split()[1:] for line in open('%s/train_1m_spk2utt' % train_dir, encoding='utf-8')}
    
    # utt2wav = [line.split() for line in open('%s/wav.scp' % train_dir, encoding='utf-8')]
    # spk2int = {line.split()[0]:i for i, line in enumerate(open('%s/spk2utt' % train_dir,encoding='utf-8'))}
    # spk2utt = {line.split()[0]:line.split()[1:] for line in open('%s/spk2utt' % train_dir,encoding='utf-8')}
    
    # utt2wav_val = [line.replace('智能家居场景说话人识别挑战赛','teamshare').split() for line in open('%s/dev_scp_mic' % val_dir)]
    utt2wav_val = None
    noise_dir = '../data/musan'
    rir_dir = '../data/RIRS_NOISES'
    noise_list = {'noise': [i.strip('\n').replace('/root/智能家居场景说话人识别挑战赛','..') for i in open('%s/noise_wav_list'% noise_dir, encoding='utf-8') ],
              'music': [i.strip('\n').replace('/root/智能家居场景说话人识别挑战赛','..') for i in open('%s/music_wav_list'% noise_dir, encoding='utf-8') ],
              'reverb': [i.strip('\n').replace('/PATH','../data') for i in open('%s/rir_list'% rir_dir, encoding='utf-8') ]}
    