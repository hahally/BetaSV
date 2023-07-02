class Config(object):
    save_dir = 'Himia_80FBANK_RepVGG_A0_APS_MQMHASTP_all_AAMsoftmax_256'
    train_dir = './data/'
    val_dir = '../data'
    
    workers = 0
    batch_size = 32
    max_frames = 200
    
    data_wavaug = True
    data_specaug = True
    specaug_masktime = [10,20]
    specaug_maskfreq = [5,10]
    
    fs = 16000
    nfft = 512
    win_len = 0.025
    hop_len = 0.01
    n_mels = 80
    
    conv_type = '2D' #1D, 2D
    model = 'RepVGGStatsPool' # ResNet34StatsPool,TDNN,ECAPA_TDNN,SEResNet34StatsPool
    in_planes = 64 # conv_type:1D, in_planes=n_mels; 2D, in_planes=32, 64
    embd_dim = 256
    hidden_dim = 1024
    classifier = 'AAMSoftmax' # AAMSoftmax,AMSoftmax, ASoftmax, Softmax
    angular_m = 0.2
    angular_s = 32
    
    warm_up_epoch = 2
    lr = 0.001

    epochs = 40
    start_epoch = 0
    load_classifier = False
    
    seed = 3007
    gpu = '0'
    
    # utt2wav = [line.replace('智能家居场景说话人识别挑战赛','teamshare').split() for line in open('%s/train.mic.scp' % train_dir)]
    # 修改路径
    # spk2int = {line.split()[0]:i for i, line in enumerate(open('%s/mic_spk2utt' % train_dir))}
    # spk2utt = {line.split()[0]:line.split()[1:] for line in open('%s/mic_spk2utt' % train_dir)}
    
    utt2wav = [line.split() for line in open('%s/train.all.scp' % train_dir)]
    spk2int = {line.split()[0]:i for i, line in enumerate(open('%s/train_all_spk2utt' % train_dir, encoding='utf-8'))}
    spk2utt = {line.split()[0]:line.split()[1:] for line in open('%s/train_all_spk2utt' % train_dir, encoding='utf-8')}
    
    utt2wav_val = [line.replace('智能家居场景说话人识别挑战赛','teamshare').split() for line in open('%s/dev.mic.scp' % val_dir)]

    noise_dir = '../data/musan'
    rir_dir = '../data/RIRS_NOISES'
    noise_list = {'noise': [i.strip('\n').replace('/root/智能家居场景说话人识别挑战赛','..') for i in open('%s/noise_wav_list'% noise_dir, encoding='utf-8') ],
              'music': [i.strip('\n').replace('/root/智能家居场景说话人识别挑战赛','..') for i in open('%s/music_wav_list'% noise_dir, encoding='utf-8') ],
              'reverb': [i.strip('\n').replace('/PATH','../data') for i in open('%s/rir_list'% rir_dir, encoding='utf-8') ]}
    