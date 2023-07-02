class Config(object):
    save_dir = 'Himia_80FBANK_ECAPA_TDNN_ASP_1m_AAMsoftmax_256'
    train_dir = '/root/teamshare/data/'
    val_dir = '/root/teamshare/data/himia/dev/SPEECHDATA'
    
    workers = 8
    batch_size = 500
    max_frames = 200
    
    data_wavaug = True
    data_specaug = True
    specaug_masktime = [10,20]
    specaug_maskfreq = [5,10]
    
    # noise_dir  = 'MUSAN'
    # rir_dir    = 'RIR_Noise'
    fs = 16000
    nfft = 512
    win_len = 0.025
    hop_len = 0.01
    n_mels = 80
    
    conv_type = '1D' #1D, 2D
    model = 'ECAPA_TDNN' # ResNet34StatsPool,TDNN,ECAPA_TDNN
    in_planes = 80 # conv_type:1D, in_planes=n_mels; 2D, in_planes=32, 64
    embd_dim = 256
    hidden_dim = 1024
    classifier = 'AAMSoftmax' # AAMSoftmax,AMSoftmax, ASoftmax, Softmax
    angular_m = 0.2
    angular_s = 64
    
    warm_up_epoch = 2
    lr = 0.001

    epochs = 40
    start_epoch = 0
    load_classifier = False
    
    seed = 3007
    gpu = '0'
    
    # 全部数据
    # utt2wav = [line.split() for line in open('%s/wav.scp' % train_dir)]
    # spk2int = {line.split()[0]:i for i, line in enumerate(open('%s/spk2utt' % train_dir))}
    # spk2utt = {line.split()[0]:line.split()[1:] for line in open('%s/spk2utt' % train_dir)}
    
    utt2wav = [line.split() for line in open('%s/train.1m.scp' % train_dir)]
    spk2int = {line.split()[0]:i for i, line in enumerate(open('%s/train_1m_spk2utt' % train_dir))}
    spk2utt = {line.split()[0]:line.split()[1:] for line in open('%s/train_1m_spk2utt' % train_dir)}
    
    utt2wav_val = [line.replace('智能家居场景说话人识别挑战赛','teamshare').split() for line in open('%s/dev_scp_mic' % val_dir)]

    noise_dir = '/root/teamshare/data/musan'
    rir_dir = '/root/teamshare/data/RIRS_NOISES'
    noise_list = {'noise': [i.strip('\n').replace('智能家居场景说话人识别挑战赛','teamshare') for i in open('%s/noise_wav_list'% noise_dir) ],
              'music': [i.strip('\n').replace('智能家居场景说话人识别挑战赛','teamshare') for i in open('%s/music_wav_list'% noise_dir) ],
              'reverb': [i.strip('\n').replace('智能家居场景说话人识别挑战赛','teamshare') for i in open('%s/rir_list'% rir_dir) ]}
    
