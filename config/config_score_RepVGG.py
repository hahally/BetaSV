class Config(object):
    save_dir = 'Himia_80FBANK_RepVGG_A0_MQMHASTP_all_AAMsoftmax_256'
    # val_dir = '../data/'
    val_dir = '../data/himia/test/SPEECHDATA/'
    save_name = 'repvgg_asp_dev_1m'
    scoring = True     # True: extract and scoring 
                   # False: extract, not scoring
    onlyscoring = False  # True : has npy
                   # False : no npy

    workers = 0
    batch_size = 500
    max_frames=200
    
    fs = 16000
    nfft = 512
    win_len = 0.025
    hop_len = 0.01
    n_mels = 80
    
    conv_type = '2D' #1D, 2D
    model = 'RepVGGStatsPool' # ResNet34StatsPool,TDNN,ECAPA_TDNN
    in_planes = 64 # conv_type:1D, in_planes=n_mels; 2D, in_planes=32/64
    embd_dim = 256
    hidden_dim = 1024 # ECAPA_TDNN

    gpu = '0'
    
    # utt2wav_val = [line.split() for line in open('%s/dev.mic.scp' % val_dir)]
    utt2wav_val = [line.split() for line in open('%s/test.wav.scp' % val_dir, encoding='utf-8')]
    trails_file ='%s/trials_1m' % val_dir 
    # trails_file ='%s/trails_dev_1m' % val_dir
