
trials_mic = open(file='../data/himia/test/SPEECHDATA/trials_mic', mode='r', encoding='utf-8').readlines()
trials_1m = open(file='../data/himia/test/SPEECHDATA/trials_1m', mode='r', encoding='utf-8').readlines()
scp = set()
for s in (trials_mic+trials_1m):
    for ss in s.split():
        scp.add(ss)

with open(file='../data/himia/test/SPEECHDATA/test.wav.scp', mode='w', encoding='utf-8') as f:
    for line in scp:
        f.write(line+ f' /root/智能家居场景说话人识别挑战赛/data/himia/test/SPEECHDATA/wav/{line}' + '\n')
    