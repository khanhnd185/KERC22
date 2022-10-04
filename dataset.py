from torch.utils.data import Dataset

class KERC_loader(Dataset):
    def __init__(self, txt_file):
        self.dialogs = []

        f = open(txt_file, 'r', encoding='utf-8')
        dataset = f.readlines()
        f.close()
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        self.emoSet = set()
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker, utt, emo = data.strip().split('\t')
            context.append(utt)

            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)

            if speaker != "내레이터":
                self.dialogs.append([context_speaker[:], context[:], emo])
                self.emoSet.add(emo)

        self.labelList = sorted(self.emoSet)
        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList

class KERCTest_loader(Dataset):
    def __init__(self, txt_file):
        self.dialogs = []

        f = open(txt_file, 'r', encoding='utf-8')
        dataset = f.readlines()
        f.close()
        temp_speakerList = []
        context = []
        context_speaker = []
        id_list = []
        self.speakerNum = []
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                id_list = []
                continue
            id, speaker, utt = data.strip().split('\t')
            context.append(utt)
            id_list.append(id)

            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)

            if speaker != "내레이터":
                self.dialogs.append([context_speaker[:], context[:], id_list[:]])
    
        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]
