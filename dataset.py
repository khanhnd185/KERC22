from torch.utils.data import Dataset

class KERC22Narrator(Dataset):
    def __init__(self, txt_file, include_narrator=False):
        super(KERC22Narrator, self).__init__()
        with open(txt_file, 'r', encoding='utf-8') as f:
            dataset = f.readlines()

        context = []
        context_speaker = []
        temp_speakerList = []
        self.emoSet = set()
        self.dialogs = []
        self.speakerNum = []

        self.labelList = sorted({"dysphoria", "euphoria", "neutral"})

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

            if include_narrator == True or speaker != "내레이터":
                label_ind = self.labelList.index(emo)
                self.dialogs.append([context_speaker[:], context[:], label_ind])
                self.emoSet.add(emo)
        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    def get_largest_speaker_num(self):
        return max(self.speakerNum)

class KERC22Narrator_Test(Dataset):
    def __init__(self, txt_file):
        super(KERC22Narrator_Test, self).__init__()
        with open(txt_file, 'r', encoding='utf-8') as f:
            dataset = f.readlines()

        id_list = []
        context = []
        context_speaker = []
        temp_speakerList = []
        self.dialogs = []
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

    def get_largest_speaker_num(self):
        return max(self.speakerNum)