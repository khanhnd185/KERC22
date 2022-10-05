from torch.utils.data import Dataset

class KERC22Narrator(Dataset):
    def __init__(self, txt_file, include_narrator=False):
        super(KERC22Narrator, self).__init__()
        with open(txt_file, 'r', encoding='utf-8') as f:
            dataset = f.readlines()

        context = []
        context_speaker = []
        temp_speakerList = []
        self.dialogs = []
        self.speakerNum = []
        label_list = []
        speaker_list = []

        self.labelList = sorted({"dysphoria", "euphoria", "neutral"})

        for i, data in enumerate(dataset):
            if data == '\n':
                self.speakerNum.append(len(temp_speakerList))

                for j, s in enumerate(speaker_list):
                    if include_narrator == True or s != "내레이터":
                        self.dialogs.append([context_speaker[:(j+1)], context[:(j+1)], context_speaker[j:], context[j:], label_list[j]])

                temp_speakerList = []
                context = []
                context_speaker = []
                label_list = []
                speaker_list = []
                continue
            speaker, utt, emo = data.strip().split('\t')
            context.append(utt)

            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            label_list.append(self.labelList.index(emo))
            speaker_list.append(speaker)

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
        speaker_list = []

        for i, data in enumerate(dataset):
            if data == '\n':
                self.speakerNum.append(len(temp_speakerList))

                for j, s in enumerate(speaker_list):
                    if s != "내레이터":
                        self.dialogs.append([context_speaker[:(j+1)], context[:(j+1)], context_speaker[j:], context[j:], id_list[:(j+1)]])

                temp_speakerList = []
                context = []
                context_speaker = []
                id_list = []
                speaker_list = []
                continue
            id, speaker, utt = data.strip().split('\t')
            context.append(utt)
            id_list.append(id)

            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            speaker_list.append(speaker)

        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    def get_largest_speaker_num(self):
        return max(self.speakerNum)