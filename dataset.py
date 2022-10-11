from torch.utils.data import Dataset

class KERC22(Dataset):
    def __init__(self, data_file_name, label_file_name=None, include_description=True):
        super(KERC22, self).__init__()
        with open(data_file_name, 'r', encoding='utf-8') as f:
            data_file = f.readlines()[1:]
        if label_file_name:
            with open(label_file_name, 'r', encoding='utf-8') as f:
                label_file = f.readlines()[1:]

        context = []
        context_speaker = []
        context_descrip = []
        dialogs = []
        pre_scene = ""
        self.labelList = sorted({"dysphoria", "euphoria", "neutral"})

        for i, data in enumerate(data_file):
            sentence_id, person, sentence, scene, description = data.strip().split('\t')
            if label_file_name:
                sentence_id, label = label_file[i].strip().split(',')

            if pre_scene != scene:
                context = []
                context_speaker = []
                context_descrip = []

            if description != "NaN":
                context_descrip.append(description)
            else:
                context_descrip.append(None)

            pre_scene = scene
            context.append(sentence)
            context_speaker.append(person)

            if label_file_name:
                label_ind = self.labelList.index(label)
                dialogs.append([context_speaker[:], context[:], context_descrip[:], label_ind])
            else:
                dialogs.append([context_speaker[:], context[:], context_descrip[:], int(sentence_id)])
        
        self.dialogs = []
        for i, dialog in enumerate(dialogs):
            context_speakers, context, ret = dialog
            context_speaker_idx = []
            context_new = []
            speaker_set = []
            
            for person in context_speakers:
                if person not in speaker_set:
                    speaker_set.append(person)
                speakerCLS = speaker_set.index(person)
                context_speaker_idx.append(speakerCLS)

            for data in context:
                for i, person in enumerate(speaker_set):
                    data = data.replace(person, "<s{}>".format(i + 1))
                context_new.append(data)
            
            self.dialogs.append([context_speaker_idx, context_new, ret])

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]
