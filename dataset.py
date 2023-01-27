from torch.utils.data import Dataset

class KERC22(Dataset):
    def __init__(self, tokenizer, data_file_name, label_file_name=None):
        super(KERC22, self).__init__()
        with open(data_file_name, 'r', encoding='utf-8') as f:
            data_file = f.readlines()[1:]
        if label_file_name:
            with open(label_file_name, 'r', encoding='utf-8') as f:
                label_file = f.readlines()[1:]

        self.dialogs = [] 
        self.labelList = sorted({"dysphoria", "euphoria", "neutral"})

        skip_list = []
        token_list = []
        return_list = []
        speaker_list = []
        speaker_name_list = []

        pre_scene = ""
        pre_description = ""

        for i, data in enumerate(data_file):
            sentence_id, person, sentence, scene, description = data.strip().split('\t')
            if label_file_name:
                sentence_id, label = label_file[i].strip().split(',')
            
            if pre_scene != scene and len(token_list) > 0:
                self.dialogs.append((
                    token_list,
                    speaker_list,
                    skip_list,
                    return_list
                ))

                skip_list = []
                token_list = []
                return_list = []
                speaker_list = []
                speaker_name_list = []

                pre_scene = scene


            if description.lower() != "nan" and description != pre_description:
                if "내레이터" not in speaker_name_list:
                    speaker_name_list.append("내레이터")

                speaker_id = speaker_name_list.index("내레이터")
                token = tokenizer.tokenize(description)
                token = tokenizer.convert_tokens_to_ids(token)
                skip_list.append(True)
                token_list.append(token)
                speaker_list.append(speaker_id)

                pre_description = description

            if person not in speaker_name_list:
                speaker_name_list.append(person)

            speaker_id = speaker_name_list.index(person)
            token = tokenizer.tokenize(sentence)
            token = tokenizer.convert_tokens_to_ids(token)
            skip_list.append(False)
            token_list.append(token)
            speaker_list.append(speaker_id)

            if label_file_name:
                return_list.append(self.labelList.index(label))
            else:
                return_list.append(int(sentence_id))


        if len(token_list) > 0:
            self.dialogs.append((
                token_list,
                speaker_list,
                skip_list,
                return_list
            ))



    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

 