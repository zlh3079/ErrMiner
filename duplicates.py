import json
from tqdm import tqdm


class dialogset:
    def __init__(self, file_name):
        #self.tokenizer = BertTokenizer.from_pretrained(tokenizer_address)
        with open(file_name, 'r') as r:
            data_dict = json.load(r)

        self.ids = data_dict['ids']
        self.dialog = data_dict['dialog']
        self.role = data_dict['role']
        self.label = data_dict['label']
        self.edge = data_dict['edge']
        self.len = len(self.ids)

    def __getitem__(self, index):
        return self.dialog[index], \
            torch.tensor(self.label[index]), \
            self.ids[index], \
            self.role[index], \
            self.edge[index], \
            len(self.dialog[index])


if __name__ == '__main__':

    # print(a.dialog[1])
    count = 0

    for train_data in 'angular', 'appium', 'dl4j', 'docker', 'gitter', 'typescript':
        jsontext = {'ids': [], 'dialog': [],
                    'role': [], 'edge': [], 'label': []}
        train_address = f"./data/Processed/{train_data}"
        # print(train_address)
        # json_address=train_address+'_train.json'
        b = dialogset(train_address + "_train.json")
        for indexb in tqdm(range(0, b.len)):
            txtb = b.dialog[indexb]
            if len(txtb) == 1:
                tempstrb = str(txtb)
                xb = tempstrb.split(" ")
                flag = 0
                if (len(xb) >= 8):
                    tempb = ""
                    for i in range(2, 7):
                        # print(x[i])
                        tempb = tempb + xb[i] + " "
                    for test_data in "angular", "appium", "dl4j", "docker", "gitter", "typescript":
                        test_address = f"./data/Processed/{test_data}"
                        a = dialogset(test_address + "_test.json")
                        for indexa in range(0, a.len):
                            txta = a.dialog[indexa]
                            if len(txta) == 1:
                                tempstra = str(txta)
                                # print(tempstra)
                                xa = tempstra.split(" ")
                                if (len(xa) >= 8):
                                    tempa = ""
                                    for i in range(2, 7):
                                        # print(x[i])
                                        tempa = tempa + xa[i] + " "
                                    if tempa == tempb:
                                        flag = 1
                if flag == 0:
                    jsontext['ids'].append(b.ids[indexb])
                    jsontext['dialog'].append(b.dialog[indexb])
                    jsontext['role'].append(b.role[indexb])
                    jsontext['edge'].append(b.edge[indexb])
                    jsontext['label'].append(b.label[indexb])
            if len(txtb) > 1:
                tempstrb2 = str(txtb[1])
                xb2 = tempstrb2.split(" ")
                flag2 = 0
                if (len(xb2) >= 8):
                    tempb2 = ""
                    for i in range(2, 7):
                        # print(x[i])
                        tempb2 = tempb2 + xb2[i] + " "

                    for test_data in "angular", "appium", "dl4j", "docker", "gitter", "typescript":
                        test_address = f"./data/Processed/{test_data}"
                        a = dialogset(test_address + "_test.json")
                        for indexa in range(0, a.len):
                            txta2 = a.dialog[indexa]
                            if len(txta2) > 1:
                                tempstra2 = str(txta2[1])
                                # print(tempstra)
                                xa2 = tempstra2.split(" ")
                                if (len(xa2) >= 8):
                                    tempa2 = ""
                                    for i in range(2, 7):
                                        # print(x[i])
                                        tempa2 = tempa2 + xa2[i] + " "
                                    if tempa2 == tempb2:
                                        flag2 = 1
                if flag2 == 0:
                    jsontext['ids'].append(b.ids[indexb])
                    jsontext['dialog'].append(b.dialog[indexb])
                    jsontext['role'].append(b.role[indexb])
                    jsontext['edge'].append(b.edge[indexb])
                    jsontext['label'].append(b.label[indexb])
        json.dump(jsontext, open(train_data + '_train.json', 'w'))
        print("已生成" + train_data + "_train.json")
