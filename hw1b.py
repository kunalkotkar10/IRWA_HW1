import argparse
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
# import lightgbm as lgb
# from xgboost import XGBClassifier
#from sklearn.ensemble import AdaBoostClassifier

class SegmentClassifier:
    def train(self, trainX, trainY):
        # self.clf = DecisionTreeClassifier()  # TODO: experiment with different models
        #self.clf = lgb.LGBMClassifier()
        # self.clf = XGBClassifier()
        # self.clf = RandomForestClassifier()
        self.clf = MLPClassifier()
        #self.clf = AdaBoostClassifier()
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        # self.pos = load_wordlist("data/part-of-speech.histogram")
        # pos_list = list(self.pos)
        # pos_dict = {}
        # for item in pos_list:
        #     split_item = item.split(' ')
        #     freq = split_item[0]
        #     word = split_item[1]
        #     pos = split_item[-1]
        #     try:
        #         pos_dict[pos].append(word)
        #     except KeyError:
        #         pos_dict[pos] = [word]
        # self.pos_dict = pos_dict
        #print(pos_dict["CD"])
        words = text.split()
        def check_head():
            i=0
            for w in words:
                if w == "Subject:":
                    i = i + 1
                if w == "From:":
                    i = i + 1
                if w=="Date:":
                    i = i + 1
                if w=="Keywords:": 
                    i = i + 1
                if w=="Organization:":
                    i = i + 1
            return i

        # def check_pos():
        #     # print(self.pos_dict["CD"])
        #     y = 0
        #     for w in words:
        #         if w.lower()==self.pos_dict["NP"]:
        #             y = y + 1
        #     return y
        
        def check_num():
            w = words[0]
            c = w[0]
            x = c.isnumeric()
            if x:
                return 1
            else:
                return 0

        def check_item():
            y = 0
            for w in words:
                for i in range(0, len(w)-1):
                    if w[i].isnumeric():
                        if w[i+1]=="." or w[i+1]==")":
                            y = y+1
                return y
        
        def check_list():
            y = 0
            for z in range(0,len(text)-2):
                if text[z].isnumeric():
                    if text[z+1]==".":
                        if text[z+2]==" ":
                            #print(text)
                            return 1
                if text[z]=="(":
                    if text[z+1].isnumeric():
                        if text[z+2]==")":
                                return 1
            return 0

        def check_quote():
            i = 0
            for w in words:
                for c in w:
                    if c==":":
                        i = i + 1
                    if c==">":
                        i = i + 1
                if w=="wrote:":
                    i = i + 1
            return i
        
        def check_address():
            y = 0
            states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA','HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM','NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
            for w in words:
                if w.lower()=="fax:":
                    y = y + 1
                if w.lower()=="email:":
                    y = y + 1
                if w in states:
                    y = y + 1
                if w.lower()=="phone:":
                    y = y + 1
                if w.lower()=="tel:":
                    y = y + 1
            return y
        
        def check_sig():
            for w in words:
                if w=="--":
                    return 1
            return 0
        
        def check_graph():
            y = 0
            symbols = ["/","\\","-","|","_","^",")","(","~"]
            #print(symbols)
            for w in words:
                for c in w:
                    if c in symbols:
                        y = y + 1  
            return y

            
        features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            text.count(' '),
            check_head(),
            check_num(),
            check_item(),
            check_quote(),
            check_address(),
            check_sig(),
            check_list(),
            check_graph(),
            # check_pos(),
            # sum(1 if w==pos_dict['NP'] else 0 for w in words),
            sum(1 if w.isupper() else 0 for w in words)   
        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY

# def load_wordlist(file):
#     with open(file) as fin:
#         return set([x.strip() for x in fin.readlines()])


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()