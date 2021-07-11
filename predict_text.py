import joblib
import sys
from main import TextModel, BertDataset

meta = joblib.load("meta.bin")
lbl_enc = meta["lbl_enc"]

class textClassifier:
    def __init__(self, texts, batch_size = 1, device = "cpu"):
        self.model = TextModel(num_classes = len(lbl_enc.classes_), num_train_steps = meta["num_train_steps"])
        self.model.load("model.bin", device = device)
        self.dataset = BertDataset(inputs, targets = [0] * len(inputs))
        self.batch_size = batch_size

    def pred_text_category(self):
        """
        possible categories : ['business' 'entertainment' 'politics' 'sport' 'tech']
        """
        preds = self.model.predict(self.dataset, batch_size = self.batch_size)
        labels = []
        for pred in preds:
            max = pred.argmax()
            label = lbl_enc.classes_[max]
            labels.append(label)
        return labels


inputs = ["Google is planning to move its headquarter to India", "Shahrukh Khan will shoot his next movie in Maldeevs", "India beats Pakistan in 2011 to win cricket world cup only for the second time"] #list of text
classifier = textClassifier(inputs)
predictions = classifier.pred_text_category()

print(predictions)
