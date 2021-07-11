# Bert_text_classifier
## Multiclass text classification using Bert transformer.
Run `main.py` to train the model. <br />
After successfully training the model , the weights would automatically be serialized in the same directory as main.py. <br />
Then for inference, run ```predict.py```. <br />
In `predict.py`, a class is defined for inference purpose. You have to create a instance of that classe by passing the list of text as follows

```
text = ["Your text goes here", "your second text goes here", "and so on"]
classifier = textClassifier(text, batch_size = batch_size, device = device) 
predictions = classifier.pred_text_category() 
```
This `pred_text_category()` method would return you a list whose elements would give the predictions, one for individual text from text input list. <br />
Supported categories are `['business' 'entertainment' 'politics' 'sport' 'tech']`
