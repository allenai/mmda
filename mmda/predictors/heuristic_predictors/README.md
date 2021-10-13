# heuristic predictors

### example usage - word predictor

```
from mmda.types.document import Document
from mmda.predictors.heuristic_predictors.dictionary_word_predictor import DictionaryWordPredictor

# instantiate a Document
with open('...') as f_in:
    doc_dict = json.load(f_in)
    doc = Document.from_json(doc_dict=doc_dict)

# instantiate the word predictor.  file_path can be an empty file.
dictionary_word_preditor = DictionaryWordPredictor(dictionary_file_path='...')

# predict words
words = dictionary_word_preditor.predict(document=doc)

# add them to Document
doc.annotate(words=words)

# examine them in relation to tokens
for word in doc.words:
    if len(word.tokens) > 1:
        print(f'Word={word.text} \t\t Tokens{[token.symbols for token in word.tokens]}')
        
        
Word=theorems 		        Tokens[['theo-'], ['rems']]
Word=voting 		        Tokens[['vot-'], ['ing']]
Word=pre-serves 		Tokens[['pre-'], ['serves']]
Word=Condorcet’s 		Tokens[['Con-'], ['dorcet’s']]
Word=preferences 		Tokens[['prefer-'], ['ences']]
Word=eval-uated 		Tokens[['eval-'], ['uated']]
Word=distributions, 	        Tokens[['dis-'], ['tributions,']]
Word=widely-accepted 	        Tokens[['widely-'], ['accepted']]
Word=large-scale 		Tokens[['large-'], ['scale']]
Word=asymptotic 		Tokens[['asymp-'], ['totic']]
Word=impossibility 		Tokens[['im-'], ['possibility']]
Word=theorem 		        Tokens[['theo-'], ['rem']]
Word=commonly-studied           Tokens[['commonly-'], ['studied']]
Word=high-dimensional           Tokens[['high-'], ['dimensional']]
Word=interest 		        Tokens[['in-'], ['terest']]
Word=constraints, 		Tokens[['con-'], ['straints,']]
Word=computational 		Tokens[['compu-'], ['tational']]
Word=tech-nically 		Tokens[['tech-'], ['nically']]
Word=analysis 		        Tokens[['analy-'], ['sis']]
Word=framework 		        Tokens[['frame-'], ['work']]
Word=Lepel-ley 		        Tokens[['Lepel-'], ['ley']]
```