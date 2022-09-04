##
import gensim
import pandas as pd
##
df = pd.read_json("reviews_Cell_Phones_and_Accessories_5.json", lines=True)
##
#TODO:tokenize the text reviews
text=
#TODO:init your model choose the parameters.
model = gensim.models.Word2Vec(vector_size=, window=)
##
model.build_vocab(text, progress_per=1000)
model.train(text, total_examples=model.corpus_count, epochs=model.epochs)
##
voc=model.wv.key_to_index
vec=model.wv.get_normed_vectors()
the=vec[voc['the']]
print('the:', the)
