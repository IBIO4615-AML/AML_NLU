# AML_NLU

## 1. Word embedding (1.5 Point)
In the first part of this task you are going to do your own word embedding using word2vec.

1. we are going to use a subset of Amazon reviews from the Cell Phones & Accessories category. Download the datset from http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
2. Open the code available in Codes/embeding.py. 
3. After loading the dataset, you will need to prepare the data of the column reviewText to be converted to numeric vectors. The model should receive an array where each phrase is a row and the words are separated.
```bash
#Example
0         [they, look, good, and, stick, good, just, don...
1         [these, stickers, work, like, the, review, say...
2         [these, are, awesome, and, make, my, phone, lo...
3         [item, arrived, in, great, time, and, was, in,...
4         [awesome, stays, on, and, looks, great, can, b...
                                ...                        
194434    [works, great, just, like, my, original, one, ...
194435    [great, product, great, packaging, high, quali...
194436    [this, is, great, cable, just, as, good, as, t...
194437    [really, like, it, becasue, it, works, well, w...
194438    [product, as, described, have, wasted, lot, of...
```


 
## 3. Attention visualization (1 Point)

Now, we want to visualize the attention mechanism. To do this, visit this repository https://github.com/jessevig/bertviz and do the Interactive Colab Tutorial https://colab.research.google.com/drive/1s8XCCyxsKvNRWNzjWi5Nl8ZAYZ5YkLm_#scrollTo=p_Mlw1LNVIkq. 

1. Choose a sentence to analize.
2. Let's see the self-attention in your sentence.
 ![image](https://user-images.githubusercontent.com/98495468/187809182-826e373d-a57d-4604-b9cb-2ce79f257e5e.png)
#### Note: We want to see the importance of each word give to the others. See also the Query, Key and QxK values (click in the +)
 ![image](https://user-images.githubusercontent.com/98495468/187809410-9536b56a-bd6d-4516-aadd-aba6fcc968d2.png)
3. Analize this interactions in different layers and heads, then, answer the fllowing questions:
* What are the most important words in your sentence?
* Why do you think is the reason of that?
* Do you see differences between the heads? What is the importance of the multi-head attention in this models?

## 4. CLIP: Image captioning classification (1 Point)
In this section we want to test what limitations the CLIP model can have. For this you should think of descriptions that make the classification task difficult.

![image](https://user-images.githubusercontent.com/98495468/187804121-3107c28a-1fc8-47eb-8cf0-e8fb9721da4d.png)

1. Select an image from internet or take a photo that you think could be difficult to classify.
2. Open the Codes/caption_classification.py code. Add your image with the url or by uploading a file.

```bash
#Example
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
```
3. Change the two posible descriptions of the image.
```bash
#Example
inputs = processor(text=["image of two cats","image of two cats sleeping"], images=image, return_tensors="pt", padding=True)
```

4. Run the code and check the probability of each description for your clasiffication result.

5. Try different pairs of descriptions, look for   the most specific descriptions in order to see  limitations of CLIP model.
6. Submit two description that can confuse the model, so that shown similar probability between them. The highest probability should be less than 0.6. Submit also the image.
