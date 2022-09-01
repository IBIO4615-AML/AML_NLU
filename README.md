# AML_NLU

## 3. Attention visualization (1 Point)

Now, we want to visualize the attention mechanism. To do this, visit this repository https://github.com/jessevig/bertviz and do the Interactive Colab Tutorial https://colab.research.google.com/drive/1s8XCCyxsKvNRWNzjWi5Nl8ZAYZ5YkLm_#scrollTo=p_Mlw1LNVIkq. 

1. Choose a sentence to analize.
2. Let's see the self-attention in your sentence.
![image](https://user-images.githubusercontent.com/98495468/187809182-826e373d-a57d-4604-b9cb-2ce79f257e5e.png)
### Note: We want to see the importance of each word give to the others. See also the Query, Key and QxK values (click in the +)
![image](https://user-images.githubusercontent.com/98495468/187809410-9536b56a-bd6d-4516-aadd-aba6fcc968d2.png)
3. Analize this interactions in different layers and heads, then, answer the fllowing questions:
-> d

## 4. CLIP: Image captioning classification (1 Point)
In this section we want to test what limitations the CLIP model can have. For this you should think of descriptions that make the classification task difficult.

![image](https://user-images.githubusercontent.com/98495468/187804121-3107c28a-1fc8-47eb-8cf0-e8fb9721da4d.png)

1. Select an image from internet or take a photo that you think could be difficult to classify.
2. Open the CLIP/caption_classification.py code. Add your image with the url or by uploading a file.

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
