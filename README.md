# AML_NLU

## 3. CLIP: Image captioning classification (1 Point)
In this section we want to test what limitations the CLIP model can have. For this you should think of descriptions that make the classification task difficult.

![image](https://user-images.githubusercontent.com/98495468/187804121-3107c28a-1fc8-47eb-8cf0-e8fb9721da4d.png)

1. Select an image from internet or take a photo that you think could be difficult to classify.
2. Open the CLIP/caption_classification.py code. Add your image with the url or by uploading a file.

```bash
#Example
inputs = processor(text=["image of two cats","image of two cats sleeping"], images=image, return_tensors="pt", padding=True)
```
3. Change the two posible descriptions of the image.
```bash
#Example
inputs = processor(text=["image of two cats","image of two cats sleeping"], images=image, return_tensors="pt", padding=True)
```

4. Run the code and check the probability of each description for your clasiffication result.

5. Try different pairs of descriptions, look for   the most specific descriptions in order to see  limitations of CLIP model.
6. Submit two description that can make the model  that shown similar probability between them. The highest probability should be less than 0.6.
