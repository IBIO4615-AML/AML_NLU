# AML_NLU

## 3.CLIP: Image captioning classification (1 Point)

![image](https://user-images.githubusercontent.com/98495468/187804121-3107c28a-1fc8-47eb-8cf0-e8fb9721da4d.png)

1. Select an image from internet or take photo that you think is difficult to classify.
2. Use the CLIP/caption_classification.py and change the posible descriptions to the image.

```bash
#Example
inputs = processor(text=["image of two cats","image of two cats sleeping"], images=image, return_tensors="pt", padding=True)
```

3.See in the line 16 the probability of each description.
4. Try to do the most specific descriptions in order to see the limitation in CLIP model.
5. Submit two description that can make the model  that shown similar probability between them. The maximum probability should be less than 0.6.
