# ResNet50-vd
ResNet50-vd is from "Bag of Tricks for Image Classiﬁcation with Convolutional Neural Networks".
![](https://tva1.sinaimg.cn/large/008i3skNgy1guf268up3fj60ba0ioglz02.jpg)

### Compare with ResNet50
- Modify the conv7x7 to (conv3x3->conv3x3->conv3x3).
- Modify the stage downlayer, (conv1x1 s=2 -> conv3x3 -> conv1x1) to (conv1x1 -> conv3x3 s=2 -> conv1x1) and add the avgpool(2) on the short cut which before the conv1x1.

### Accuracy
|model|bs|lr|epoch|trick|acc@top-1|
|:---:|:---:|:---:|:---:|:---:|:---:|
|R50-vd|1024|0.4|300|labelsmooth+mixup|78.25%|

If you want use R50-vd model,you can get the imagenet1k pretrain weights, from [here](https://drive.google.com/file/d/1Llh0ZYqbVTvbxzyb1CNVMBpG03DDLbiR/view?usp=sharing).


### Useage
```python
from resnetd import resnet50
model = resnet50(pretrainde=False, num_classes=1000)

# if you want use the pretrain
if pretrain:
    state_dict = torch.load("r50vd.pth", map_location="cpu")['state_dict']
    model.load_state_dict(state_dict, strict=True)

```




