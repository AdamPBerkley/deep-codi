# DEEP-CODI (Coronavirus Diagnostic)

### Brief:

The COVID-19 pandemic is severely impacting the health and wellbeing of countless people worldwide. Early detection of infected patients is a crucial first step in controlling the disease, which can be achieved through radiography, according to prior literature that shows COVID-19 causes chest abnormalities noticeable in chest x-rays.

Deep Codi learns these abnormalities and is able to accurately predict whether a patient is infected with coronavirus based on the patient’s chest x-ray. Codi is an effective diagnosis tool that has immediate downstream effects in clinical settings and in the field of radiology.


### ToDo:


[x] Check-in 1: [Outline](https://docs.google.com/document/d/1EEI7X_CQr9wfGwV87lb6Td_VjfkSVE8X5ixjkUxLoks/edit?usp=sharing)
[x] Check-in 1 meet: [11/13](https://brown.zoom.us/j/93398220099) 
[x] [DevPost](https://devpost.com/software/deep-codi-coronavirus-diagnostic?ref_content=user-portfolio&ref_feature=in_progress)
[x] Check-in 2: [Reflection](https://docs.google.com/document/d/1cysJC3PYWxQsm3N-E76wBlRUhUhj_eQgIkaq2fx0ai8/edit?usp=sharing)
[x] Check-in 2 meet: [11/29](https://brown.zoom.us/j/98971593330) at 2:00pm
[x] [DevPost:update](https://devpost.com/software/deep-codi-coronavirus-diagnostic#updates)
[ ] Run the Keras [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) model on the dataset. 
[ ] Evaluate Keras VGG16 with same metrics in this [paper](https://arxiv.org/pdf/2004.09363.pdf) as a benchmark: sensitivity and specificity outlined in section `4.2`; evaluate, also, with `dice_coef` found in `metrics.py`. Share with team and add to README under "METRICS". Compare with results in [paper](https://arxiv.org/pdf/2004.09363.pdf).
[ ] Complete implementation of custom VGG16 in `vgg_model.py`. `filter` and `strides` need to be updated. Model needs to run. Fine-tune model architecture to equal or beat performance of Keras VGG16.
[ ] Circle back to preprocessing. (There is a lot of undesirable data in the COVID positive sets). When we reach this point, talk as team to consider remaining TODOs.
[ ] Ethical Questions, write-up, poster, video presentation
[ ] Check-in 3: week of 11/30 - 12/9
[ ] Check-in 3 meet: TBD
[ ] Final Project is due 12/11


### Data:

The data folder is omitted from the git repo since it is large. For clarity and consistency, the folder structure is:

```
|--code
|--data
|----main_dataset
|------test
|--------covid
|--------non
|----------Atelectasis
|----------Cardiomegaly
|----------Consolidation
|----------Edema
|----------Enlarged_Cardiomediastinum
|----------Fracture
|----------Lung_Lesion
|----------Lung_Opacity
|----------No_Finding
|----------Pleural_Other
|----------Pneumonia
|----------Pneumothorax
|----------Support_Devices
|------train
|--------covid
|--------non
```

Where `main_dataset` has been renamed from the original folder `data_upload_v2`, as additional data sets may be added at a later point.
The `main_dataset` can be found [here](https://github.com/shervinmin/DeepCovid/tree/master/data).


### Metrics:

See TODO.

### Docs:

Documents submitted for this project are conveniently linked here:
* [DevPost](https://devpost.com/software/deep-codi-coronavirus-diagnostic)
* [Outline](https://docs.google.com/document/d/1EEI7X_CQr9wfGwV87lb6Td_VjfkSVE8X5ixjkUxLoks/edit?usp=sharing)
* [Reflection](https://docs.google.com/document/d/1cysJC3PYWxQsm3N-E76wBlRUhUhj_eQgIkaq2fx0ai8/edit?usp=sharing)

### Other:

Please feel free to add / edit.