# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/steve3424/CarND-Traffic-Sign-Classifier-Project/blob/master/PROJECT%20SUBMISSION%20FILES/Traffic_Sign_Classifier.ipynb)



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As per the suggestion in the project instructions, I started out using the LeNet architecture as is from the previous lab. This gave around an 89% validation accuracy without making any changes.

Then I chose 2 pre-processing techniques. First I converted the images to grayscale. This decision was made for 2 reasons: it was suggested in Yann Lecun and Pierre Sermanet's paper describing the use of LeNet for classifying traffic signs (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and it gave performance benefits.

The second technique was to normalize the images to values between -1 and 1. I tried 2 other normalizing techniques (0 to 1 and -0.5 to 0.5), but this provided the most performance boost.

This was all the pre-processing that was done before training.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image						| 
| Convolution 5x5x6    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| tanh					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| tanh					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 5x5x6 					|
| flatten				| output 400									|
| Fully connected		| output 120  									|
| tanh					|												|
| Dropout				| keep_prob 0.6									|
| Fully connected		| output 84  									|
| tanh					|												|
| Dropout				| keep_prob 0.6									|
| Output layer			| output = n_classes = 43						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the mean cross entropy as the loss operation with the AdamOptimizer. I used a learning rate of 0.0005 and ran the training for 30 epochs. I also included a dropout rate of 0.6.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final validation accuracy was 96.5% 

I did this by starting with the LeNet architecture and going from there. The only changes I made to the actual architecture was to change the activation function from RELU to TANH. This was an arbitrary decision made when experimenting with different activation functions. This proved to be the best one. I also added dropout with a rate of 0.6 probability.

Then I tuned just 2 of the hyperparameters. I cut the learning rate from 0.001 to 0.0005 and increased the EPOCHS from 10 to 30. Both of these changes and pre-processing provided performance improvements and brought my validation accuracy to over 93%.

 

### Test a Model on New Images

#### 1. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/hr      		| 50 km/hr   									| 
| no passing     		| no passing 									|
| right of way			| beware of ice/snow							|
| straight or left	    | straight or left					 			|
| keep right			| keep right      								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The one image that was misclassified was probably due to the fact that the image was rotated a bit. If I had trained on augmented data, it may have eliminated this issue and classified this correctly.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

correct label in bold

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| **50 km/hr**   								| 
| .00066   				| 30 km/hr 										|
| .00001				| stop											|
| .00001				| 80 km/hr							 			|
| .000007				| wild animals crossing      					|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			| **no passing**  								| 
| .10   				| 30 km/hr 										|
| .015					| stop											|
| .015					| 80 km/hr							 			|
| .003					| wild animals crossing      					|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .71         			| Beware of ice/snow   							| 
| .09   				| Children crossing 							|
| .06					| Bicycles crossing								|
| .06					| **Right-of-way**						 		|
| .03					| Roundabout mandatory      					|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| **Go straight or left**   					| 
| .001   				| Roundabout mandatory 							|
| .0002					| Traffic signals								|
| .0002					| No entry							 			|
| .0001					| Bumpy road    								|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| **Keep right**   								| 
| .00066   				| End of speed limit (80km/h) 					|
| .00001				| Turn left ahead								|
| .00001				| End of no passing by vehicles over 3.5 metric tons	|
| .000007				| Speed limit (20km/h)      					|

