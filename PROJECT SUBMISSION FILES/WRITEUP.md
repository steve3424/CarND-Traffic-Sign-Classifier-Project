# **Traffic Sign Recognition** 


---
### Writeup / README

#### 1. Link to project files

Here is a link to my project files(https://github.com/steve3424/CarND-Traffic-Sign-Classifier-Project/tree/master/PROJECT%20SUBMISSION%20FILES)



### Architecture design and techniques

As per the suggestion in the project instructions, I started out using the LeNet architecture as is from the previous lab. This gave around an 89% validation accuracy without making any changes.

Then I chose 2 pre-processing techniques. First I converted the images to grayscale. This decision was made for 2 reasons: it was suggested in Yann Lecun and Pierre Sermanet's paper describing the use of LeNet for classifying traffic signs (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and it gave performance benefits.

The second technique was to normalize the images to values between -1 and 1. I tried 2 other normalizing techniques (0 to 1 and -0.5 to 0.5), but this provided the most performance boost.

This was all the pre-processing that was done before training.


#### 2. Final model architecture

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
 


#### 3. Model training

To train the model, I used the mean cross entropy as the loss operation with the AdamOptimizer to reduce the error. I used a learning rate of 0.0005 and ran the training for 30 epochs. I also included a dropout rate of 0.6.

#### 4. Strategies to improve accuracy

My final validation accuracy was 96.5% 

I did this by starting with the LeNet architecture and going from there. The only changes I made to the actual architecture was to change the activation function from RELU to TANH. This was an arbitrary decision made when experimenting with different activation functions. This proved to be the best one. I also added dropout with a rate of 0.6 probability.

Then I tuned just 2 of the hyperparameters. I cut the learning rate from 0.001 to 0.0005 and increased the EPOCHS from 10 to 30. Both of these changes and pre-processing provided performance improvements and brought my validation accuracy to over 93%.

 

### Test a Model on New Images

#### 1. Prediction results on new images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/hr      		| 50 km/hr   									| 
| no passing     		| no passing 									|
| right of way			| beware of ice/snow							|
| straight or left	    | straight or left					 			|
| keep right			| keep right      								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The one image that was misclassified was probably due to the fact that the image was rotated a bit. If I had trained on augmented data, it may have eliminated this issue and classified this correctly.

#### 3. Top 5 probs on new images

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

