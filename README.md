# neural-network-class-competion
Class competitions for Neural Networks 2026
Ranked #4 in class competition

The learning objectives of this assignment are to:

1. build a neural network for a text sentiment classification task 
2. practice tuning model hyper-parameters

# Create a CodaBench account

You must create a CodaBench account and join the competition:

1. Visit the competition website: [https://www.codabench.org](https://www.codabench.org)

2. In the upper right corner of the page, you should see a "Sign Up" button. Click that and go through the process to create an account. **Please use your @arizona.edu account when signing up.** Your username will be displayed publicly on a leaderboard showing everyone's scores. **If you wish to remain anonymous, please select a username that does not reveal your identity.** Your instructor will still be able to match your score with your name via your email address, but your email address will not be visible to other students. 

3. While logged in to your CodaBench account, use the following url to access the competition website:
[https://www.codabench.org/competitions/15425/?secret_key=1ceae1d3-6564-4ab4-818c-3b8674a6204e](https://www.codabench.org/competitions/15425/?secret_key=1ceae1d3-6564-4ab4-818c-3b8674a6204e) .
Click the "My Submissions" tab, accept the terms and conditions of the competition, and register for the task.

4. Wait for your instructor to manually approve your request. This may take a day or two. If three days have passed and you have not been admitted, email the instructors.

5. After you have been admitted to the competition, you should then be able to return to the "My Submissions" tab and see a "Submission upload" form. Seeing this means you are fully registered for the task.

# Read about the CodaBench Competition

You will be participating in a class-wide competition.
The direct link to the competition website is:

[https://www.codabench.org/competitions/15425](https://www.codabench.org/competitions/15425)

(You will not be able to resolve this direct link until you have logged in with a CodaBench account.)

You should visit the site and read all the details of the competition, which include the task definition, how your model will be evaluated, the format of submissions to the competition, etc.

# Clone the repository

Clone the repository created by GitHub Classroom to your local machine:

```
git clone https://github.com/ua-2026-spring-nn/graduate-project-<your-username>.git
```

You are now ready to begin working on the assignment.

# Download the data

Go to the "Get Started" tab on the CodaBench site, and click on the "Files" sub-tab. You should see a button to download the training and dev (validation) data for the task. Download and unzip that data into your cloned repository directory.

Please **do not commit the data to the repository**.

When the test phase of the competition (Test Phase) begins, you may return to the "Files" tab to download the unlabeled test data for the task.

# Write your code

You should design a neural network model to perform the task described on the CodaBench site. Your code should train a model on the provided training data and you can use the dev (validation) data to tune your model's hyper-parameters. Your code should be able to make predictions on either the dev data or the test data. Your code should package up its predictions in a `submission.zip` file, following the formatting instructions on CodaBench.

You must create and train your neural network using the TensorFlow/Keras framework that we have been using in the class. You should train and tune your model using the training and development data that you downloaded from the CodaBench site.

**If you would like to use any additional resource to for developing your model, you must first ask for permission by contacting the instructors by email (Clay and Rishabh)**

The `nn.py` file in this repository provides sample code to help get you started. This code is described briefly on the CodaBench site. You can modify or build on this code as you wish, or you could delete the code entirely and start from scratch if you prefer. In either case, you should still place your model's code in `nn.py` for your final submission.

# Development Phase: Test your model predictions on the dev (validation) set

During the Development Phase of the competition, the CodaBench site will expect predictions on the dev set. To test the performance of your model, run your model on the dev input data, format your model predictions as instructed on the CodaBench site, and upload your model's predictions using the "My Submissions" tab of the CodaBench site.

During the Development Phase, you are allowed to upload predictions many times. You are **strongly** encouraged to upload your model's dev set predictions to CodaBench after every significant change to your code to make sure you have all the formatting correct.

# Test Phase: Submit your model predictions on the test set

When the Test Phase of the competition begins (consult the CodaBench site for the exact timing), the instructor will update the CodaBench site to expect predictions on the test set, rather than predictions on the development set. The instructor will also release the unlabeled (i.e., just the input) version of the test set on CodaBench as described above under "Download the Data". To submit your model's predictions on the test data, download the test data, run your model on the test data, format your model predictions as instructed on the CodaBench site, and upload your model's predictions using the "My Submissions" tab of the
CodaBench site.

During the Test Phase, you are allowed to upload predictions only once. This is why it is critical to debug any formatting problems during the development phase.
 
# Grading

You will be graded first by your model's accuracy, and second on how well your model ranks in the competition. If your model achieves better accuracy on the test set than the baseline model included in this repository, you will get at least a B. If your model achieves better accuracy on the test set than a second baseline that the instructor will reveal after the competition, you will get an A. All models within the same letter grade will receive scores that are distributed evenly across the grade range, based on their rank. So for example, the highest ranked model in the A range will get 100%, and the lowest ranked model in the B range will get 80%.

**You can build off of existing open source models, but if you do so, you must let the instructors know. If you do not obtain prior permission from the instructors, you will lose 10% from your final score.**

**You are NOT allowed to use external or proprietary services/models during hte training/execution of your model. This means that you cannot incorporate calls to GPT, Gemini, Claude, Mistral, Grok, etc. The purpose of this project is for you to gain experience constructing and training your own model.**
