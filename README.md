<h1 style="text-align:center;font-size:30px;" > Quora Question Pairs </h1>
<img src  = 'https://www.learnopencv.com/wp-content/uploads/2018/12/Quora-Post-Image.jpg'>

<h1> Business Problem </h1>
<h2> Description </h2>
<p>Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.</p>
<p>
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
</p>
<br>
> Credits: Kaggle 
</p>

__ Problem Statement __
- Identify which questions asked on Quora are duplicates of questions that have already been asked. 

<h2> Real world/Business Objectives and Constraints </h2>
- The cost of a mis-classification can be very high.</br>
- You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.</br>
- No strict latency concerns.</br>
- Interpretability is partially important.</br>

<h1> Machine Learning Probelm </h1>
<h3> Data Overview </h3>
<p> 
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. Total we have 404290 entries. Splitted data into train and test with 70% and 30%.

I derived some features from questions like no of common words, word share and some distances between questions with the help of word vectors. will discuss those below
</p>
<h3> Example Data point </h3>

<pre>
"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
</pre>

<h2> Mapping the real world problem to an ML problem </h2>
<h3> Type of Machine Leaning Problem </h3>
<p> It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. </p>

<h3> Performance Metric </h3>
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s): 
* log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
* Binary Confusion Matrix

<h2> Approach to solve to problem</h2>
<br>I approached this problem in 3 steps implementation:</br>

<### Feature Extraction:
- ##### Basic Features - Extracted some features before cleaning of data as below.
  - <b>freq_qid1</b> = Frequency of qid1's
  - <b>freq_qid2</b> = Frequency of qid2's
  - <b>q1len</b> = Length of q1
  - <b>q2len</b> = Length of q2
  - <b>q1_n_words</b> = Number of words in Question 1
  - <b>q2_n_words</b> = Number of words in Question 2
  - <b>word_Common</b> = (Number of common unique words in Question 1 and Question 2)
  - <b>word_Total</b> =(Total num of words in Question 1 + Total num of words in Question 2)
  - <b>word_share</b> = (word_common)/(word_Total)
  - <b>freq_q1+freq_q2</b> = sum total of frequency of qid1 and qid2
  - <b>freq_q1-freq_q2</b> = absolute difference of frequency of qid1 and qid2
- ##### Advanced Features - Did some preprocessing of texts and extracted some other features.Used fuzzywuzzy 
  - <b>cwc_min</b> = common_word_count / (min(len(q1_words), len(q2_words)) 
  - <b>cwc_max</b> = common_word_count / (max(len(q1_words), len(q2_words)) 
  - <b>csc_min</b> = common_stop_count / (min(len(q1_stops), len(q2_stops)) 
  - <b>csc_max</b> = common_stop_count / (max(len(q1_stops), len(q2_stops)) 
  - <b>ctc_min</b> = common_token_count / (min(len(q1_tokens), len(q2_tokens)) 
  - <b>ctc_max</b> = common_token_count / (max(len(q1_tokens), len(q2_tokens)) 
  - <b>last_word_eq</b> = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
  - <b>first_word_eq</b> = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
  - <b>abs_len_diff</b> = abs(len(q1_tokens) - len(q2_tokens))
  - <b>mean_len</b> = (len(q1_tokens) + len(q2_tokens))/2
  - <b>fuzz_ratio</b> = How much percentage these two strings are similar, measured with edit distance.
  - <b>fuzz_partial_ratio</b> = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.
  - <b>token_sort_ratio</b> = sorting the tokens in string and then scoring fuzz_ratio.
  - <b>longest_substr_ratio</b> = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
- ##### Extracted Tf-Idf features for this combained question1 and question2 and got 1,2,3 gram features with Train data. Transformed test data into same vector space. 
- ##### Got [Word Movers Distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) with pretrained glove word vectors. 

### Machine Learning Models:
   - Trained a random model to check Worst case log loss and got log loss as 0.887699
   - Trained some models and also tuned hyperparameters using Random and Grid search.
   ##### Tokenizer: TFIDF Weighted W2V

<p>After that we have applied Logistic Regression on ~100K dataset with hyperparameter tuning, which producs the log loss of 0.50, which is significantly lower than Random Model.We have applied Linear SVM on ~100K dataset with hyperparameter tuning, which produces the log loss of 0.48, which is slightly lower than Logistic Regression.We applied XGBoost Model on ~100k dataset with no hyperparameter tuning, which produces the log loss of 0.35, which is significantly lower than Linear SVM.
Finally, we applied XGBoost Model on ~100k dataset with hyperparameter tuning, which produces the log loss of 0.34, which is slightly lower than XGBoost Model with no hyperparameter tuning.As we know that, on high dimension dataset 'XGBoost' does not perform well, but it does perform well in above dataset because of low dimension of 794.Whereas 'Logistic Regression' and 'Linear SVM' performs moderately on low dimension data.To test this, we performed 'Logistic Regression' and 'Linear SVM' on complete ~400k dataset, and we got better results as compared to above models.<p>
 </br>
 
  ##### Tokenizer: TFIDF
  
<p> First we have applied simple Random Model(Dumb Model), which gives the log loss of 0.74, that means, the other models has to produce  less than 0.74.After that we have applied Logistic Regression on ~400K dataset with hyperparameter tuning, which produces the log loss of 0.45, which is significantly lower than Random Model, also it is lower than previous logistic regression model(performed using TFIDF Weighted W2V).We have applied Linear SVM on ~400K dataset with hyperparameter tuning, which produces the log loss of 0.45, which is similar to Logistic Regression, but it is lower than previous Linear SVM model(performed using TFIDF Weighted W2V).
Finally for this case study, we conclude that on low dimesion data,we will use hyperparameter tuned 'XGBoost' model and for high dimension data we will use either 'Linear SVM' or 'Logistic Regression'<p>
