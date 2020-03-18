---
layout: post
title: 📊 Customer Churn
subtitle: A machine learning effort to tackle this problem
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [churn, data-science, machine-learning]
# comments: true
---

**TL;DR:** Hi there!👋 I will be describing my experience of what it takes to solve a business problem with the help of Data Science. 

<!-- 
{: .box-note}
As Jeff Bezos once said, __"We see our customers as invited guests to a party, and we are the hosts. It’s our job every day to make every important aspect of the customer experience a little bit better.”__ 

So, in order to make customer experience more enjoyable, we should focus on improving our products and making them more responsive to the ever-increasing demands of a customer.
But first let us start by discussing what exactly customer churn means? -->

First of all, I would like to say to all the aspiring data scientists and budding ML enthusiasts that if you have done some work which involves any kind of Data Science essence in it, please, please share it just as I am doing by writing down my work. Also, if you are wondering what got me into writing a blog is all thanks to [David Robinson](http://varianceexplained.org/) and his tweet - 

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">When you’ve written the same code 3 times, write a function<br><br>When you’ve given the same in-person advice 3 times, write a blog post</p>&mdash; David Robinson (@drob) <a href="https://twitter.com/drob/status/928447584712253440?ref_src=twsrc%5Etfw">November 9, 2017</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
Yes, by the time I write down this post, I had given numerous presentations and [KT](https://en.wikipedia.org/wiki/Knowledge_transfer) sessions to different stakeholders involved during the course of this project.

So.... what are you guys waiting for? Grab all those required tools and get down to some writing skills.

Enough motivation and let us come back to the topic being discussed here.


### ❓ Defining Customer Churn

Honestly, this was the most uneventful step of the project. 
I started working and the product managers(PM) had done some research at their own level and came up with a definition of churn which was kind of an assumption stating a NiYO Bharat* Customer who hasn't received salary in the past 3 months is classified as churned.

{: .box-note}
**Note:** - * - NiYO Bharat customers are the [blue-collar](https://en.wikipedia.org/wiki/Blue-collar_worker) workers for which NiYO Bharat Salary cards are issued. Also, the churn is predicted for January 2020 taking consideration of vintage data from July-December 2019.

So, we are done with defining what churn actually is in our world.


### 👨‍🔬 Feature Engineering

The most exhaustive and time-consuming step is here. I am going to break down how I managed to build a full list of important features very shortly.

{: .box-note}
**Note:** - You might be wondering how can I start feature engineering when I don't have the required data to carry on my work. Well, you might be right theoretically and it might be true for all the ML Hackathons/Competitions. But in real life, first you need to think, discuss what all features might be important and then write specific SQL queries corresponding to each feature and then start working on the fetched data by merging them together.

- Firstly, I stressed on the fact that I should first know what happens at the ground zero by contacting the sales representative and gather information as to what might be the thought-process of the customers when they interact while onboarding them.
- Then, I had a discussion with my mentor which led to the conclusion of building a 3-level feature set.
- Also, the opinions of PM were considered in making the feature set.

So, Feature Engineering was broadly classified into - 

- Transaction - related features. 
- Demographic - related features.
- Corporate - specific features.
- Clickstream - related features**
- Customer support - related features**
- Customer Account Balance - related features**

{: .box-note}
**Note:** - ** - To be done at a later stage as these data points are still not integrated into the master DB

We have tried to make this v0 project to be simple and focused majorly on identifying the factors and key points leading to a customer churn.

Finally, the data set for model training, with all the steps of Data Cleaning & Feature Engineering & Data Encoding, came out to be a size of around 29,000 rows and 53 columns.


### 🚆 Training the model

I decided to use XGBoost as my base-line model for classifying churn. 

{: .box-note}
**Note:** - If you all want to know more about xgboost, [have a look at](https://shirinsplayground.netlify.com/2018/11/ml_basics_gbm/) and want to know more about it mathematically, then have a look at this [interesting article.](https://medium.com/syncedreview/tree-boosting-with-xgboost-why-does-xgboost-win-every-machine-learning-competition-ca8034c0b283)

The highest F1-score came out to be 83% after multiple parameter tuning. 
<!-- Insert image -->
Once again, I would like to highlight the important point that our main aim was to not get higher accuracy or in this case F1-score, rather identify and analyze the features/factors responsible for a customer churning out of our product.


### 💥 Impact created

- First of all, I stressed on the importance of clickstream data. We should really focus on this untapped data as this is crucial in understanding360-degree customer journey experience.
- I was also responsible for the decision to focus on analyzing the customer support queries to know their intent.
- I also highlighted the decision to implement a knowledge-graph for the NiYO Bharat customer base to understand their actions better.


### 🏁 Wrapping up

You might have noticed that I majorly worked at the intersection of data, engineering, product, and growth. If you got this gist, then I was successful in highlighting what makes it really tick in the real world of Data Science/ML at the industry level. You need to first understand business requirements, understand the product better, discuss the impact you can input, actually build the solution you suggested earlier, validate whether it meets the requirements, and then finally push the model/solution to production to be used in the product line by the customers.

__Hope you all enjoyed reading my experience. Don't forget to share this with your friends. I am open to suggestions and valuable feedback.This was my first post with many more to come further. Till then, keep reading, building, breaking stuff.__ 👍💯






