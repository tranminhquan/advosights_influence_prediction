# AF score specification - version 4.
---
R: Quan Tran

Modification date:
* 2 - June - 2020 (v4.0)
* 3 - June - 2020 (v4.0)
---

# Agenda
1. [Introduction](#introduction)
2. [Context](#context)
3. [Scope](#scope)
4. [KPIs](#kpis)
5. [Limitation](#limitation)
<!-- 6. Improvement -->

---

# 1. Introduction <a id= 'introduction' />

Amplifcation factor (AF) is one of $3$ metrics of Advosights used to measure the **amplification of an individual in a specific network where compagin are launched**.  

Generally, AF score requires $3$ kinds of data: *User attributes*, *User behaviors* and *Posts* (Figure below). These data forms the structural data called **Directed Knowledge Graph**.

![af_input_output](https://i.imgur.com/yTB518v.png)


## For a brand

Generally, the purpose of AF is to increase sale of business. In particular, by measuring the AF of users, we can find out:
* Which users are the most influencers in a specific network
* Predict how many users are influenced by one given user
* Predict which direction of information when a user submit a new post

> *For example, brand X owns a small community in Facebook (e.g, a FB group of X). They aim to find out which users are most influencers among ones to make them as potential KOLs. Thanks to their large and fast diffusion, when they submit posts mentioning to X, other users are able to easily get the information very quickly.*

## For a specific user

When taking part in an advocated community of a brand, a user intents to increase the amplication factor as much as possible due to:
* The user are going to be more potential to be hired by the brand
* AF model is able to predict how much user's new post will be amplified in his community
* By monitoring the AF score, users are able to know things needs to improve
* User can directly measure AF to an other specific users

> *For example, user Y is in an advocated member in group G of brand X. AF score is calculated by attributes of Y and others users in G. When user Y intents to submit a new post, Y is able to be predicted who users are likely to be influenced by that post. Y is also able to measure AF to a specific user Z in G to know whether Z are influenced by Y or not*


# 2. Context <a id='context' />

AF score requires the commuinty networks among users. In particular, it acquires **set of users which some of them connected to each others** to form a **graph network** (see below figures).

![graph_example](https://i.imgur.com/IxqBzxn.gif)

Therfore, our AF score is assumpted to formulated by *user's attributes*, *user's relationship* and *the corresponding posts*:
* **User's attributes** are behavior of a user, these includes their public information and their interaction on some specific social networks in our purpose.

* **User's relationship** are the relationships between the concerned user and the others in a network.

* **User's posts** are the content of previous posts has been submited.

The figure below visualizes the dependence of AF on user's relationship (user's attributes) and posts.  
![AF_assumption](https://i.imgur.com/Ob1a8Q5.png)

Since we defined our problem as a knowledge graph which handles the relationship among users, there are $3$ things need to be concerened: *users' attributes*, *users' relationship* and *their corresponding posts*.

### User's attributes
Users' attributes contains **their public information** and **behavior** in an social network. For a specific social context, these attributes can vary depends on the supplied services of that social network

* version 4.0:  
The current version is being experimented on Facebook database. As a result, we concern on following $7$ attributes (not including **id** of users:
    * **total_followers**: number of followers
    * **total_friends**: number of friends of a user
    * **books_count**: number of books user
    * **films_count**: number of films user
    * **music_count**: number of music user
    * **restaurants_count**: number of restaurants user
    * **sex**: gender ($3$ if private)

*All the missing values were completely filled by mean of the dedicated ones*


### User's relationship
Is the way we define the relationship among users depending on specific social network.

* version 4.0:  
The current version is being experimented on Facebook social network. As a result, **we define the relationship when a user reacts, comments or shares on any post from other user** (detail in below figures). Then, we calculate the **weight of relationship** as below formula. The underlying weights are partly referenced from version 2.
  
                              weights = 0.1*number of reactions + 0.3*number of comments + 0.6*number of shares

![edge_definition](https://i.imgur.com/l0iqTKO.png)

### User's posts
Are all the attributes of a specific user's posts depending on specific social netork. **By crawling posts of users, we are able to analyse useful features such as which topics user interested in, writing style of user, etc**. This also helps us know which style of post that user can be easily influenced, since each user will prefer a specific one.
* version 4.0:  
We currently focus on posts' contents that user submited


## 3. Scope <a id='scope' />
Below is all attributes requirements and their corresponding scope

| Name | Scope | Requirement |
|:------|:------|:------|
| Number of users | $> 100$ | - Including KOLs and users who interact with them. <br> - All users must have at least $1$ connection with other user (avoid isolated user) |
| User attributes | $7$ attributes: total_followers, total_friends, <br> books_count, films_count, music_count, restaurants_count, sex | |
| Number of posts per user | $> 10$ | Posts have: <br> - Content <br> - Number of reactions, shares, comments <br> - Range of time: at least $> 10$ posts $1$ latest month  <br> - Information of corrsponding user ids who interact|

# 4. Scenarios
Due to the variance of data provided by customers in compagins that they are unable to sufficently supply full of requested information. We provide $3$ different scenarios that efficiently support for the customer, vary from *non-information* to *full-information*. As a consequence, there is a tradeoff among scenerios. The below table will provide in detail:

| Scenerio | Used method | Attribute needed | Accuracy | Dedicated version |
|:---------|:------------|:-----------------|:---------|:------------------|
| $1$ | Node2vec | Only relationship among users needed | Low | v4.0 |
| $2$ | GAE or VGAE | User's relationship and attribute | Medium | v4.1 |
| $3$ | GAE, VGAE + Word embedding | Above + Post content of user | High | v4.2 |


# 5. KPIs <a id='kapis' />

* [x] AF score version 4. proposal
* [x] Overall data analytics
* [x] Data preprocessing and transformation
* [x] Training user embedding model
* [ ] Training post embedding model
* [ ] Preparing training data for AF model
* [ ] Training AF model
* [ ] Release


# 6. Limitation <a id='limitation' />
* Since AF score takes **deep learning** into account, the efficency of AF score depends on the quality of data source and preprocessing method. *For the current version (4.), we just take very simple preprocessing method.*
* User's attributes are invariance, depends on specific kind of social network.
* User's attributes reach affordable number of attributes.


