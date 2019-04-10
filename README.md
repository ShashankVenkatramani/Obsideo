## Inspiration
Our goal is to both simplify the process of transferring money while also respecting the privacy and right to anonymity of the users. We saw a lot of issues with Venmo, including their huge lack of security and fraud problems, and wanted to change that.

## What it does
This app leverages (centralized) blockchain technology and databases combined with a beautiful frontend iOS to provide an extremely powerful money transfer app.

- When you are making a transaction, this app simplifies the process with the option of face ID or a biometric scanner.
- We do not store any information regarding our users, except for their email which they can keep anonymous.
- Our database is fully secure and can not be associated with an identity, which creates privacy.


## How we built it

**Frontend:**

We wanted obsideo to be accessible to all age groups, so having an unique UI was important. We had multiple drawings of the app and used Sketch to design and create them.
After incorporating our UI and backend system we used Xcode and Swift to create the application. We had many viewcontrollers for specific purposes.
After creating the app, we had our team navigate and extensively test the application, before being satisfied with our project just in time for submission.

**Backend:**

To create Obsideo we used a custom Block and Chain model, which had a chain made up of multiple individual Blocks, each containing a hash, previous hash, a nonce, and some data. 
In the database itself, the data is taken from the database, processed, and then a model is created. Once all the nonces have been calculated, the data is uploaded to the database as individual blocks of 3 transactions.
We also have an XGBoost classifier which we made but did not include in our final model due to size and processing power constraints. This model is able to classify a transaction as fraudulent or genuine. 
In order to make this model,~350 lines of Python preprocessing, EDA, and data cleaning where done. An extremely comprehensive EDA, when combined with proper data processing, makes our model robust and optimized, achieving 99.6% accuracy on random samples (consistent 99.4% + avg).


## Challenges we ran into
We had a lot of issues on the backend with the model creation, where coreml was extremely finicky with the XGB classifier we had trained. It only accepted XGB models, and we had a huge issue transferring the classifer to a model. Once that was finally done, after many numpy and Dmatrix errors, we had a final model finished. However, by this point, we were too late; time had essentially ran out.

Pyrebase also had some issues with child setting and updating and pushing data, but these were easier to fix.

We also tried to implement gossip protocol, but it proved to be insanely difficult. After a few hours spent trying to implement it, we moved on to other things.

## Accomplishments that we're proud of
**Swift** - Finished a full XCode application, used a collection view for a stunning UI, and extensive backend for uploading and downloading data

**Backend** - Our initial machine learning model was 350 lines of code of in-depth EDA, preprocessing, and model building using XGBoost. We had many issues with numpy, matplotlib, and converting the XGBoost model to coreml, but eventually, we succeeded. However, we did not have sufficient time to incorporate the model into the iOS app. When it comes to the blockchain backend of the app, we have a block and chain class implemented, along with a method to continuously download from the database and upload the hashed values

**Firebase** - Created a data model with arrays and objects to manage different transaction types, with their corresponding user ids (transferer and transfer recipient), values, and hashes.

## What we learned
- A lot about blockchain
- How to do transaction software
- iOS development
- Gradient boosting models and technology
- Some fancier website development
- using Pyrebase

## What's next for Obsideo
- Implement gossip protocol
- Add website integration

Authentication screen

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/795/274/datas/gallery.jpg)

Transactions home screen

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/795/294/datas/gallery.jpg)

Showcase website

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/795/295/datas/gallery.jpg)
