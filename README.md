# Amazon-Apparel-Recommendation-System
Content based recommendation system 
it was project of workshop at [AAIC](http://www.appliedaicourse.com) all necessary data were provided by [AAIC](http://www.appliedaicourse.com)
## requirements ##
      anaconda , tensorflow , nltk, gensim,keras 
      or you can install each module manually 

## Algorithm ##
     1. load json file that contains data of products using pandas
     2. remove those rows that doesn't contain that feature's value that you are going to
        include in your model or that feature is null 
     3. remove duplicate rows or which have very less difference
     4. remove stop words
     5. try bag of words , tf-idf and idf  based model for text based similarity and
        keep that model which fit best to your requirements
     6. try different versions of  word2vec model for schematic based similarity 
     7. combine text based similarity model and schematic based model using weighted word2vec
     8. now if you want to give some weights to your features ( i have used color and brand) 
         and want to use along with title then for every title make a d-dimension vector
         where d is number of unique elements in  that feature using one hot encoding 
         let's call it extra features
     9. now combine vectors we get from word2vec model and extra features e.g word2vec have 300-d vector and
        extra feature have n-d vector then you should get 300+n -d vector for each row
     10. Inorder to get image based similarity use keras  tensorflow to convert images into d-dimensional vector
         using VGG16.let's say this data cnn
     11. now combine both vectors (word2vec + extrafeature) + cnn

## output would look like this ##
    in this image first image is query image and rest images are recommended images
![Alt text](img.png?raw=true "output")
