# analyze-2019-with-tensorflow-twilio-texts

Presented at the first TensorFlow.js Community Show and Tell and found on the [Twilio blog here](https://www.twilio.com/blog/how-positive-was-your-year-with-tensorflow-js-and-twilio).

### Prerequisites
- [Make a Twilio account, a trial one is okay](https://www.twilio.com/try-twilio)
- [A Twilio phone number with SMS capabilities](https://www.twilio.com/console/phone-numbers/search)
- [Node.js installed](https://nodejs.org/en/download/)

This uses a TensorFlow.js-provided, pre-trained [model](https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json) trained on a set of 25,000 movie reviews from IMDB, given either a positive or negative sentiment label, and two model architectures to use: CNN or LSTM. This post will be using the CNN.

Replace 
```
process.env.TWILIO_ACCOUNT_SID,
process.env.TWILIO_AUTH_TOKEN
```
with your Twilio account credentials and then run `node sentimentanalysissms.js` on the command line to run this file and see how your year went based on text messages sent from your Twilio client/phone numbers.
To test the sentiment analysis on any string, you can change
```
setup().then(function (result) {
  run(result);
});
```
to 
```
run([]);
```
