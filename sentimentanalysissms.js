
const tf = require("@tensorflow/tfjs");

const fetch = require("node-fetch");
client = require("twilio")(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

const setup = async () => {
  var texts = [];
   await client.messages
     .list({
       dateSentBefore: new Date(Date.UTC(2019, 11, 31, 0, 0, 0)),
       dateSentAfter: new Date(Date.UTC(2019, 0, 1, 0, 0, 0)),
       from: "+16507878004",
       limit: 500
     })
     .then(messages => messages.forEach(m => texts.push(m.body)));
  return Array.from(new Set(texts)); //only unique texts
}

const getMetaData = async () => {
  const dt = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
  return dt.json()
}

const predict = (text, model, metadata) => {
  //tokenize text: remove non-alphanumeric chars besides spaces, apos
  var trimmed = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, "").split(/\s+/g);
  //look up word indices
  const inputBuffer = tf.buffer([1, metadata.max_len], "float32"); //buffer is an array of integers defining output tensor shape, type
  //fill buffer with trimmed words and their indices
  trimmed.forEach((word, i) => inputBuffer.set(metadata.word_index[word] + metadata.index_from, 0, i));
  const input = inputBuffer.toTensor(); //returns Tensor obj
  const predictOut = model.predict(input);
  let positivity = predictOut.dataSync()[0]; //output data retrieved
  predictOut.dispose();
  //some words aren't in the English dictionary and TFJS doesn't like these words
  if(isNaN(positivity)) {
    return  0;
  }
  return positivity;
}

async function run(text) {
  //load pretrained model at remote URL
  const url = `https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json`
  const model = await tf.loadLayersModel(url); 
  const metadata = await getMetaData();
  var sum = 0, avg = 0;
  text.forEach(function (prediction) {
    avg = predict(prediction, model, metadata);
    sum += parseFloat(avg, 10);
    console.log(`text ${prediction} avg ${avg}`);
  })
  console.log(getSentiment(sum/text.length));
}
const getSentiment = (score) => {
  if (score > 0.66) {
    return `Score of ${score} is Positive`;
  }
  else if (score > 0.4) {
    return `Score of ${score} is Neutral`;
  }
  else {
    return `Score of ${score} is Negative`;
  }
}

setup().then(function(result) {
  run(result); 
});