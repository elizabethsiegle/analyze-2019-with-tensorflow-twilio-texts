
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

//code from https://github.com/tensorflow/tfjs-examples/blob/482226b15a757f39871038f35b3b8aad7729e594/sentiment/index.js
//make sequences the same length. pre-padding is the default
const padSequences = (sequences, metadata) => {
  return sequences.map(seq => {
    // sequences longer than metadata.max_len truncated at the start of sequence
    if (seq.length > metadata.max_len) {
      seq.splice(0, seq.length - metadata.max_len);
    }
    // sequences shorter than metadata.max_len padded before sequence
    if (seq.length < metadata.max_len) {
      const pad = [];
      for (let i = 0; i < metadata.max_len - seq.length; ++i) {
        pad.push(0);
      }
      seq = pad.concat(seq);
    }
    return seq;
  });
}

const predict = (text, model, metadata) => {
  //tokenize text: remove non-alphanumeric chars besides spaces, apos
  var trimmed = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  //Convert the words to a sequence of word indices so we can truncate and pad
  const sequence = trimmed.map(word => {
    let wordIndex = metadata.word_index[word] + metadata.index_from;
    if (wordIndex > metadata.vocabulary_size) {
      wordIndex = 2; //oov_index
    }
    console.log(`wordIndex ${wordIndex}`);
    return wordIndex;
  });
  console.log(`sequence ${sequence}`);
  // Perform truncation and padding.
  const paddedSequence = padSequences([sequence], metadata);
  console.log(`paddedSeq ${paddedSequence}`);
  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  return score;
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