const tf = require("@tensorflow/tfjs");
const fetch = require("node-fetch");
const client = require("twilio")(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

const setup = async () => {
  const messages = await client.messages.list({
    dateSentAfter: new Date(Date.UTC(2019, 0, 1, 0, 0, 0)),
    dateSentBefore: new Date(Date.UTC(2019, 11, 31, 0, 0, 0)),
    from: "process.env.MY_PHONE_NUMBER"
  });
  return messages.map(m => m.body);
}

const getMetaData = async () => {
  const metadata = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
  return metadata.json()
}
const padSequences = (sequences, metadata) => {
  return sequences.map(seq => {
    if (seq.length > metadata.max_len) {
      seq.splice(0, seq.length - metadata.max_len);
    }
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
  const trimmed = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  const sequence = trimmed.map(word => {
    const wordIndex = metadata.word_index[word];
    if (typeof wordIndex === 'undefined') {
      return 2; //oov_index
    }
    return wordIndex + metadata.index_from;
  });
  const paddedSequence = padSequences([sequence], metadata);
  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  console.log(`text: ${text} has score of ${score}`)
  return score;
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
const loadModel = async () => {
  const url = `https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json`;
  const model = await tf.loadLayersModel(url);
  return model;
};
const run = async (text) => {
  const metadata = await getMetaData();
  const model = await loadModel();
  var sum = 0;
  text.forEach(function (prediction) {
    console.log(` ${prediction}`);
    perc = predict(prediction, model, metadata);
    sum += parseFloat(perc, 10);
  })
  console.log(getSentiment(sum / text.length));
}
setup().then(function (result) {
  run(result);
});
// run([]); //to test on a string

