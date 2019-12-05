
require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

const fetch = require("node-fetch");
client = require("twilio")(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

async function setup() {
  var texts = [];
   await client.messages
    .list({
      dateSentAfter: new Date(Date.UTC(2019, 0, 1, 0, 0, 0)),
      dateSentBefore: new Date(Date.UTC(2019, 10, 31, 0, 0, 0)),
      from: "+16507878004",
      limit: 500
    })
     .then(messages => messages.forEach(m => texts.push(m.body)));
  return Array.from(new Set(texts)); //only unique texts
  }


function getSentiment(score) {
  switch(true) {
    case(score > 0.66):
      return `Score of ${score} is Positive`;
    case (score > 0.4):
      return `Score of ${score} is Neutral`;
    default:
      return `Score of ${score} is Negative`;
  }
}

function predict(text, model, metadata) {
  var trimmed = text.replace(/[|&;$%@"<>()+,$|\d]^\s*[·]+\s*\b/g, "").trim().split("\n");
  //not in dictionary, TF returns error
  const inputBuffer = tf.buffer([1, metadata.max_len], "float32");
  try {
    trimmed.forEach((word, i) => inputBuffer.set(metadata.word_index[word] + metadata.index_from, 0, i));
    const input = inputBuffer.toTensor(); //returns Tensor obj
    const predictOut = model.predict(input);
    let positivity = predictOut.dataSync()[0];
    predictOut.dispose();
    return positivity;
  }
  catch {
    return 0; //word/item passed in not in dictionary, return 0
  }
  
}

const getMetaData = async () => {
  const dt = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
  return dt.json()
}
function getAvg(arr) {
  const total = arr.reduce((partial_sum, a) => partial_sum + a, 0); 
  console.log("total ", total, " len arr ", arr.length);
  return total / arr.length;
}

async function run(text) {
  const u2 = "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"
  const model = await tf.loadLayersModel(u2);
  const metadata = await getMetaData();
  var avg;
  var posArr = [];
  text.forEach(function (prediction) {
    avg = predict(prediction, model, metadata);
    posArr.push(avg);
  });
  console.log(getSentiment(getAvg(posArr)));
}

setup().then(function(result) {
  var filtered = result
    .filter(function(el) {
      return el != "";
    })
    .map(v => v.toLowerCase()); 
  run(filtered);
});