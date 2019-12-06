
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
    if(score > 0.66) {
      return `Score of ${score} is Positive`;
    } 
    else if(score > 0.4) {
      return `Score of ${score} is Neutral`;
    }
    else {
      return `Score of ${score} is Negative`;
    }
}

function predict(text, model, metadata) {
  var trimmed = text.replace(/[|&;$%@"<>()+,$|\d]^\s*[·]+\s*\b/g, "").trim().split("\n");
  //not in dictionary, TF returns error
  const inputBuffer = tf.buffer([1, metadata.max_len], "float32");
  console.log(`trimmed ${trimmed}`);
    trimmed.forEach((word, i) => inputBuffer.set(metadata.word_index[word] + metadata.index_from, 0, i));
    const input = inputBuffer.toTensor(); //returns Tensor obj
    const predictOut = model.predict(input);
    let positivity = predictOut.dataSync()[0];
    predictOut.dispose();
    console.log(`try ${positivity}`);
    if(isNaN(positivity)) {
      return  0;
    }
    return positivity.toFixed(3);
  
}

const getMetaData = async () => {
  const dt = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
  return dt.json()
}

async function run(text) {
  const u2 = "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"
  const model = await tf.loadLayersModel(u2);
  const metadata = await getMetaData();
  var sum = 0.000;
  
  text.forEach(function (prediction) {
    avg = predict(prediction, model, metadata);
    sum += parseFloat(avg, 10);
  })
  console.log(getSentiment(sum/text.length));
}

setup().then(function(result) {
  var filtered = result
    .filter(function(el) {
      return el != "";
    })
    .map(v => v.toLowerCase()); 
  run(filtered);
});