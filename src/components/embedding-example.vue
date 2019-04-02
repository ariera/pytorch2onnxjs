<template lang="pug">
  div
    p
      | Trained an embedding on pytorch, following official docs
      a(href="https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html") [1]
      br
      | Exported following again the offical docs
      a(href="https://pytorch.org/docs/stable/onnx.html") [2]
    ol
      li
        a(href="https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html") https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
      li
        a(href="https://pytorch.org/docs/stable/onnx.html") https://pytorch.org/docs/stable/onnx.html

    h3 Learned text
    pre
      | When forty winters shall besiege thy brow,
      | And dig deep trenches in thy beauty's field,
      | Thy youth's proud livery so gazed on now,
      | Will be a totter'd weed of small worth held:
      | Then being asked, where all thy beauty lies,
      | Where all the treasure of thy lusty days;
      | To say, within thine own deep sunken eyes,
      | Were an all-eating shame, and thriftless praise.
      | How much more praise deserv'd thy beauty's use,
      | If thou couldst answer 'This fair child of mine
      | Shall sum my count, and make my old excuse,'
      | Proving his beauty by succession thine!
      | This were to be new made when thou art old,
      | And see thy blood warm when thou feel'st it cold.

    label(name="context")
      | Enter 2 words and the 3rd will be predicted
      |
      input(v-model="userInput")
    button(@click="run") predict
    p.has-success(v-if="prediction") {{ prediction }}
    pre.has-success(v-if="rawResult")
      | {{ rawResult }}
    pre.has-error(v-if="error")
      | {{ error }}
</template>

<script>
import { Tensor, InferenceSession } from 'onnxjs'
import axios from 'axios'

async function runInceptionV2(input) {
  // Creat the session and load the pre-trained model
  const session = new InferenceSession({ backendHint: 'webgl' });
  // await session.loadModel("/models/inception_v2/model.onnx");
  await session.loadModel("models/embeddings.onnx");

  // const x = new Float32Array(3 * 4 * 5).fill(1);
  // const y = new Float32Array(3 * 4 * 5).fill(2);
  // const tensorX = new Tensor(x, 'float32', [3, 4, 5]);
  // const tensorY = new Tensor(y, 'float32', [3, 4, 5]);

  const tensorX = new Tensor(new Int32Array(input), 'int32', [2])
  // Run model with Tensor inputs and get the result by output name defined in model.
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get("target word")

  // Check if result is expected.
  // assert.deepEqual(outputData.dims, [3, 4, 5]);
  // assert(outputData.data.every((value) => value === 3));
  console.log(`Got an Tensor of size ${outputData.data.length} with all elements being ${outputData.data[0]}`);
  return outputData
}

function restult2word(results, word_to_ix) {
  const sorted = Object.keys(results).sort((a, b) => -(results[a] - results[b]))
  const predicted_word_idx = sorted[0]
  const ix_to_word = Object.keys(word_to_ix).reduce((obj,key) => {
    obj[ word_to_ix[key] ] = key
    return obj
  },{})
  return ix_to_word[parseInt(predicted_word_idx)]
}

export default {
  data () {
    return {
      rawResult: null,
      error: null,
      userInput: 'When forty',
      prediction: ''
    }
  },

  mounted () {
    axios.get('models/word_to_ix.json')
      .then(response => this.word_to_ix = response.data)
  },

  methods: {
    run () {
      this.prediction = null
      this.rawResult = null
      console.log("running")
      const words = this.userInput.split(" ")
      if (words.length !== 2) {
        this.error = "please introduce 2 and only 2 words"
        return
      }
      console.debug(words[0], words[1])
      const context_idxs = [this.word_to_ix[words[0]], this.word_to_ix[words[1]]]
      if (context_idxs.some(idx => idx === undefined)) {
        this.error = "please introduce only words from the vocabulary"
        return
      }

      runInceptionV2(context_idxs)
        .then((result) => {
          this.rawResult = result
          this.prediction = restult2word(result.data, this.word_to_ix)
        })
        .catch(e => this.error = e )
    }
  }
}
</script>

<style scoped>
  pre {
    padding: 20px;
    background-color: #ccc;
    color: #444;
  }
  .has-success {
    background-color: rgb(143, 201, 165);
    color: #444;
  }
  .has-error {
    background-color: rgb(238, 138, 138);
    color: #444;
  }
</style>
