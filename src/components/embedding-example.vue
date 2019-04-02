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
    button(@click="run") run
    pre(v-if="result")
      | {{ result }}
    pre.has-error(v-if="error")
      | {{ error }}
</template>

<script>
import { Tensor, InferenceSession } from 'onnxjs'

async function runInceptionV2() {
  // Creat the session and load the pre-trained model
  const session = new InferenceSession({ backendHint: 'webgl' });
  // await session.loadModel("/models/inception_v2/model.onnx");
  await session.loadModel("models/embeddings.onnx");

  // const x = new Float32Array(3 * 4 * 5).fill(1);
  // const y = new Float32Array(3 * 4 * 5).fill(2);
  // const tensorX = new Tensor(x, 'float32', [3, 4, 5]);
  // const tensorY = new Tensor(y, 'float32', [3, 4, 5]);

  const tensorX = new Tensor(new Float32Array([1.0,2.0]), 'float32', [2,2])
  // Run model with Tensor inputs and get the result by output name defined in model.
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('sum');

  // Check if result is expected.
  // assert.deepEqual(outputData.dims, [3, 4, 5]);
  // assert(outputData.data.every((value) => value === 3));
  console.log(`Got an Tensor of size ${outputData.data.length} with all elements being ${outputData.data[0]}`);
  return outputData
}

export default {
  data () {
    return {
      result: null,
      error: null,
    }
  },

  methods: {
    run () {
      console.log("running")
      runInceptionV2()
        .then(result => this.result = result)
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
  .has-error {
    background-color: rgb(238, 138, 138);
    color: #444;
  }
</style>
