<template lang="pug">
  div
    p
      | following the simple example from onnx.js offical docs:
      br
      a(href="https://github.com/Microsoft/onnxjs/tree/master/examples/node/add") https://github.com/Microsoft/onnxjs/tree/master/examples/node/add
    button(@click="run") run
    pre
      | {{ result }}
</template>

<script>
import { Tensor, InferenceSession } from 'onnxjs'

async function runInceptionV2() {
  // Creat the session and load the pre-trained model
  const session = new InferenceSession({ backendHint: 'webgl' });
  // await session.loadModel("/models/inception_v2/model.onnx");
  await session.loadModel("models/add.onnx");

  const x = new Float32Array(3 * 4 * 5).fill(1);
  const y = new Float32Array(3 * 4 * 5).fill(2);
  const tensorX = new Tensor(x, 'float32', [3, 4, 5]);
  const tensorY = new Tensor(y, 'float32', [3, 4, 5]);

  // Run model with Tensor inputs and get the result by output name defined in model.
  const outputMap = await session.run([tensorX, tensorY]);
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
    }
  },

  methods: {
    run () {
      console.log("running")
      runInceptionV2().then(result => this.result = result)
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
</style>
