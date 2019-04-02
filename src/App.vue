<template>
  <div id="app">
    <div id="nav">
      <h3>From PyTorch to ONNX.js</h3>
      <router-link to="/add">Add</router-link> |
      <router-link to="/embedding-example">Embeddings</router-link> |
      <router-link to="/about">About</router-link>
    </div>
    <div class="container">
      <router-view/>
    </div>
  </div>
</template>

<style lang="scss">
body {
  background-color: #44444d;
  color: #ddd;
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
a {
  font-weight: bold;
  color: #90b9e2;
  &.router-link-exact-active {
    color: #7391af;
  }
}
#nav {
  padding: 30px;
}
.container {
  margin: 0 30px;
}
</style>
